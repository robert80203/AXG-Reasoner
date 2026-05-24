# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import json
import numpy as np
import argparse

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def iou(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)

def sample_n(lst, n_samples=3):
    n = len(lst)
    if n_samples <= 0:
        return []
    if n_samples > n:
        raise ValueError("Number of samples cannot exceed list length.")

    part = n / n_samples
    samples = []
    for i in range(n_samples):
        start = int(i * part)
        end = int((i + 1) * part)
        mid = (start + end - 1) // 2
        samples.append(lst[mid])
    return samples

def subbg_reasoning(model, tokenizer, action):
    msg = "You are a system that given a image, outputs the action the person is doing and the detailed state for every object the person is working with. The person is in the process of doing '%s'. Do not use vague verbs such as prepare, manipulate, etc. If the person's hand is not visible, outputs 'the person is waiting.' Otherwise, select one of the output formats in the following to output the action: The person is [verb] [object1]. The state of [object1] is [state1].\nThe person is [verb] [object1] [prep1] [object2]. The state of [object1] is [state1]. The state of [object2] is [state2].\nThe person is [verb] [object1] and [object2]. The state of [object1] is [state1]. The state of [object2] is [state2].\nThe person is [verb] [object1] [prep1] [object2] [prep2] [object3]. The state of [object1] is [state1]. The state of [object2] is [state2]. The state of [object3] is [state3]." % action

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tea")
    parser.add_argument("--dataset", type=str, default="EgoPER")
    parser.add_argument(
        "--tas_backbone",
        choices=["gt", "gtg2vid"],
        default="gt",
        help="Choose tas backbone or use GT tas.",
    )
    parser.add_argument("--numf", type=int, default=3) # number of frames to sample
    args = parser.parse_args()

    task = args.task
    dataset = args.dataset
    tas_backbone = args.tas_backbone
    num_sampling_frames = args.numf #8 #3

    # ============================= variables ====================================
    with open(os.path.join("../clust_config.json"), "r") as fp:
        num_clusters_dict = json.load(fp)[dataset]

    split = "test"
    error_threshold = 0.5

    MODEL_PATH = "Qwen/Qwen2.5-VL-32B-Instruct"
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

    # 2 L40S
    # llm = LLM(
    #     model=MODEL_PATH,
    #     limit_mm_per_prompt={"image": 10, "video": 10},
    #     tensor_parallel_size=2, # use 2 gpus
    #     gpu_memory_utilization=0.9,
    #     max_model_len=32786 // 8, #32768,
    #     enable_prefix_caching=True,
    # )
    
    # 1 H100
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 12, "video": 12},
        gpu_memory_utilization=0.9,
        max_model_len=32768 // 4,
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        # max_tokens=512,
        max_tokens=256,
        top_k=-1,
        stop_token_ids=[],
    )
    # ============================= variables ====================================

    input_dir = f"../output/{dataset}/{task}/data_for_vlm_{num_clusters_dict[task]}_clust_{tas_backbone}"
    output_dir = f"../output/{dataset}/{task}/error_reasoning_{num_clusters_dict[task]}_clust_{tas_backbone}_{num_sampling_frames}f"
    # update it with your own frame directory
    frame_dir = f"/data/shihpo/{dataset}/{task}/frames_10fps"

    if not os.path.exists(output_dir):
        print("Create dir:", output_dir)
        os.mkdir(output_dir)

    with open(os.path.join("../data", dataset, task, split+".txt"), "r") as fp:
        filenames = fp.readlines()

    llm_inputs_all = []
    output_file_paths = []

    print("Start processing...")
    for name in filenames:
        vid = name.strip("\n")
        print(vid)
        with open(os.path.join(input_dir, vid+".json"), "r") as fp:
            all_data = json.load(fp)

        is_error = False
        text = "You are trying to do '%s'\nYou have finished the following steps:\n" % task
        action_text = ""
        action_list = []
        for action in all_data["action"]:
            # print(vid, action["name"])
            caption_paths = action["caption_paths"]
            action_name = action["name"]

            for subaction in action["subactions"]:
                subaction_name = subaction["name"]
                caption_idx = subaction["caption_idx"]
                if subaction_name == "BG": # all dropped frames are normal for now
                    '''
                    Do reasoning on subaction == BG
                    For now, dropped == error
                    '''
                    # if args.subbg:
                    ########## uniformly sample frames, using original frame rate
                    start_tokens = caption_paths[caption_idx[0]].split("/")
                    end_tokens = caption_paths[caption_idx[-1]].split("/")
                    data_vid_name = start_tokens[-2]
                    start_frame = int(start_tokens[-1][:-4])
                    end_frame = int(end_tokens[-1][:-4])


                    numbers = list(range(start_frame, end_frame))

                    samples = sample_n(numbers, num_sampling_frames)

                    current_action_text = "Now you are trying to do an action '%s' and it seems like you are making a mistake.\n" % (action_name)
                    output_text = "Given a sequence of images for '%s', use one sentence to describe what mistake you are making."

                    frame_paths = []
                    for i in samples:
                        # uniformly sample frames, using original frame rate
                        if dataset == "CaptainCook4D":
                            frame_paths.append(os.path.join(frame_dir, data_vid_name+"_360p", "%05d.jpg" % i))
                        else:
                            frame_paths.append(os.path.join(frame_dir, data_vid_name, "%06d.png" % i))

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": frame_paths,
                                },
                                {"type": "text", "text": text + action_text + current_action_text + output_text},
                            ],
                        }
                    ]

                    prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

                    mm_data = {}
                    if image_inputs is not None:
                        mm_data["image"] = image_inputs
                    if video_inputs is not None:
                        mm_data["video"] = video_inputs
                    llm_inputs = {
                        "prompt": prompt,
                        "multi_modal_data": mm_data,

                        # FPS will be returned in video_kwargs
                        "mm_processor_kwargs": video_kwargs,
                    }
                    llm_inputs_all.append(llm_inputs)

                else:
                    
                    # uniformly sample frames, using original frame rate
                    start_tokens = caption_paths[caption_idx[0]].split("/")
                    end_tokens = caption_paths[caption_idx[-1]].split("/")
                    data_vid_name = start_tokens[-2]
                    start_frame = int(start_tokens[-1][:-4])
                    end_frame = int(end_tokens[-1][:-4])

                    numbers = list(range(start_frame, end_frame))

                    samples = sample_n(numbers, num_sampling_frames)
                    
                    current_action_text = "Now you are trying to do an subaction '%s' of an action '%s'.\n" % (subaction_name, action_name)
                    
                    output_text = "Given a sequence of images for '%s', output a score to show the correctness of the action and your reason. The score ranges from 0 to 1. Higher score indicates the action shown by those images is more correct. The output format: Score: x\nReason: y" % subaction_name
                    
                    frame_paths = []
                    for i in samples:
                        if dataset == "CaptainCook4D":
                            frame_paths.append(os.path.join(frame_dir, data_vid_name+"_360p", "%05d.jpg" % i))
                        else:
                            frame_paths.append(os.path.join(frame_dir, data_vid_name, "%06d.png" % i))

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": frame_paths,
                                },
                                {"type": "text", "text": text + action_text + current_action_text + output_text},
                            ],
                        }
                    ]

                    prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

                    mm_data = {}
                    if image_inputs is not None:
                        mm_data["image"] = image_inputs
                    if video_inputs is not None:
                        mm_data["video"] = video_inputs
                    llm_inputs = {
                        "prompt": prompt,
                        "multi_modal_data": mm_data,

                        # FPS will be returned in video_kwargs
                        "mm_processor_kwargs": video_kwargs,
                    }
                    llm_inputs_all.append(llm_inputs)
            
            # should not include BG
            if action_name not in action_list:
                action_list.append(action_name)
                action_text += "%s\n" % action_name

    outputs = llm.generate(llm_inputs_all, sampling_params=sampling_params)
    output_idx = 0
    print(len(outputs), len(llm_inputs_all))
    for name in filenames:
        vid = name.strip("\n")
        # print(vid)
        with open(os.path.join(input_dir, vid+".json"), "r") as fp:
            all_data = json.load(fp)
        
        action_idx = 0
        
        for action in all_data["action"]:
            subaction_idx = 0
            for subaction in action["subactions"]:
                subaction_name = subaction["name"]
                is_error = False
                if subaction_name == "BG":
                    
                    '''
                    Do reasoning on subaction == BG
                    '''
                    er_result = outputs[output_idx].outputs[0].text 
                    is_error = True
                    error_score = 1.0
                    output_idx += 1

                else:
                    response = outputs[output_idx].outputs[0].text
                    
                    tokens = response.split("\n")
                    tokens_2 = response.split("\n\n")
                    if len(tokens) == 2:
                        score, er_result = tokens
                        score = float(score.split(": ")[1])
                        er_result = er_result.split(": ")[1]
                    elif len(tokens_2) == 2:
                        score, er_result = tokens_2
                        score = float(score.split(": ")[1])
                        er_result = er_result.split(": ")[1]
                    else: # special case when llm fails
                        score = 1
                        er_result = "llm failure, set it correct, output: " + response

                    error_score = 1 - score
                    if error_score >= error_threshold:
                        is_error = True
                    
                    output_idx += 1

                all_data["action"][action_idx]["subactions"][subaction_idx]["is_error"] = is_error
                all_data["action"][action_idx]["subactions"][subaction_idx]["error_score"] = error_score
                all_data["action"][action_idx]["subactions"][subaction_idx]["reasoning"] = er_result
                subaction_idx += 1
            action_idx += 1
        
        with open(os.path.join(output_dir, vid+".json"), "w") as fp:
            json.dump(all_data, fp, indent=4)
