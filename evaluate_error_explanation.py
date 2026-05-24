from PIL import Image
import os
import json
import numpy as np
import argparse
from utils.metrics import mstcn_f_score

try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    use_nlp_metrics = True
except:
    print("pycocoevalcap library is not installed. Skip it.")
    use_nlp_metrics = False


def generate_start_end_actions(labels, label_types, ignore_idx="BG"):
    
    pre_label = None
    pre_label_type = None

    steps = []
    step_types = []
    timestamps = []
    st, ed = 0, 0
    
    for i in range(len(labels)):
        label = labels[i]
        label_type = label_types[i]

        if pre_label is None:
            pre_label = label
            pre_label_type = label_type
            
        
        if pre_label != label:
            if pre_label != ignore_idx:
                steps.append(pre_label)
                step_types.append(pre_label_type)
                timestamps.append([st, ed])
            st = ed
            pre_label = label
            pre_label_type = label_type

        ed += 1

    if pre_label != ignore_idx:
        steps.append(pre_label)
        step_types.append(pre_label_type)
        timestamps.append([st, ed])

    return steps, step_types, timestamps


def evaluate_metrics(hypotheses, references):
    """
    hypotheses: list of strings (model outputs)
    references: list of list of strings (multiple refs per sample)
    """

    # Convert to dict format expected by pycocoevalcap
    gts = {}  # ground truth
    res = {}  # results
    for i, (h, r) in enumerate(zip(hypotheses, references)):
        gts[i] = r
        res[i] = [h]

    # BLEU (BLEU-1~BLEU-4)
    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(gts, res)

    # ROUGE-L
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)

    # CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)

    return {
        # "BLEU-1": bleu_score[0],
        # "BLEU-2": bleu_score[1],
        # "BLEU-3": bleu_score[2],
        # "BLEU-4": bleu_score[3],
        # "BLEU": (bleu_score[0] + bleu_score[1] + bleu_score[2] + bleu_score[3]) / 4,
        "BLEU-1": bleu_score[0],
        "ROUGE-L": rouge_score,
        "CIDEr": cider_score,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tea")
    parser.add_argument("--dataset", type=str, default="EgoPER")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--label_path", type=str, default="labels_10fps")
    parser.add_argument(
        "--tas_backbone",
        choices=["gt", "gtg2vid", "fact", "egoped"],
        default="gt",
        help="Choose tas backbone or use GT tas.",
    )
    parser.add_argument("--numf", type=int, default=3) # number of frames to sample
    args = parser.parse_args()

    if not args.eval:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "Qwen/Qwen2.5-32B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    task = args.task
    dataset = args.dataset
    label_path = args.label_path
    tas_backbone = args.tas_backbone
    num_sampling_frames = args.numf
    split = "test"

    # ============================= variables ====================================
    with open(os.path.join("clust_config.json"), "r") as fp:
        num_clusters_dict = json.load(fp)[dataset]
        
    overlap = 0.5 # this is only useful when the TAS is not GT

    output_dir = f"./output/{dataset}/{task}/error_explanation_results_{num_clusters_dict[task]}_clust_{tas_backbone}_{num_sampling_frames}f"
    input_dir = f"./output/{dataset}/{task}/error_reasoning_{num_clusters_dict[task]}_clust_{tas_backbone}_{num_sampling_frames}f"
                
    # directly read generated predictions
    if args.eval:
        input_dir = f"./output/{dataset}/{task}/error_explanation_results_{num_clusters_dict[task]}_clust_{tas_backbone}_{num_sampling_frames}f"
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    with open(os.path.join("./data", dataset, task, split+".txt"), "r") as f:
        filenames = f.readlines()
    
    with open(os.path.join(f"./data/clean_action_dict.json"), "r") as fp:
        clean_action_dict = json.load(fp)


        
    total_sim = []
    total_sim_ignore_missing = []

    references = []
    hypotheses = []
    hypotheses_all = []
    for name in filenames:
        vid = name.strip("\n")
        # print(vid)
        with open(os.path.join(input_dir, vid+".json"), "r") as f:
            all_data = json.load(f)
        
        with open(os.path.join("./data", dataset, task, label_path, vid+".txt"), "r") as f:
            lines = f.readlines()
        
        labels = []
        label_types = []
        for i in range(len(lines)):
            labels.append(lines[i].split("|")[0])
            if len(lines[i].split("|")) == 3:
                label_types.append("Error~"+lines[i].split("|")[2].strip("\n"))
            else:
                label_types.append("Normal~"+lines[i].split("|")[0])

        actions, action_types, timestamps = generate_start_end_actions(labels, label_types)
        

        for i in range(len(actions)):
            if actions[i] in clean_action_dict:
                actions[i] = clean_action_dict[actions[i]]
        
        y_start = []
        y_end = []
        for x, y in timestamps:
            y_start.append(x)
            y_end.append(y)

        for action in all_data["action"]:
            action_name = action["name"]
            start, end = action["start_end"]

            is_gt_action_error = False
            if "Error" in label_types[(start + end) // 2]:
                is_gt_action_error = True

            is_action_error = False
            for subaction in action["subactions"]:
                if subaction["is_error"]:
                    is_action_error = True
            
            intersection = np.minimum(end, y_end) - np.maximum(start, y_start)
            union = np.maximum(end, y_end) - np.minimum(start, y_start)
            IoU = (1.0*intersection / union)*([action_name == actions[x] for x in range(len(actions))])
            
            # # Get the best scoring segment
            idx = np.array(IoU).argmax()

            if IoU[idx] >= overlap:
                if args.eval:
                    if "semantic_reasoning_score" in action:
                        total_sim.append(action["semantic_reasoning_score"])
                        if action["semantic_reasoning"] != "Did not capture this error.":
                            total_sim_ignore_missing.append(action["semantic_reasoning_score"])
                    
                    action_des = labels[(start + end) // 2]
                    error_des = label_types[(start + end) // 2].split("~")[1]
                    if is_gt_action_error:
                        gt_reasoning = "The person is performing the action '%s'. However, the person is making a mistake which is '%s'." % (action_des, error_des)
                    else:
                        gt_reasoning = "The person is correctly performing the action '%s'" % (action_des)
                    

                    pred_reasoning = ""
                    if is_action_error and is_gt_action_error:
                        for subaction in action["subactions"]:
                            is_error = subaction["is_error"]
                            if is_error:
                                pred_reasoning += subaction["reasoning"] + ". "
                        hypotheses.append(pred_reasoning)
                        hypotheses_all.append(pred_reasoning)
                        references.append([gt_reasoning])
                    
                    elif not is_action_error and is_gt_action_error:
                        hypotheses.append("None")
                        for subaction in action["subactions"]:
                            pred_reasoning += subaction["reasoning"] + " "
                        hypotheses_all.append(pred_reasoning)
                        references.append([gt_reasoning])
                else:
                    if is_action_error and "Error" in action_types[idx]: # true positive
                        
                        pred_reasoning = ""
                        for subaction in action["subactions"]:
                            is_error = subaction["is_error"]
                            if is_error:
                                pred_reasoning += subaction["reasoning"] + "\n"

                        gt_type, gt_des = action_types[idx].split("~")
                        print("Correct action:", actions[idx], "Wrong action:", gt_des)
                        gt_reasoning = "The person is performing the action '%s'. However, the person is making a mistake which is '%s'." % (actions[idx], gt_des)

                        messages = [
                            {
                                "role": "system", 
                                "content": "You are a system that given a reference description and a target description, outputs a score that represents the semantic similarity between two descriptions and your reason. Higher score means the two descriptions are more similar in semantics. The score is from 0 to 1. Output format: Score: <score>\nReason: <reason>."
                            },
                            {
                                "role": "user", 
                                "content": "Reference description: " + gt_reasoning + "\nTarget description: " + pred_reasoning
                            }
                        ]
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                        generated_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=512
                        )
                        generated_ids = [
                            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]

                        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                        tokens = output.split("\n")
                        tokens_2 = output.split("\n")

                        if len(tokens) == 2:
                            action["semantic_reasoning_score"] = float(tokens[0].split(": ")[1])
                            action["semantic_reasoning"] = tokens[1].split(": ")[1]
                        elif len(tokens_2) == 2:
                            action["semantic_reasoning_score"] = float(tokens_2[0].split(": ")[1])
                            action["semantic_reasoning"] = tokens[1].split(": ")[1]
                        else:
                            action["semantic_reasoning_score"] = 0.0
                            action["semantic_reasoning"] = "Cannot parse the output."
                    
                    elif not is_action_error and "Error" in action_types[idx]: # false negative
                        action["semantic_reasoning_score"] = 0.0
                        action["semantic_reasoning"] = "Did not capture this error."

        if not args.eval:
            with open(os.path.join(output_dir, vid+".json"), "w") as f:
                json.dump(all_data, f, indent=4)

    if args.eval:
        print("Avg Reasoning Scores: %.2f" % np.array(total_sim).mean())
        print("Avg Reasoning Scores (ignore missing): %.2f" % np.array(total_sim_ignore_missing).mean())

        if use_nlp_metrics:
            scores = evaluate_metrics(hypotheses, references)
            scores_all = evaluate_metrics(hypotheses_all, references)
            output_dict = {
                "BLEU-1": 0,
                "ROUGE-L": 0,
                "CIDEr": 0
            }
            all_output_dict = {
                "BLEU-1": 0,
                "ROUGE-L": 0,
                "CIDEr": 0
            }
            for k, v in scores.items():
                output_dict[k] = v
            for k, v in scores_all.items():
                all_output_dict[k] = v

            print()
            print(f"|{'BLEU-1':^{15}}|{'ROUGEL':^{15}}|{'CIDEr':^{15}}|{'A-BLEU-1':^{15}}|{'A-ROUGEL':^{15}}|{'A-CIDEr':^{15}}|")
            print(f"|{'---':^{15}}|{'---':^{15}}|{'---':^{15}}|{'---':^{15}}|{'---':^{15}}|{'---':^{15}}|")
            print(f"|{output_dict['BLEU-1']:^{15}.4f}|{output_dict['ROUGE-L']:^{15}.4f}|{output_dict['CIDEr']:^{15}.4f}|{all_output_dict['BLEU-1']:^{15}.4f}|{all_output_dict['ROUGE-L']:^{15}.4f}|{all_output_dict['CIDEr']:^{15}.4f}|")
