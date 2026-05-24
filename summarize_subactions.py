import torch
import os
import torch.nn.functional as F
import numpy as np
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_start_end_actions(labels, ignore_idx="BG"):
    
    pre_label = None

    steps = []
    timestamps = []
    st, ed = 0, 0
    
    for i in range(len(labels)):
        label = labels[i]

        if pre_label is None:
            pre_label = label
        
        if pre_label != label:
            if pre_label != ignore_idx:
                steps.append(pre_label)
                timestamps.append([st, ed])
            st = ed
            pre_label = label
        
        ed += 1

    if pre_label != ignore_idx:
        steps.append(pre_label)
        timestamps.append([st, ed])

    return steps, timestamps

def llm_check_exist(model, tokenizer, past_subaction, subaction):
    messages = [
        {
            "role": "system", 
            "content": "You are a system that given a list of actions and a target action, answer the following question: does the target action highly match any action in the list and explain why? Focus the verb, objects, and preposition. Objects with different colors are considered matched. Output yes or no and the reason. The output format: Yes/No\nReason:..."
        },
        {
            "role": "user", 
            "content": "List of actions:\n" + (past_subaction) + "\nTarget action: " + subaction
        }
    ]
    # print("List of actions:\n" + (past_subaction) + "\nTarget action: " + subaction)
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

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def llm_check_reasonable(model, tokenizer, target_action, past_subaction, subaction):
    messages = [
        {
            "role": "system", 
            "content": "You are a system that given a list of existing subactions and a new subaction, answer the following question: does the new subaction make any conflict with the existing subactions or the main action? The conflict indicates if the new subaction changes the objects used by the existing subactions or the object specified by the main action. Output yes or no and the reason. The output format: Yes/No\nReason:..."
        },
        {
            "role": "user", 
            "content": "Main action: " + target_action + "\nList of subactions:\n" + past_subaction + "\nNew subaction: " + subaction
        }
    ]
    # print("List of actions:\n" + (past_subaction) + "\nTarget action: " + subaction)
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

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def llm_check_reasonable2(model, tokenizer, target_action, subaction):
    messages = [
        {
            "role": "system", 
            "content": "You are a system that given a main action, and a new subaction, answer the following question: is the new subaction a appropriate subaction of the main action? Appropriate means the object used in the subaction has similar shape or purpose as the one used in the main action. Output yes or no and the reason. The output format: Yes/No\nReason:..."
        },
        {
            "role": "user", 
            "content": "Main action: " + target_action + "\nNew subaction: " + subaction
        }
    ]
    # print("List of actions:\n" + (past_subaction) + "\nTarget action: " + subaction)
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

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tea")
    parser.add_argument("--dataset", type=str, default="EgoPER")
    args = parser.parse_args()

    dataset = args.dataset
    task = args.task

    # ============================= variables ====================================
    with open(os.path.join("clust_config.json"), "r") as fp:
        num_clusters_dict = json.load(fp)[dataset]
    
    input_dir = f"./output/{dataset}/{task}/subactions_{num_clusters_dict[task]}_clust"
    obj_state_input_dir = f"./output/{dataset}/{task}/obj_state_{num_clusters_dict[task]}_clust"
    output_dir = f"./output/{dataset}/{task}/summarized_subactions_{num_clusters_dict[task]}_clust"

    BG = "the person is waiting."

    # only select first 10 object states
    select_obj_state = 10 

    model_name = "Qwen/Qwen2.5-32B-Instruct"
    # ============================= variables ====================================

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #### clean old data
    os.system("rm -rf %s/*" % (output_dir))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    #######################################
    # generate substeps
    #######################################
    with open(os.path.join("./data", dataset, "action2idx.json"), "r") as fp:
        action2idx = json.load(fp)[task]

    with open(os.path.join(f"./data/clean_action_dict.json"), "r") as fp:
        clean_action_dict = json.load(fp)

    for target_action, idx in action2idx.items():

        if target_action in clean_action_dict:
            target_action = clean_action_dict[target_action]
        
        if target_action == "BG":
            continue
        
        print("="*20)
        print("Target action:", target_action)

        if not os.path.exists(os.path.join(output_dir, target_action)):
            os.mkdir(os.path.join(output_dir, target_action))
        
        output_dict = {}
        past_subaction = ""
        subact_idx = 1
        
        if not os.path.exists(os.path.join(input_dir, target_action + ".json")):
            with open(os.path.join(output_dir, target_action, "output.json"), "w") as fp:
                json.dump(output_dict, fp, indent=4)
            print("No subactions for", target_action, ", then skip...")
            continue
        
        #########################
        # object summarization
        #########################

        with open(os.path.join(obj_state_input_dir, target_action+".json"), "r") as fp:
            idx2subactions = json.load(fp)
        
        
        object_states = []
        for _, elements in idx2subactions.items():
            subactions = elements["subactions"]
            # only select first 10 instances
            if len(subactions) > select_obj_state:
                object_states.extend(subactions[0:select_obj_state])
            else:
                object_states.extend(subactions)
        
        messages = [
            {
                "role": "system", 
                "content": "You are a system that given a lsit of object states, outputs at most five objects that most frequently appear. Select one of the output formats in the following: object1\nobject1,object2\nobject1,object2,object3\nobject1,object2,object3,object4\nobject1,object2,object3,object4,object5\n"
            },
            {
                "role": "user", 
                "content": "List of object states:\n" + ("\n".join(object_states))
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

        object_names = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        #########################
        # subaction summarization
        #########################
        
        with open(os.path.join(input_dir, target_action+".json"), "r") as fp:
            idx2subactions = json.load(fp)
        
        for _, elements in idx2subactions.items():
            subactions = elements["subactions"]
            messages = [
                {
                    "role": "system", 
                    "content": "You are a system that given a list of actions, outputs only one sentence to describe the action that most commonly occurs. You do not need to output your reason. Focus on the following objects if they are in the actions: %s. Select one of the output formats in the following: The person is [verb] [object1].\nThe person is [verb] [object1] [prep1] [object2].\nThe person is [verb] [object1] and [object2].\nThe person is [verb] [object1] [prep1] [object2] [prep2] [object3]." % object_names
                },
                {
                    "role": "user", 
                    "content": "List of actions:\n" + ("\n".join(subactions))
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

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


            '''
            check if should add the subaction into the real subaction list
            '''
            is_adding = False
            if past_subaction != "": # if is not empty, then use LLM to verify repetition
                '''
                check if the summarzied subaction exists
                '''
                is_exist_reason = llm_check_exist(model, tokenizer, past_subaction, response)
                if len(is_exist_reason.split("\n")) != 2: # llm failure, then add
                    is_adding = True
                else:
                    is_exist, reason = is_exist_reason.split("\n")
                    if is_exist.lower() == "no":
                        is_adding = True
                    
                    print("-"*20)
                    print("Existing subactions:\n", past_subaction)
                    print()
                    print("Does %s exist? %s. %s\n" % (response, is_exist, reason))
            else:
                is_adding = True
            
            if is_adding:
                '''
                check if the summarzied subaction has a conflict with target action
                '''
                is_reasonable_reason = llm_check_reasonable(model, tokenizer, target_action, past_subaction, response)
                try:
                    is_conflict, reason = is_reasonable_reason.split("\n")
                    print("Does '%s' make a conflict? %s. %s\n" % (response, is_conflict, reason))
                    if is_conflict.lower() == "yes":
                        is_adding = False
                except: 
                    print("LLM failed in reasoning subaction....................")
                    print("Skip this process..................")
            
            if is_adding:
                '''
                check the the summarzied subaction is appropriate for this target action
                '''
                is_reasonable_reason2 = llm_check_reasonable2(model, tokenizer, target_action, response)
                try:
                    is_proper, reason = is_reasonable_reason2.split("\n")
                    print("Is '%s' appropriate in '%s'? %s. %s\n" % (response, target_action, is_proper, reason))
                    if is_proper.lower() == "no":
                        is_adding = False
                except: 
                    print("LLM failed in reasoning subaction....................")
                    print("Skip this process..................")

            if past_subaction == "" or is_adding:
                past_subaction += response+"\n"

                output_dict[int(subact_idx)] = {
                    "summarized_subaction": response,
                    "others": elements
                }

                subact_idx += 1
        
        with open(os.path.join(output_dir, target_action, "output.json"), "w") as fp:
            json.dump(output_dict, fp, indent=4)
        
        print("="*20)
