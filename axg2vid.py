import torch
import math
import os
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.dag import lexicographical_topological_sort
from graph_utils import compute_generalized_metadag_costs, generalized_metadag2vid
import argparse
import time
from sentence_transformers import SentenceTransformer

'''
The graph2vid is using the implementation by this paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_Error_Recognition_in_Procedural_Videos_using_Generalized_Task_Graph_ICCV_2025_paper.pdf
'''


def mode_filter(x, window_size=10):
    assert window_size >= 1, "Window size must be at least 1"
    assert isinstance(window_size, int), "Window size must be an integer"
    
    n = len(x)
    filtered = np.zeros_like(x, dtype=int)
    
    for i in range(n):
        start = max(0, i - window_size // 2 + 1)
        end = min(n, i + window_size // 2)
        filtered[i] = np.bincount(x[start:end]).argmax()
    
    return filtered


def generate_start_end_actions(labels, ignore_idx=0):
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tea")
    parser.add_argument("--dataset", type=str, default="EgoPER")
    parser.add_argument(
        "--tas_backbone",
        choices=["gt", "gtg2vid", "fact", "egoped"],
        default="gt",
        help="Choose tas backbone or use GT tas.",
    )
    parser.add_argument("--label_path", type=str, default="labels_10fps")
    args = parser.parse_args()

    dataset = args.dataset
    task = args.task
    split = "test"
    tas_backbone = args.tas_backbone
    label_path = args.label_path

    # ============================= variables ====================================
    
    with open(os.path.join("clust_config.json"), "r") as fp:
        num_clusters_dict = json.load(fp)[dataset]
    
    # num of consecutive frames for BG (3 seconds)
    bg_ncf = 5
    # num of consecutive frame for action (2 seconds)
    action_ncf = 3 
    dfr = 0.0

    subaction_dir = f"./output/{dataset}/{task}/summarized_subactions_{num_clusters_dict[task]}_clust"
    subaction_feature_dir = f"./output/{dataset}/{task}/v_subaction_features_{num_clusters_dict[task]}_clust"
    frame_feature_dir = f"./output/{dataset}/{task}/vc_v_features_10fps"
    # frame_feature_dir = f"/data/shihpo/{dataset}/{task}/vc_v_features_10fps"
    axg_dir = f"./output/{dataset}/{task}/axg_{num_clusters_dict[task]}_clust"
    action_dir = f"./data/{dataset}/{task}/action_object_state"
    output_dir = f"./output/{dataset}/{task}/data_for_vlm_{num_clusters_dict[task]}_clust_{tas_backbone}"
    tas_result_dir = f"./tas_output/{dataset}/{task}/{tas_backbone}"
    
    BG = "the person is waiting."

    # ============================= variables ====================================

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join("./data", dataset, "action2idx.json"), "r") as fp:
        action2idx = json.load(fp)[task]
        action2idx["Error"] = '-1'

    idx2action = {}
    for k, v in action2idx.items():
        idx2action[str(v)] = k
    
    with open(os.path.join(f"./data/clean_action_dict.json"), "r") as fp:
        clean_action_dict = json.load(fp)

    with open(os.path.join("./data", dataset, task, split+".txt"), "r") as fp:
        filenames = fp.readlines()
    



    #######################################
    # buliding stepgraph
    #######################################
    
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")


    for name in filenames:

        vid = name.strip("\n")
        all_data = {}

        if args.tas_backbone == "gt":
            with open(os.path.join("./data", dataset, task, label_path, vid + ".txt"), "r") as fp:
                lines = fp.readlines()
            tas = []
            for i in range(len(lines)):
                tokens = lines[i].split("|")
                action, action_type = tokens[0], tokens[1]
                if "Addition" in action_type:
                    tas.append(1) # randomly assign an action to addition error
                else:
                    if action not in action2idx: # for captaincook4d
                        tas.append(1) # randomly assign an action to addition error
                    else:
                        tas.append(int(action2idx[action]))
                
            tas = np.array(tas)
            actions, timestamps = generate_start_end_actions(tas)
            for i in range(len(actions)):
                actions[i] = idx2action[str(actions[i])]
        else: # get Predicted TAS results, without drop, without smoothing
            with open(os.path.join(tas_result_dir, vid+".txt"), "r") as fp:
                tas = fp.readlines()
            new_tas = []
            for t in tas:
                new_tas.append(int(t.strip("\n")))
            # smoohting
            tas = mode_filter(new_tas)
            actions, timestamps = generate_start_end_actions(tas)
            for i in range(len(actions)):
                actions[i] = idx2action[str(actions[i])]

        all_data["tas"] = tas.tolist()
        all_data["action"] = []
        all_data_idx = 0

        if dataset == "CaptainCook4D":
            vid_features = torch.from_numpy(np.load(os.path.join(frame_feature_dir, vid + "_360p.npy")))
        else:
            vid_features = torch.from_numpy(np.load(os.path.join(frame_feature_dir, vid + ".npy")))

        for action, timestamp in zip(actions, timestamps):

            if action in clean_action_dict:
                action = clean_action_dict[action]

            with open(os.path.join(subaction_dir, action, "output.json"), "r") as fp:
                subaction_dict = json.load(fp)
            
            subaction_embedding = []
            idx2subaction = {}
            for subaction_idx, elements in subaction_dict.items():
                idx2subaction[int(subaction_idx)] = elements["summarized_subaction"]
                subaction_embedding.append(torch.from_numpy(np.load(os.path.join(subaction_feature_dir, action, str(subaction_idx)+'.npy'))))

            start, end = timestamp
            captions = []
            caption_embedding = []
            caption_paths = []
            for t in range(start, end): #end+1):
                if os.path.exists(os.path.join(action_dir, vid, "%06d.txt" % t)):
                    with open(os.path.join(action_dir, vid, "%06d.txt" % t)) as fp:
                        # captions.append(fp.read().strip("\n"))
                        #### for object state
                        all_text = fp.read()
                        captions.append(all_text.split(".")[0]) # only the first sentence is subaction
                        caption_paths.append(os.path.join(action_dir, vid, "%06d.txt" % t))
                    
                    caption_embedding.append(vid_features[t])

            if len(caption_embedding) == 0: # too short action, skip it:
                print("Too short for", action, ", then skip it...")
                continue

            all_data["action"].append({})
            all_data["action"][all_data_idx]["name"] = action
            all_data["action"][all_data_idx]["idx2subaction"] = idx2subaction
            all_data["action"][all_data_idx]["start_end"] = [start, end]
            # all_data["action"][all_data_idx]["is_error"] = True if ed[start+1] == 1 else False
            all_data["action"][all_data_idx]["captions"] = captions
            all_data["action"][all_data_idx]["caption_paths"] = caption_paths

            ############# using SBERT
            # subactions = []
            # for idx, subaction in idx2subaction.items():
            #     subactions.append(subaction)
            # subaction_embedding = model.encode(subactions)
            # caption_embedding = model.encode(captions)

            # using VLM
            subaction_embedding = torch.stack(subaction_embedding)
            caption_embedding = torch.stack(caption_embedding)

            sims = model.similarity(caption_embedding, subaction_embedding) # N, K, where N is number of frames and K is number of subactions
            
            N, K = sims.shape

            ########################################################
            # make all thread has the same start node and end node
            ########################################################
            # add first and last frames
            new_sims = torch.cat([torch.zeros((1, K)), sims, torch.zeros((1, K))], axis=0)

            # add start and end nodes
            sims = torch.cat([torch.zeros((N + 2, 1)), new_sims, torch.zeros((N + 2, 1))], axis=1)

            sims = (sims + 1) / 2

            # assign similarities for start node/first frame and end node/last frame
            sims[0, 0] = 1.0
            sims[-1, -1] = 1.0

            ########################################################
            
            node_dict = {}
            with open(os.path.join(axg_dir, action + ".json"), "r") as fp:
                edges_info = json.load(fp)

            start = edges_info[0][0][0]
            end = edges_info[0][-1][1]

            StepGraph = nx.DiGraph()

            StepGraph.add_node(str(start))
            StepGraph.add_node(str(end))

            for thread in edges_info:
                current_nodes = []
                for n1, n2 in thread:
                    if n1 not in current_nodes and n1 != start:
                        current_nodes.append(n1)
                    if n2 not in current_nodes and n2 != end:
                        current_nodes.append(n2)
                
                for n in current_nodes:
                    if n not in node_dict:
                        node_dict[n] = 0
                    else:
                        node_dict[n] += 1

                for n1, n2 in thread:
                    if n1 == start:
                        StepGraph.add_edge(str(n1), str(n2)+","+str(node_dict[n2]))
                    elif n2 == end:
                        StepGraph.add_edge(str(n1)+","+str(node_dict[n1]), str(n2))
                    else:
                        StepGraph.add_edge(str(n1)+","+str(node_dict[n1]), str(n2)+","+str(node_dict[n2]))

            sorted_node_ids = list(lexicographical_topological_sort(StepGraph))
            idx2node = {idx: node_id for idx, node_id in enumerate(sorted_node_ids)}


            zx_costs, drop_costs, node_drop_costs = compute_generalized_metadag_costs(sims, idx2node, dfr, -100)
            _, pred, type_pred = generalized_metadag2vid(zx_costs.cpu().numpy(), drop_costs.cpu().numpy(), node_drop_costs.cpu().numpy(), StepGraph, idx2node)
            # remove first and last subaction, as they are source and sink
            pred = pred[1:-1]

            # if only single action between two drops (-1), then assign that action to -1
            new_pred = pred
            for pred_i in range(1, len(pred)-2):
                if pred[pred_i - 1] == -1 and pred[pred_i + 1] == -1:
                    new_pred[pred_i] = -1
                elif pred[pred_i - 1] == -1 and pred[pred_i + 2] == -1:
                    new_pred[pred_i] = -1
                    new_pred[pred_i + 1] = -1 
            pred = new_pred

            pre_sub = None
            idx = 0
            start = 0
            num_consecutive_frames = 0
            
            all_data["action"][all_data_idx]["subaction_list"] = pred.tolist()
            all_data["action"][all_data_idx]["subactions"] = []
            all_data_sub_idx = 0
            for sub in pred:
                if pre_sub is None:
                    pre_sub = sub

                # normal action
                if pre_sub != sub and pre_sub in idx2subaction and num_consecutive_frames > action_ncf:
                    key_captions = []
                    key_caption_idx = []
                    print("Subaction %d:"%(pre_sub), idx2subaction[pre_sub])
                    print("Caption: [%d,%d]" % (start, idx))
                    print("[")
                    for i in range(start, idx):
                        print(captions[i])
                        key_captions.append(captions[i])
                        key_caption_idx.append(i)
                    print("]")
                    start = idx
                    num_consecutive_frames = 0
                    all_data["action"][all_data_idx]["subactions"].append({})
                    all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["name"] = idx2subaction[pre_sub]
                    all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["caption_idx"] = key_caption_idx
                    all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["captions"] = key_captions

                    all_data_sub_idx += 1
                # BG with more than bg_ncf frames
                elif pre_sub != sub and pre_sub == -1 and num_consecutive_frames >= bg_ncf:
                    key_captions = []
                    key_caption_idx = []
                    print("Subaction -1: BG")
                    print("Caption: [%d,%d]" % (start, idx))
                    print("[")
                    for i in range(start, idx):
                        print(captions[i])
                        key_captions.append(captions[i])
                        key_caption_idx.append(i)
                    print("]")
                    start = idx
                    num_consecutive_frames = 0
                    all_data["action"][all_data_idx]["subactions"].append({})
                    all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["name"] = "BG"
                    all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["caption_idx"] = key_caption_idx
                    all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["captions"] = key_captions

                    all_data_sub_idx += 1
                elif pre_sub != sub:
                    start = idx
                    num_consecutive_frames = 0

                idx += 1
                num_consecutive_frames += 1
                pre_sub = sub
            

            if pre_sub in idx2subaction and num_consecutive_frames >= action_ncf:
                key_captions = []
                key_caption_idx = []
                print("Subaction:", idx2subaction[pre_sub])
                print("Caption: [%d,%d]" % (start, idx))
                print("[")
                for i in range(start, idx):
                    print(captions[i])
                    key_captions.append(captions[i])
                    key_caption_idx.append(i)
                print("]")
                all_data["action"][all_data_idx]["subactions"].append({})
                all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["name"] = idx2subaction[pre_sub]
                all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["caption_idx"] = key_caption_idx
                all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["captions"] = key_captions

            elif pre_sub == -1 and num_consecutive_frames >= bg_ncf:
                key_captions = []
                key_caption_idx = []
                print("Subaction -1: BG")
                print("Caption: [%d,%d]" % (start, idx))
                print("[")
                for i in range(start, idx):
                    print(captions[i])
                    key_captions.append(captions[i])
                    key_caption_idx.append(i)
                print("]")
                all_data["action"][all_data_idx]["subactions"].append({})
                all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["name"] = "BG"
                all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["caption_idx"] = key_caption_idx
                all_data["action"][all_data_idx]["subactions"][all_data_sub_idx]["captions"] = key_captions


            print("\n=========================\n")
            all_data_idx += 1

        with open(os.path.join(output_dir, vid+'.json'), "w") as fp:
            json.dump(all_data, fp, indent=4)