import torch
import os
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.dag import lexicographical_topological_sort
from graph_utils import compute_generalized_metadag_costs, generalized_metadag2vid
from sentence_transformers import SentenceTransformer
import itertools
import argparse
import math


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

def sinusoidal_embedding(x, dim=256, scale=10000.0):
    """
    Compute sinusoidal embeddings for numeric inputs.
    
    Args:
        x (torch.Tensor): Tensor of shape [N] or [N, 1] containing scalar values.
        dim (int): Embedding dimension (must be even).
        scale (float): Frequency scaling factor (default = 10000.0)
    
    Returns:
        torch.Tensor: Sinusoidal embeddings of shape [N, dim].
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)  # [N, 1]
    device = x.device
    
    # Generate frequencies
    half_dim = dim // 2
    freq = torch.exp(-math.log(scale) * torch.arange(0, half_dim, device=device) / half_dim)
    
    # Outer product (broadcast): [N, half_dim]
    angles = x * freq.unsqueeze(0)
    
    # Apply sin and cos
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tea")
    parser.add_argument("--dataset", type=str, default="EgoPER")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--label_path", type=str, default="labels_10fps")
    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    dataset = args.dataset
    task = args.task
    split = args.split
    label_path = args.label_path

    # ============================= variables ====================================
    with open(os.path.join("clust_config.json"), "r") as fp:
        num_clusters_dict = json.load(fp)[dataset]

    input_dir = f"./data/{dataset}/{task}/action_object_state"
    frame_feature_input_dir = f"./data/{dataset}/{task}/vc_v_features_10fps"
    # frame_feature_input_dir = f"/data/shihpo/{dataset}/{task}/vc_v_features_10fps"

    summarized_subaction_dir = f"./output/{dataset}/{task}/summarized_subactions_{num_clusters_dict[task]}_clust"
    axg_output_dir = f"./output/{dataset}/{task}/axg_{num_clusters_dict[task]}_clust"
    subaction_feature_dir = f"./output/{dataset}/{task}/v_subaction_prog_features_{num_clusters_dict[task]}_clust"

    BG = "the person is waiting."

    # ============================= variables ====================================

    if not os.path.exists(axg_output_dir):
        os.mkdir(axg_output_dir)

    #######################################
    # generate substeps
    #######################################
    with open(os.path.join("./data", dataset, "action2idx.json"), "r") as fp:
        action2idx = json.load(fp)[task]

    with open(os.path.join("./data", dataset, task, split+".txt"), "r") as fp:
        filenames = fp.readlines()

    with open(os.path.join(f"./data/clean_action_dict.json"), "r") as fp:
        clean_action_dict = json.load(fp)


    for target_action, idx in action2idx.items():

        if target_action in clean_action_dict:
            target_action = clean_action_dict[target_action]
        
        if target_action == "BG":
            continue
        
        filename2subaction = {}

        with open(os.path.join(summarized_subaction_dir, target_action, "output.json"), "r") as fp:
            subaction_dict = json.load(fp)
        
        num_clusters = len(subaction_dict)
        subactions = []
        subaction_embedding = []
        
        for subaction_idx, elements in subaction_dict.items():
            subactions.append(elements["summarized_subaction"])
            subaction_embedding.append(torch.from_numpy(np.load(os.path.join(subaction_feature_dir, target_action, str(subaction_idx)+'.npy'))))
        subaction_embedding = torch.stack(subaction_embedding)

        '''
        build all possible paths
        '''
        from itertools import permutations, combinations
        path_dict = {}
        def all_combinations(lst):
            result = []
            for r in range(1, len(lst) + 1):  # lengths from 1 to len(lst)
                for combo in combinations(lst, r):
                    for perm in permutations(combo):
                        result.append(list(perm))
            return result

        # Example
        lst = [i for i in range(1, num_clusters+1)]
        perms = all_combinations(lst)

        # perms = itertools.permutations([i for i in range(1, num_clusters+1)])
        edges_info = []
        for perm in perms:
            # path_dict[perm] = 0 
            edges = []
            edges.append([0, perm[0]])
            for i in range(len(perm) - 1):
                edges.append([perm[i], perm[i+1]])
            edges.append([perm[-1], num_clusters+1])

            edges_info.append(edges)


        '''
        build AXG for graph2video alignment method
        '''
        node_dict = {}

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
        ########################################################

        valid_name = 0
        for name in filenames:
            name = name.strip("\n")
            filename2subaction[name] = []
            with open(os.path.join("./data", dataset, task, label_path, name+".txt"), "r") as fp:
                lines = fp.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].split("|")[0]

            actions, timestamps = generate_start_end_actions(lines)

            caption_embedding = []
            captions = []
            if dataset == "CaptainCook4D":
                vid_features = torch.from_numpy(np.load(os.path.join(frame_feature_input_dir, name + "_360p.npy")))
            else:
                vid_features = torch.from_numpy(np.load(os.path.join(frame_feature_input_dir, name + ".npy")))
            
            for action_idx in range(len(actions)):

                if actions[action_idx] in clean_action_dict:
                    actions[action_idx] = clean_action_dict[actions[action_idx]]

                if target_action == actions[action_idx]:
                    st, ed = timestamps[action_idx]
                    for frame_idx in range(st, ed):
                        if not os.path.exists(os.path.join(input_dir, name, "%06d.txt"%frame_idx)):
                            continue
                        
                        with open(os.path.join(input_dir, name, "%06d.txt"%frame_idx), "r") as fp:
                            ############ using object state
                            all_text = fp.read().lower()
                            caption = all_text.split(".")[0] # only the first sentence is subaction
                        
                        # if caption not in captions and caption != BG:
                        if caption != BG:
                            captions.append(caption)

                            ratio = (frame_idx - st) / (ed - st)
                            progress = int(ratio * 100)
                            progress_embedding = sinusoidal_embedding(torch.tensor([progress]).float()).squeeze(0)
                            caption_embedding.append(vid_features[frame_idx] + progress_embedding)
                            # no temporal tembedding
                            # caption_embedding.append(vid_features[frame_idx])
            
            if len(captions) == 0: # action dose not exist or other issues, than skip it
                print("Filename", name, "does not have", target_action, ", so skip it...")
                continue



            ################## using VLM or subaction prototype
            caption_embedding = torch.stack(caption_embedding)

            ################## using SBERT
            # caption_embedding = model.encode(captions)

            sims = model.similarity(caption_embedding, subaction_embedding) # N, K, where N is number of frames and K is number of captions
            N, K = sims.shape
            
            ########################################################
            # make all thread has the same start node and end node
            ########################################################
            # add first and last frames
            new_sims = torch.cat([torch.zeros((1, K)), sims, torch.zeros((1, K))], axis=0)

            # add start and end nodes
            sims = torch.cat([torch.zeros((N + 2, 1)), new_sims, torch.zeros((N + 2, 1))], axis=1)

            sims = (sims + 1) / 2

            sims[0, 0] = 1.0
            sims[-1, -1] = 1.0

            ########################################################
            
            # no drop
            zx_costs, drop_costs, node_drop_costs = compute_generalized_metadag_costs(sims, idx2node, -100, -200)
            _, pred, type_pred = generalized_metadag2vid(zx_costs.cpu().numpy(), drop_costs.cpu().numpy(), node_drop_costs.cpu().numpy(), StepGraph, idx2node)
            
            pred = pred[1:-1]
            path = []
            for p in pred:
                if p not in path:
                    path.append(p.item())
            path = tuple(path)

            if path not in path_dict:
                path_dict[path] = 1
            else:
                path_dict[path] += 1
            
            valid_name += 1
        
        ActionGraph = []
        for key, value in path_dict.items():
            if value > valid_name / len(path_dict) - 1 and value > 1: # must be larger than average, assume there is no loop and it should not be
                edges = []
                edges.append([0, key[0]])
                for i in range(len(key) - 1):
                    edges.append([key[i], key[i+1]])
                edges.append([key[-1], num_clusters+1])
                ActionGraph.append(edges)

        assert len(ActionGraph) != 0, "ActionGraph cannot be empty"
        with open(os.path.join(axg_output_dir, target_action+".json"), "w") as fp:
            json.dump(ActionGraph, fp, indent=4)

