import torch
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer
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

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tea")
    parser.add_argument("--dataset", type=str, default="EgoPER")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--label_path", type=str, default="labels_10fps")
    args = parser.parse_args()

    dataset = args.dataset
    task = args.task
    split = args.split
    label_path = args.label_path
    
    # ============================= variables ====================================
    with open(os.path.join("clust_config.json"), "r") as fp:
        num_clusters_dict = json.load(fp)[dataset]
    
    input_dir = f"./output/{dataset}/{task}/summarized_subactions_{num_clusters_dict[task]}_clust"
    frame_feature_input_dir = f"./data/{dataset}/{task}/vc_v_features_10fps"
    # frame_feature_input_dir = f"/data/shihpo/{dataset}/{task}/vc_v_features_10fps"
    subaction_feature_output_dir = f"./output/{dataset}/{task}/v_subaction_features_{num_clusters_dict[task]}_clust"
    subaction_feature_prog_output_dir = f"./output/{dataset}/{task}/v_subaction_prog_features_{num_clusters_dict[task]}_clust"
    summzried_input_dir = f"./output/{dataset}/{task}/summarized_subactions_{num_clusters_dict[task]}_clust"

    # ============================= variables ====================================
    
    if not os.path.exists(subaction_feature_output_dir):
        os.mkdir(subaction_feature_output_dir)
    if not os.path.exists(subaction_feature_prog_output_dir):
        os.mkdir(subaction_feature_prog_output_dir)

    os.system("rm -rf %s/*" % (subaction_feature_output_dir))
    os.system("rm -rf %s/*" % (subaction_feature_prog_output_dir))


    with open(os.path.join("./data", dataset, "action2idx.json"), "r") as fp:
        action2idx = json.load(fp)[task]

    with open(os.path.join(f"./data/clean_action_dict.json"), "r") as fp:
        clean_action_dict = json.load(fp)

    gt_timestamp_dict = {}

    with open(os.path.join("./data", dataset, task, split+".txt"), "r") as fp:
        filenames = fp.readlines()

    for name in filenames:
        vid = name.strip("\n")

        gt_labels = []
        with open(os.path.join("./data", dataset, task, label_path, vid+".txt"), "r") as fp:
            labels = fp.readlines()
            for label in labels:
                gt_labels.append(label.split("|")[0])
            steps, timestamps = generate_start_end_actions(gt_labels)
        
        gt_timestamp_dict[vid] = timestamps


    for target_action, idx in action2idx.items():

        if target_action in clean_action_dict:
            target_action = clean_action_dict[target_action]
        
        if target_action == "BG":
            continue

        if not os.path.exists(os.path.join(subaction_feature_output_dir, target_action)):
            os.mkdir(os.path.join(subaction_feature_output_dir, target_action))
        if not os.path.exists(os.path.join(subaction_feature_prog_output_dir, target_action)):
            os.mkdir(os.path.join(subaction_feature_prog_output_dir, target_action))

        subact_idx = 1
        
        with open(os.path.join(input_dir, target_action, "output.json"), "r") as fp:
            idx2subactions = json.load(fp)
        
        target_subactions = []
        target_progress_list = []
        for _, elements in idx2subactions.items():
            subaction_feature = []
            subaction_feature_prog = []

            target_subaction = [elements["summarized_subaction"]]
            target_progress = int(np.array(elements["others"]["progress_list"]).mean())
            subactions = elements["others"]["subactions"]
            frame_paths = elements["others"]["frame_paths"]
            progress_list = elements["others"]["progress_list"]

            target_progress_embedding = sinusoidal_embedding(torch.tensor([target_progress]).float(), dim=384).numpy()
            target_embedding = model.encode(target_subaction) + target_progress_embedding

            progress_embedding = sinusoidal_embedding(torch.tensor(progress_list).float(), dim=384).numpy()
            embeddings = model.encode(subactions) + progress_embedding

            sims = model.similarity(target_embedding, embeddings)

            joint_list = []
            for i in range(len(subactions)):
                joint_list.append([sims[0, i], frame_paths[i], progress_list[i]])
            
            sorted_joint_list = sorted(joint_list, key=lambda x: x[0], reverse=True)

            # for i in range(len(sorted_joint_list)//2 + 1): # select the first half
            for i in range(len(sorted_joint_list)):
                _, frame_path, progress = sorted_joint_list[i]
                vid, frame_idx = frame_path.split("/")
                if dataset == "CaptainCook4D":
                    vid_features = torch.from_numpy(np.load(os.path.join(frame_feature_input_dir, vid+"_360p.npy")))
                else:
                    vid_features = torch.from_numpy(np.load(os.path.join(frame_feature_input_dir, vid+".npy")))
                add_index = int(frame_idx[:-4])
                progress_embedding = sinusoidal_embedding(torch.tensor([progress]).float(), dim=256).squeeze(0)
                subaction_feature_prog.append(vid_features[add_index] + progress_embedding)
                subaction_feature.append(vid_features[add_index])

            subaction_feature_prog = np.stack(subaction_feature_prog).mean(0)
            subaction_feature = np.stack(subaction_feature).mean(0)
            
            np.save(os.path.join(subaction_feature_prog_output_dir, target_action, "%d.npy" % subact_idx), subaction_feature_prog)
            np.save(os.path.join(subaction_feature_output_dir, target_action, "%d.npy" % subact_idx), subaction_feature)

            subact_idx += 1
        
    print("Done")