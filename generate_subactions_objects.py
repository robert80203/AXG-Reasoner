import torch
import os
from sklearn.cluster import KMeans
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import argparse



def kmean_clustering(embeddings, num_clusters, percent_in_total, subactions, frame_paths, progress_list):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(embeddings)
    idx2subactions = {}
    total_num_subactions = 0
    for i in range(1, num_clusters+1):
        if i not in idx2subactions:
            idx2subactions[i] = {
                "subactions": [],
                "frame_paths": [],
                "progress_list": []
            }
        for j in range(len(kmeans.labels_)):
            if kmeans.labels_[j] == i - 1:
                idx2subactions[i]["subactions"].append(subactions[j])
                idx2subactions[i]["frame_paths"].append(frame_paths[j])
                idx2subactions[i]["progress_list"].append(progress_list[j])
                total_num_subactions += 1
    filtered_idx2subactions = {}
    filtered_clus_idx = 1
    for clus, elem in idx2subactions.items():
        if len(elem["subactions"]) / total_num_subactions > percent_in_total:
            filtered_idx2subactions[filtered_clus_idx] = elem
            filtered_clus_idx += 1
    # print(len(filtered_idx2subactions))
    return filtered_idx2subactions

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

def sinusoidal_embedding(pos, dim=384, base=10000.0, scale=1.0, device='cpu'):
    if not torch.is_tensor(pos):
        pos = torch.tensor(pos, dtype=torch.float32, device=device)
    pos = pos.unsqueeze(-1) * scale  # [N, 1]
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    div = torch.pow(base, (2 * i) / dim)  # [dim/2]
    angles = pos / div                    # [N, dim/2]
    pe = torch.empty(pos.shape[0], dim, device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    # L2-normalize for cosine-like clustering
    pe = torch.nn.functional.normalize(pe, dim=-1)
    return pe



if __name__ == "__main__":
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
        
    num_clusters = num_clusters_dict[task]
    num_obj_clusters = num_clusters

    keep_percentile = 80
    percent_in_total = 1 / num_clusters
    
    input_dir = f"./data/{dataset}/{task}/action_object_state"
    output_dir = f"./output/{dataset}/{task}/subactions_{num_clusters}_clust"
    obj_state_output_dir = f"./output/{dataset}/{task}/obj_state_{num_clusters}_clust"
    
    BG = "the person is waiting."
    # ============================= variables ====================================
    
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(obj_state_output_dir)

    #######################################
    # generate substeps
    #######################################
    with open(os.path.join(f"./data/{dataset}/action2idx.json"), "r") as fp:
        action2idx = json.load(fp)[task]

    with open(os.path.join(f"./data/{dataset}/{task}/{split}.txt"), "r") as fp:
        filenames = fp.readlines()
    
    with open(os.path.join(f"./data/clean_action_dict.json"), "r") as fp:
        clean_action_dict = json.load(fp)

    for target_action, idx in action2idx.items():

        if target_action in clean_action_dict:
            target_action = clean_action_dict[target_action]
        
        if target_action == "BG": # ignore background action in the list
            continue

        subactions = []
        progress_embeddings = []
        progress_list = []
        frame_paths = []
        subaction2idx = {}
        object_state = []

        for name in filenames:
            name = name.strip("\n")

            with open(os.path.join("./data", dataset, task, label_path, f"{name}.txt"), "r") as fp:
                lines = fp.readlines()
            
            for i in range(len(lines)):
                lines[i] = lines[i].split("|")[0]

            actions, timestamps = generate_start_end_actions(lines)

            for action_idx in range(len(actions)):
                if actions[action_idx] in clean_action_dict:
                    actions[action_idx] = clean_action_dict[actions[action_idx]]

                if target_action == actions[action_idx]:
                    st, ed = timestamps[action_idx]
                    for frame_idx in range(st, ed):
                        if not os.path.exists(os.path.join(input_dir, name, f"{frame_idx:06d}.txt")):
                            continue
                        
                        with open(os.path.join(input_dir, name, f"{frame_idx:06d}.txt"), "r") as fp:
                            all_text = fp.read().lower()
                            first_line = all_text.split("\n")
                            if len(first_line) > 1: # means this is a BG due to LLM
                                subaction = BG
                            else:
                                tokens = first_line[0].split(".")
                                if len(tokens) == 2: # means only BG exists, this is a BG
                                    subaction = BG
                                else:
                                    subaction = tokens[0].lower()
                                    # select first 100 instances
                                    # if len(object_state) < 100:
                                    for i in range(1, len(tokens) - 1):
                                        object_state.append(tokens[i].lower())
                        
                        # if subaction not in subactions and subaction != BG:
                        if subaction != BG:
                            subactions.append(subaction)
                            ratio = (frame_idx - st) / (ed - st)
                            progress = int(ratio * 100)
                            progress_list.append(progress)
                            progress_embedding = sinusoidal_embedding(torch.tensor([progress]).float()).squeeze(0)
                            progress_embeddings.append(progress_embedding)
                            frame_paths.append(os.path.join(name, f"{frame_idx:06d}.png"))

        
        if len(subactions) > num_clusters:
            print("Find subactions for", target_action)
            '''
            clustering for subactions
            '''
            embeddings = model.encode(subactions)
            progress_embeddings = torch.stack(progress_embeddings).numpy()
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(embeddings + progress_embeddings * 0.1)

            distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[kmeans.labels_], axis=1)
            threshold = np.percentile(distances, keep_percentile)
            outliers = distances > threshold

            idx2subactions = {}

            total_num_subactions = 0
            for i in range(1, num_clusters+1):
                if i not in idx2subactions:
                    idx2subactions[i] = {
                        "subactions": [],
                        "frame_paths": [],
                        "progress_list": []
                    }
                for j in range(len(kmeans.labels_)):
                    if kmeans.labels_[j] == i - 1 and not outliers[j]:
                        idx2subactions[i]["subactions"].append(subactions[j])
                        idx2subactions[i]["frame_paths"].append(frame_paths[j])
                        idx2subactions[i]["progress_list"].append(progress_list[j])
                        total_num_subactions += 1

            '''
            filter out subactions with less than percent_in_total of the total
            '''
            filtered_idx2subactions = {}
            filtered_clus_idx = 1
            for clus, elem in idx2subactions.items():
                if len(elem["subactions"]) / total_num_subactions > percent_in_total:
                    filtered_idx2subactions[filtered_clus_idx] = elem
                    filtered_clus_idx += 1
            
            with open(os.path.join(output_dir, target_action+".json"), "w") as fp:
                json.dump(filtered_idx2subactions, fp, indent=4)


            '''
            clustering for object states
            '''
            embeddings = model.encode(object_state)
            kmeans = KMeans(n_clusters=num_obj_clusters, random_state=0, n_init="auto").fit(embeddings)

            # filter out outliers
            distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[kmeans.labels_], axis=1)
            threshold = np.percentile(distances, keep_percentile)
            outliers = distances > threshold

            idx2subactions = {}

            total_num_subactions = 0
            for i in range(1, num_obj_clusters+1):
                if i not in idx2subactions:
                    idx2subactions[i] = {
                        "subactions": []
                    }
                for j in range(len(kmeans.labels_)):
                    if kmeans.labels_[j] == i - 1 and not outliers[j]:
                        idx2subactions[i]["subactions"].append(object_state[j])
                        total_num_subactions += 1

            '''
            filter out objects with less than percent_in_total of the total
            '''
            filtered_idx2subactions = {}
            filtered_clus_idx = 1
            for clus, elem in idx2subactions.items():
                if len(elem["subactions"]) / total_num_subactions > 1 / num_obj_clusters:
                    filtered_idx2subactions[filtered_clus_idx] = elem
                    filtered_clus_idx += 1

            with open(os.path.join(obj_state_output_dir, target_action+".json"), "w") as fp:
                json.dump(filtered_idx2subactions, fp, indent=4)

        else:
            print("Skip subactions for", target_action)