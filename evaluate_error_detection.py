import json
import os
import numpy as np
from utils.metrics import Video, Checkpoint
from utils.utils import draw_pred_v2, compute_per_out_log
import torch
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tea")
    parser.add_argument("--dataset", type=str, default="EgoPER")
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--label_path", type=str, default="labels_10fps")
    parser.add_argument(
        "--tas_backbone",
        choices=["gt", "gtg2vid", "fact", "egoped"],
        default="gt",
        help="Choose tas backbone or use GT tas.",
    )
    parser.add_argument("--numf", type=int, default=3) # number of frames to sample
    args = parser.parse_args()

    task = args.task
    dataset = args.dataset
    label_path = args.label_path
    tas_backbone = args.tas_backbone
    num_sampling_frames = args.numf #8 #3
    split = "test"

    # ============================= variables ====================================
    with open(os.path.join("clust_config.json"), "r") as fp:
        num_clusters_dict = json.load(fp)[dataset]
    
    models = {
        f"{num_clusters_dict[task]}_clust_{tas_backbone}_{num_sampling_frames}f": f"error_reasoning_{num_clusters_dict[task]}_clust_{tas_backbone}_{num_sampling_frames}f",
    }

    error_threshold = 0.5
    results = {}
    tas_result_dir = f"./output/{dataset}"

    with open(os.path.join("./data", dataset, "action2idx.json"), "r") as fp:
        action2idx = json.load(fp)[task]
    idx2action = {}
    for k, v in action2idx.items():
        idx2action[str(v)] = k


    with open(os.path.join("./data", dataset, task, split+".txt"), "r") as fp:
        filenames = fp.readlines()



    for model_name, model_path in models.items():
        results[model_name] = {}
        input_dir = f"./output/{dataset}/{task}/{model_path}"
        vis_dir = f"./output/{dataset}/{task}/{model_path}/vis_ed"

        if not os.path.exists(input_dir):
            print("Model", input_dir, "does not exist")
            exit(0)

        if not os.path.exists(vis_dir):
            os.mkdir(vis_dir)


        video_idx = 0
        updated_pairs = []
        org_pairs = []
        error_list_gt_all = []
        error_list_pred_all = []
        for name in filenames:
            error_list_gt = []
            error_list_pred = []

            vid = name.strip("\n")

            gt_ed = []
            with open(os.path.join("./data", dataset, task, label_path, vid+".txt"), "r") as fp:
                lines = fp.readlines()

                for line in lines:
                    tokens = line.split("|")
                    if "Error" in tokens[1]:
                        gt_ed.append(1)
                    else:
                        gt_ed.append(0)

            with open(os.path.join(input_dir, vid+".json"), "r") as fp:
                all_data = json.load(fp)


            
            # for predicted TAS, segment-wise F1, considering classification and localization
            tas = all_data["tas"]
            updated_ed = np.zeros(len(tas))
            for action in all_data["action"]:
                start, end = action["start_end"]
                is_error = False
                for subaction in action["subactions"]:
                    if subaction["error_score"] > error_threshold:
                        is_error = True
                if is_error:
                    updated_ed[start:end] = 1
                else:
                    updated_ed[start:end] = 0
            updated_pairs.append(Video(video_idx, updated_ed.tolist(), gt_ed))

            # for gt TAS, segment-wise F1 BUT only considering classification
            for action in all_data["action"]:
                start, end = action["start_end"]
                error_list_gt.append(1 if gt_ed[start+1] == 1 else 0)
                is_error = False
                for subaction in action["subactions"]:
                    if subaction["error_score"] > error_threshold:
                        is_error = True
                if is_error:
                    error_list_pred.append(1)
                else:
                    error_list_pred.append(0)

            # visualization
            error_idx2action = {
                "0": "Normal",
                "1": "Error"
            }
            if args.vis:
                draw_pred_v2(torch.tensor(tas).long(), torch.from_numpy(updated_ed).long(), torch.tensor(gt_ed).long(), idx2action, error_idx2action, os.path.join(vis_dir, vid))

            error_list_gt_all.append(error_list_gt)
            error_list_pred_all.append(error_list_pred)
            video_idx += 1

        ckpt = Checkpoint(bg_class=[-100])
        ckpt.add_videos(updated_pairs)
        updated_ed_out, updated_ed_per_out = ckpt.compute_metrics()
        output, out_dict = compute_per_out_log(updated_ed_per_out)

        results[model_name]["N_f50"] = out_dict['0.500'][0]
        results[model_name]["E_f50"] = out_dict['0.500'][1]
        results[model_name]["f50"] = (results[model_name]["N_f50"] + results[model_name]["E_f50"])/2

        if tas_backbone == "gt":
            precision_list = []
            precision_n_list = []
            recall_list = []
            recall_n_list = []
            f1_list = []
            f1_n_list = []
            for i in range(len(error_list_gt_all)):
                y_true = np.array(error_list_gt_all[i])
                y_pred = np.array(error_list_pred_all[i])
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))

                tp_n = np.sum((y_true == 0) & (y_pred == 0))
                fp_n = np.sum((y_true == 1) & (y_pred == 0))
                fn_n = np.sum((y_true == 0) & (y_pred == 1))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                precision_n = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0
                recall_n = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0
                f1_n = 2 * precision_n * recall_n / (precision_n + recall_n) if (precision_n + recall_n) > 0 else 0

                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

                precision_n_list.append(precision_n)
                recall_n_list.append(recall_n)
                f1_n_list.append(f1_n)

                results[model_name]["Precision"] = np.array(precision_list).mean() * 100
                results[model_name]["Recall"] = np.array(recall_list).mean() * 100
                results[model_name]["F1"] = np.array(f1_list).mean() * 100

                results[model_name]["Precision_n"] = np.array(precision_n_list).mean() * 100
                results[model_name]["Recall_n"] = np.array(recall_n_list).mean() * 100
                results[model_name]["F1_n"] = np.array(f1_n_list).mean() * 100


    if tas_backbone == "gt":
        print("======================================================================")
        print(f"|{'Method':^{20}}|{'N F1':^{15}}|{'E F1':^{15}}|{'F1':^{15}}|")
        print(f"|{'---':^{20}}|{'---':^{15}}|{'---':^{15}}|{'---':^{15}}|")
        for model_name, out in results.items():
            # p = out["Precision"]
            # r = out["Recall"]
            f1 = out["F1"]
            f1_n = out["F1_n"]
            avg_f1 = (f1 + f1_n) / 2
            print(f"|{model_name:^{20}}|{f1_n:^{15}.1f}|{f1:^{15}.1f}|{avg_f1:^{15}.1f}|")
        print("======================================================================")
    else:
        print("======================================================================")
        print(f"|{'Method':^{20}}|{'N F1@50':^{15}}|{'E F1@50':^{15}}|{'F1@50':^{15}}|")
        print(f"|{'---':^{20}}|{'---':^{15}}|{'---':^{15}}|{'---':^{15}}|")
        for model_name, out in results.items():
            nf50 = out["N_f50"]
            ef50 = out["E_f50"]
            f50 = out["f50"]
            print(f"|{model_name:^{20}}|{nf50:^{15}.1f}|{ef50:^{15}.1f}|{f50:^{15}.1f}|")
        print("======================================================================")