import os
import json
import random
import torch
import math
import numpy as np
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torchvision.io import read_image
from scipy.ndimage import generic_filter
import torch.nn.functional as F

def compute_per_out_log(joint_per_metrics, use_ignore=False, mode="ed"):
    per_out_log = []
    num_thresholds = 0
    avg_f1_thresholds = 0
    if mode == "as":
        col1 = 10
    else:
        col1 = 20
    out_dict = {

    }
    for t, _ in joint_per_metrics.items():
        
        f1_list = []
        # per_out_log.append("|\tAction\t|\tPrecision@%s\t|\tRecall@%s\t|\tF1@%s\t|\n"%(t, t, t))
        # per_out_log.append(f"|{'Action':^{col1}}|{'Precision@%s'%t:^{15}}|{'Recall@%s'%t:^{15}}|{'F1@%s'%t:^{10}}|\n")
        # print(f"|{'Action':^{col1}}|{'Precision@%s'%t:^{15}}|{'Recall@%s'%t:^{15}}|{'F1@%s'%t:^{10}}|\n")

        total_tp, total_fp, total_fn = 0, 0, 0
        theta = 0
        for action, tp_fp_fn in joint_per_metrics[t].items():
            tp, fp, fn = 0, 0, 0
            for i in range(len(tp_fp_fn[0])):
                tp += tp_fp_fn[0][i]
                fp += tp_fp_fn[1][i]
                fn += tp_fp_fn[2][i]
                total_tp += tp_fp_fn[0][i]
                total_fp += tp_fp_fn[1][i]
                total_fn += tp_fp_fn[2][i]
            
            p = tp / float(tp+fp)
            r = tp / float(tp+fn)
            if np.isnan(p):
                p = 0.0
            if np.isnan(r):
                r = 0.0
            if p+r == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * (p*r) / (p+r)
            f1 = np.nan_to_num(f1)
            p = p * 100
            r = r * 100
            f1 = f1 * 100
            if mode == "ed":
                action = "Error" if action == 1 else "Normal"
                # out_log = "|\t%s\t|\t%.1f\t|\t%.1f\t|\t%.1f\t|\n"%(action, p, r, f1)
            out_log = f"|{action:^{col1}}|{p:^{15}.1f}|{r:^{15}.1f}|{f1:^{10}.1f}|\n"
            # print(action, "precision:", np.mean(np.array(p_r[0])), "recall:", np.mean(np.array(p_r[1])))
            per_out_log.append(out_log)
            if f1 != 0:
                theta += 1
            f1_list.append(f1)
        
        # per_out_log.append("Avg F1@%s: %.1f\n\n"%(t, np.array(f1_list).mean()))
        # print("Avg F1@%s: %.1f\n\n"%(t, np.array(f1_list).mean()))
        # per_out_log.append("\n")
        # per_out_log.append("|")
        # for f1 in f1_list:
        #     per_out_log.append("%.1f|"%(f1))
        # per_out_log.append("\n\n")
        out_dict[t] = f1_list
        avg_f1_thresholds += np.array(f1_list).mean()

    # print(per_out_log)
    return avg_f1_thresholds / len(joint_per_metrics), out_dict


def generate_partitions(inputs):
    cur_class = None
    start = 0
    step_partitions = []
    for i in range(len(inputs)):
        if inputs[i] != cur_class and cur_class is not None:
            step_partitions.append((cur_class, i - start + 1))
            start = i + 1
        cur_class = inputs[i]
    step_partitions.append((inputs[len(inputs) - 1], len(inputs) - start + 1))
    return step_partitions

def draw_pred(tas, updated_pred, org_pred, gt, mapping, error_mapping, save_path):
    clean_version = False #True
    

    mycmap = plt.matplotlib.cm.get_cmap('rainbow', len(mapping))
    category_colors = [matplotlib.colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]

    mycmap = plt.matplotlib.cm.get_cmap('rainbow', len(error_mapping))
    error_category_colors = [matplotlib.colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
    
    tas_partitions = generate_partitions(tas)
    updated_pred_partitions = generate_partitions(updated_pred)
    org_pred_partitions = generate_partitions(org_pred)
    gt_partitions = generate_partitions(gt)
    
    # Create stacked subplots (3 rows, 1 column)
    fig, axes = plt.subplots(4, 1, figsize=(20, 6), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # for ax in axes:  # draw the same thing on both axes
    #     data_cum = 0
    #     for i, (l, w) in enumerate(gt_partitions):
    #         ax.barh(name, w, left=data_cum, height=0.3, 
    #                 label=mapping[str(l.item())], 
    #                 color=category_colors[l.item()])
    #         data_cum += w
        
    #     ax.set_yticks([])

    data_cum = 0
    for i, (l, w) in enumerate(tas_partitions):
        axes[0].barh("tas", w, left=data_cum, height=0.3, 
                label=mapping[str(l.item())], 
                color=category_colors[l.item()])
        data_cum += w

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), 
               ncol=4, bbox_to_anchor=(1, 1), 
               loc='upper right', fontsize='small')

    data_cum = 0
    for i, (l, w) in enumerate(gt_partitions):
        axes[1].barh("GT", w, left=data_cum, height=0.3, 
                label=error_mapping[str(l.item())], 
                color=error_category_colors[l.item()])
        data_cum += w
    
    # axes[0].set_yticks([])

    data_cum = 0
    for i, (l, w) in enumerate(org_pred_partitions):
        axes[2].barh("Org", w, left=data_cum, height=0.3, 
                label=error_mapping[str(l.item())], 
                color=error_category_colors[l.item()])
        data_cum += w
    
    # axes[1].set_yticks([])

    data_cum = 0
    for i, (l, w) in enumerate(updated_pred_partitions):
        axes[3].barh("Updated", w, left=data_cum, height=0.3, 
                label=error_mapping[str(l.item())], 
                color=error_category_colors[l.item()])
        data_cum += w
    
    # axes[2].set_yticks([])

    # Add a single legend for both copies
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), 
               ncol=4, bbox_to_anchor=(1, -0.1), 
               loc='upper right', fontsize='small')

    plt.savefig(save_path + '.png', bbox_inches="tight")
    plt.clf()
    plt.close()


def draw_pred_v2(tas, updated_pred, gt, mapping, error_mapping, save_path):
    clean_version = False #True
    

    mycmap = plt.matplotlib.cm.get_cmap('rainbow', len(mapping))
    category_colors = [matplotlib.colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]

    mycmap = plt.matplotlib.cm.get_cmap('rainbow', len(error_mapping))
    error_category_colors = [matplotlib.colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
    
    tas_partitions = generate_partitions(tas)
    updated_pred_partitions = generate_partitions(updated_pred)
    gt_partitions = generate_partitions(gt)
    
    # Create stacked subplots (3 rows, 1 column)
    fig, axes = plt.subplots(3, 1, figsize=(20, 6), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # for ax in axes:  # draw the same thing on both axes
    #     data_cum = 0
    #     for i, (l, w) in enumerate(gt_partitions):
    #         ax.barh(name, w, left=data_cum, height=0.3, 
    #                 label=mapping[str(l.item())], 
    #                 color=category_colors[l.item()])
    #         data_cum += w
        
    #     ax.set_yticks([])

    data_cum = 0
    for i, (l, w) in enumerate(tas_partitions):
        axes[0].barh("tas", w, left=data_cum, height=0.3, 
                label=mapping[str(l.item())], 
                color=category_colors[l.item()])
        data_cum += w

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), 
               ncol=4, bbox_to_anchor=(1, 1), 
               loc='upper right', fontsize='small')

    data_cum = 0
    for i, (l, w) in enumerate(gt_partitions):
        axes[1].barh("GT", w, left=data_cum, height=0.3, 
                label=error_mapping[str(l.item())], 
                color=error_category_colors[l.item()])
        data_cum += w
    
    # axes[0].set_yticks([])

    
    # axes[1].set_yticks([])

    data_cum = 0
    for i, (l, w) in enumerate(updated_pred_partitions):
        axes[2].barh("Updated", w, left=data_cum, height=0.3, 
                label=error_mapping[str(l.item())], 
                color=error_category_colors[l.item()])
        data_cum += w
    
    # axes[2].set_yticks([])

    # Add a single legend for both copies
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), 
               ncol=4, bbox_to_anchor=(1, -0.1), 
               loc='upper right', fontsize='small')

    plt.savefig(save_path + '.png', bbox_inches="tight")
    plt.clf()
    plt.close()

# def draw_pred(outputs, name, mapping, save_path, category_colors=None):
#     clean_version = False #True
    
#     if category_colors is None:
#         mycmap = plt.matplotlib.cm.get_cmap('rainbow', len(mapping))
#         category_colors = [matplotlib.colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
    
#     gt_partitions = generate_partitions(outputs)
    
#     plt.figure(figsize=(16, 4))
#     plt.subplots_adjust(top=0.5)
#     data_cum = 0
#     for i, (l, w) in enumerate(gt_partitions):
#         rects = plt.barh(name, w, left=data_cum, height=0.3, label=mapping[str(l.item())], color=category_colors[l.item()])
#         plt.yticks([])
#         data_cum += w
    

#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(), ncol=4, bbox_to_anchor=(1, 2.2), loc='upper right', fontsize='small')

#     plt.savefig(save_path+'.png')
#     plt.clf()
#     plt.close()

def create_image_grid(img_dirs):
    img_list = []
    for img_dir in os.listdir(img_dirs):
        filenames = os.listdir(os.path.join(img_dirs, img_dir))
        for filename in filenames:
            img_list.append(read_image(os.path.join(img_dirs, img_dir, filename)))
    grid = make_grid(img_list, nrow=3)
    return grid