import copy
import json
import torch
import numpy as np
# import Levenshtein
# from actseg_src.eval import IoU
from collections import defaultdict
import math

def computeIoU_acc(o_preds, o_gts):
    
    correct = 0
    total = 0
    union = 0
    intersection = 0
    for o_pred, o_gt in zip(o_preds, o_gts):
        union += len(o_pred | o_gt)
        intersection += len(o_pred & o_gt)
        
        total += len(o_gt)
        correct += len(o_pred & o_gt)

    return intersection/union, correct/total

def omission_detection(task_graph, predstep_steps):
    o_preds = []
    o_gts = []
    nodes = set(task_graph.nodes())
    for preds, gts in predstep_steps:
        gt_omitted = nodes - set(gts.numpy().tolist())
        pred_omitted = nodes - set(preds.numpy().tolist())
        if len(gt_omitted) != 0: # if there exists omitted steps
            o_preds.append(pred_omitted)
            o_gts.append(gt_omitted)
            # print(pred_omitted, gt_omitted)
    
    print("Num of valid omitted sequences:", len(o_gts))
    

    return computeIoU_acc(o_preds, o_gts)

def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        if max(m_row, n_col) == 0:
            score = 0
        else:
            score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends

# def mstcn_edit_score(pred, gt, norm=True, bg_class=["background"]):
#     P, _, _ = get_labels_start_end_time(pred, bg_class)
#     Y, _, _ = get_labels_start_end_time(gt, bg_class)
#     return levenstein(P, Y, norm)

def mstcn_f_score(pred_segs, gt_segs, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(pred_segs, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(gt_segs, bg_class)
    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    per_action_stats = defaultdict(lambda: np.array([0, 0, 0]))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
            per_action_stats[p_label[j]][0] += 1
        else:
            fp += 1
            per_action_stats[p_label[j]][1] += 1

    fn = len(y_label) - sum(hits)

    for j, h in enumerate(hits):
        if h == 0:
            per_action_stats[y_label[j]][2] += 1

    return float(tp), float(fp), float(fn), per_action_stats

class Video():

    def __init__(self, vname='', pred=[], gt=[]):
        self.vname = vname
        self.pred_label = pred
        self.gt_label = gt

    def __str__(self):
        return "< Video %s >" % self.vname

    def __repr__(self):
        return "< Video %s >" % self.vname

class Checkpoint():

    def __init__(self, iteration=-1, bg_class=[0], error_class=None):

        # self.rslt_file = None
        self.iteration = iteration
        self.metrics = None
        self.videos = {}
        self.bg_class = bg_class
        self.error_class = error_class

    def add_videos(self, videos):
        for v in videos:
            self.videos[v.vname] = v

    def __str__(self):
        return "< Checkpoint[%d] %d videos >" % (self.iteration, len(self.videos))

    def __repr__(self):
        return str(self)

    def single_video_loc_metrics(self, v):

        if not hasattr(v, 'metrics'):
            v.metrics = {}

        # pred_label = v.pred_label = expand_pred_to_gt_len(v.pred, len(v.gt_label))
        assert len(v.pred_label) == len(v.gt_label)
        pred_label = v.pred_label 

        ## Disable IoU
        # m = IoU(self.bg_class)
        # m.add(v.gt_label, v.pred_label)
        # v.metrics['IoU'] = m.summary()

        # v.metrics['edit'] = metrics2.mstcn_edit_score(pred_segs, gt_segs, bg_class=self.bg_class) / 100
        # v.metrics['edit'] = mstcn_edit_score(v.pred_label, v.gt_label, bg_class=self.bg_class) / 100
        # v.metrics['edit'] = mstcn_edit_score(v.pred_label, v.gt_label, bg_class=[-1, 0]) / 100 # do not consider background and error

        # threshold = [0.0, 0.01, 0.1, 0.25, 0.5]
        threshold = [0.1, 0.25, 0.5, 0.9, 1.0]
        for t in threshold:
            tp1, fp1, fn1, pas = mstcn_f_score(
                        v.pred_label, v.gt_label, t, bg_class=self.bg_class)
            precision = tp1 / float(tp1+fp1)
            recall = tp1 / float(tp1+fn1)
            if precision+recall == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * (precision*recall) / (precision+recall)
            f1 = np.nan_to_num(f1)
            
            v.metrics['P@%.3f'%(t)] = precision
            v.metrics['R@%.3f'%(t)] = recall
            v.metrics['F1@%.3f'%(t)] = f1

            ########################3 per action stats
            if not hasattr(v, 'per_metrics'):
                v.per_metrics = {}
            
            v.per_metrics['%.3f'%(t)] = {}

            for action, value in pas.items():
                v.per_metrics['%.3f'%(t)][action] = []
                v.per_metrics['%.3f'%(t)][action].append(value[0]) # tp
                v.per_metrics['%.3f'%(t)][action].append(value[1]) # fp
                v.per_metrics['%.3f'%(t)][action].append(value[2]) # fn


    def joint_video_acc(self, video_list):
        gt_list = [v.gt_label for v in video_list]
        pred_list = [v.pred_label for v in video_list]
        gt_ = np.concatenate(gt_list)
        pred_ = np.concatenate(pred_list)

        correct = (gt_ == pred_)
        fg_loc = np.array([ True if g not in self.bg_class else False for g in gt_ ])
        acc = correct[fg_loc].mean()
        return acc
    
    def joint_video_acc_error(self, video_list):
        gt_list = [v.gt_label for v in video_list]
        pred_list = [v.pred_label for v in video_list]
        gt_ = np.concatenate(gt_list)
        pred_ = np.concatenate(pred_list)

        gt_error_loc = gt_ == self.error_class
        gt_ = gt_[gt_error_loc]
        pred_ = pred_[gt_error_loc]

        correct = (gt_ == pred_)
        return correct.sum() / len(gt_)

    def compute_metrics(self):
        for vname, video in self.videos.items():
            video.metrics = {}
            ########## per action stats
            video.per_metrics = {}

            self.single_video_loc_metrics(video)

        metric_keys = video.metrics.keys()
        metrics = { k: np.mean([ v.metrics[k] for v in self.videos.values() ])  
                            for k in metric_keys }

        acc = self.joint_video_acc(list(self.videos.values()))
        
        metrics['acc'] = acc


        ########## per action stats
        joint_per_metrics = {}
        for v in self.videos.values():
            for t, _ in v.per_metrics.items():
                if t not in joint_per_metrics:
                    joint_per_metrics[t] = {}
                for action, tp_fp_fn in v.per_metrics[t].items():
                    if action not in joint_per_metrics[t]:
                        joint_per_metrics[t][action] = [[], [], []]
                    joint_per_metrics[t][action][0].append(tp_fp_fn[0])
                    joint_per_metrics[t][action][1].append(tp_fp_fn[1])
                    joint_per_metrics[t][action][2].append(tp_fp_fn[2])

        self.metrics = metrics

        return self.metrics, joint_per_metrics
