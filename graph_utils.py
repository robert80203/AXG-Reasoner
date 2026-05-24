import torch
import numpy as np
import torch.nn.functional as F
import networkx as nx
from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor


'''
The code is modified based on Graph2Vid, ECCV2022
Please refer to [https://github.com/SamsungLabs/Graph2Vid] to access the official Graph2Vid implementation
'''

class Node:
    def __init__(self, node_id, parents):
        self.node_id = node_id
        self.parents = parents
        self.neighbors_up = set()

    def __repr__(self):
        return f"N{self.node_id}: p" + "_".join(str(p.node_id) for p in self.parents)

    def push_down_neighbors(self, neighbors_up):
        self.neighbors_up = self.neighbors_up.union(neighbors_up)
        if len(self.parents) == 0:
            self.neighbors_down = set()
        else:
            neigh = neighbors_up.union({self.node_id})
            all_neighbors_down = [p.push_down_neighbors(neigh) for p in self.parents]
            self.neighbors_down = set().union(*all_neighbors_down)

        return self.neighbors_down.union({self.node_id})

    def get_thread(self):
        return self.neighbors_up.union(self.neighbors_down)

    def get_parallel_nodes(self, all_nodes):
        return all_nodes - self.get_thread().union({self.node_id})


def create_dag(parents_dict):
    # assumes parents dict is tomologically sorted
    all_nodes = set(list(parents_dict.keys()))
    sinks = all_nodes
    for parents in parents_dict.values():
        sinks = sinks - set(parents)

    node_dict = {}
    for node_id, parents in parents_dict.items():
        parent_nodes = [node_dict[p] for p in parents]
        node_dict[node_id] = Node(node_id, parent_nodes)
    return node_dict, [node_dict[s] for s in sinks]


def topological_sort(dag):
    sorted_dag = dict()
    tmp_dag = {k: v for k, v in dag.items()}
    while len(tmp_dag) > 0:
        for node, parents in list(tmp_dag.items()):
            parents = [p for p in parents if p not in sorted_dag]
            if len(parents) == 0:
                sorted_dag[node] = dag[node]
                tmp_dag.pop(node)
            else:
                tmp_dag[node] = parents
    return sorted_dag


def process_dag(dag):
    sorted_dag = topological_sort(dag)
    dag_nodes, sink_nodes = create_dag(sorted_dag)
    all_nodes = set(list(dag_nodes.keys()))
    for s in sink_nodes:
        s.push_down_neighbors(set())

    thread_dict = {nid: n.get_thread() for nid, n in dag_nodes.items()}
    parallel_dict = {nid: n.get_parallel_nodes(all_nodes) for nid, n in dag_nodes.items()}
    return sorted_dag, thread_dict, parallel_dict


def compute_generalized_metadag_costs(
    sims, # N x K
    idx2node,
    drop_base=1.0,
    node_drop_base=-200
):

    num_frames, num_steps = sims.size()

    ## dynamic drop version 2
    if drop_base == -100:
        baseline_logit = torch.tensor([0.0])
        drop_logits = baseline_logit.repeat([1, num_frames])  # making it of shape [1, N]
        drop_costs = -drop_logits.squeeze()
    else:
        max_drop_cost = torch.tensor([1/(num_steps) for i in range(num_steps)])#.to(sims.device)
        max_drop_cost = - torch.sum(max_drop_cost * torch.log(max_drop_cost))
        drop_costs = - torch.sum(sims * torch.log(sims), dim=1) / max_drop_cost
        drop_costs = - (drop_costs + drop_base)
    ## no node drop
    # node_base_logits = torch.tensor([-200])
    # node_drop_logits = node_base_logits.repeat([num_steps])
    # node_drop_costs = -node_drop_logits

    # ## dynamic node drop
    node_base_logits = torch.tensor([node_drop_base])
    node_drop_logits = node_base_logits.repeat([num_steps])
    node_drop_costs = -node_drop_logits
    # values, _ = torch.topk(sims[:, :].to(node_drop_costs.device), 1, dim=0, largest=True)
    values, _ = torch.topk(sims[:, :], 1, dim=0, largest=True)
    node_drop_costs = node_drop_costs * (1 - values)
    node_drop_costs = node_drop_costs.squeeze(0)

    active_nodes = np.array([int(float(v.split(",")[0])) for v in idx2node.values()])

    meta_zx_costs = -sims.permute(1, 0).unsqueeze(0) # M = 1 x K x N
    zx_costs = meta_zx_costs[:, active_nodes, :]
    return zx_costs, drop_costs, node_drop_costs[active_nodes]

    # zx_costs = meta_zx_costs
    # return zx_costs, drop_costs, node_drop_costs

# @jit(nopython=True)
def generalized_metadag2vid(zx_costs, drop_costs, node_drop_costs, metadag, idx2node, return_meta_labels=False):
    """Generalized DAG-match algorithm that allows 
    1. drop frames or nodes. 
    2. match between differet types of steps
    
    See Algorithm xxx in the paper.

    Parameters
    ----------
    zx_costs: np.ndarray [M, K, N]
        pairwise match costs between M types (1 type), K steps and N video clips
    drop_costs: np.ndarray [N]
        drop costs for each clip
    node_drop_costs: np.ndarray [K]
        drop costs for each step
    metadag: networkx
        For each node, specifies a list of parents in the DAG.
        Assuming that the list is topologically sorted.
    exclusive: bool
        If True any clip can be matched with only one step, not many.
    return_label: bool
        if True, returns output directly useful for segmentation computation (made for convenience)
    """
    M, K, N = zx_costs.shape

    # prepare DAG parents in the usable format
    node2idx = {node_id: idx for idx, node_id in idx2node.items()}
    metadag_idx = dict()
    for idx, node in idx2node.items():
        parents_nodes = list(metadag.pred[node])
        parents_idxs = [node2idx[n] for n in parents_nodes]
        metadag_idx[idx] = parents_idxs

    # prepare the list of possible states to transition from
    prev_states_dict = dict()
    for node, parents in metadag_idx.items():
        if len(parents) == 0:
            prev_states_dict[node + 1] = [0]
        else:
            prev_states_dict[node + 1] = [s + 1 for s in parents]

    # initialize solutin matrices
    # the M + 2 last dimensions correspond to different states.
    # M types + 2 drops
    D = np.zeros([K + 1, N + 1, M + 2])


    # default matching  list: 0
    pos_states = [0]
    state2type = {}
    type_idx = 1
    normal_idx = [0, 1, 2]
    for i in range(M + 2):
        if i in normal_idx:
            state2type[i] = 0
        else: # matching for errors
            pos_states.append(i)
            state2type[i] = type_idx
            type_idx += 1
    

    # Setting the same for all DPs to change later here.
    D[1:, 0, :] = np.inf
    D[0, 1:, :] = np.inf
    D[0, 0, 1:] = np.inf

    # Allow to drop frame
    D[0, 1:, 1] = np.cumsum(drop_costs) # frame drop costs initialization in state 1
    # Allow to drop node
    D[1:, 0, 2] = np.cumsum(node_drop_costs) # node drop costs initialization in state 2

    # initialize path tracking info for each state
    P = dict()
    for xi in range(1, N + 1):
        P[(0, xi, 1)] = (0, xi - 1, 1)
    for zi in range(1, K + 1):
        prev_states = []
        for pz in prev_states_dict[zi]:
            prev_states.append((pz, 0, 2))
        P[(zi, 0, 2)] = prev_states

    # filling in the dynamic tables
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # selecting the minimum cost transition (between pos and neg) for each preceeding state
            prev_states_min = []
            for pz in prev_states_dict[zi]:
                min_idx = np.argmin(D[pz, xi - 1])
                prev_states_min.append((pz, xi - 1, min_idx))

            prev_costs = [D[s] for s in prev_states_min]
            argmin_prev_costs = np.array(prev_costs).argmin()
            min_prev_cost = prev_costs[argmin_prev_costs]
            best_prev_state = prev_states_min[argmin_prev_costs]

            # cur_states = [(zi, xi - 1, s) for s in [0, 1]]
            cur_states = [(zi, xi - 1, s) for s in range(M + 2)]
            cur_costs = [D[s] for s in cur_states]

            # all positive(matching) states
            cur_pos_states = [cur_states[s] for s in pos_states]
            cur_pos_costs = [D[s] for s in cur_pos_states]
            argmin_cur = np.array(cur_pos_costs).argmin()
            cur_pos_state = cur_pos_states[argmin_cur]
            cur_pos_cost = cur_pos_costs[argmin_cur]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # state other than 1 and 2: x is kept
            pi = 0
            for ps in pos_states:
                match_cost = zx_costs[pi][z_cost_ind, x_cost_ind]
                if cur_pos_cost < min_prev_cost:
                    D[zi, xi, ps] = cur_pos_cost + match_cost
                    P[(zi, xi, ps)] = cur_pos_state
                else:
                    D[zi, xi, ps] = min_prev_cost + match_cost
                    P[(zi, xi, ps)] = best_prev_state
                pi += 1

            # state 1: frame is dropped
            costs_neg = np.array(cur_costs) + drop_costs[x_cost_ind]
            opt_ind_neg = np.argmin(costs_neg)
            D[zi, xi, 1] = costs_neg[opt_ind_neg]
            P[(zi, xi, 1)] = cur_states[opt_ind_neg]

            # state 2: node is dropped
            prev_states_min = []
            for pz in prev_states_dict[zi]:
                min_idx = np.argmin(D[pz, xi])
                prev_states_min.append((pz, xi, min_idx))

            prev_costs = [D[s] for s in prev_states_min]

            # costs_neg = np.array(prev_costs) + node_drop_costs[z_cost_ind]
            # opt_ind_neg = np.argmin(costs_neg)
            # D[zi, xi, 2] = costs_neg[opt_ind_neg]
            # P[(zi, xi, 2)] = prev_states_min[opt_ind_neg]

            argmin_prev_costs = np.array(prev_costs).argmin()
            min_prev_cost = prev_costs[argmin_prev_costs]
            best_prev_state = prev_states_min[argmin_prev_costs]
            D[zi, xi, 2] = min_prev_cost + node_drop_costs[z_cost_ind]
            P[(zi, xi, 2)] = best_prev_state


    cur_state = D[K, N, :].argmin()

    # backtracking the solution
    # x dropped, z dropped
    labels = np.zeros([N], dtype=int)
    type_labels = np.zeros([N], dtype=int)
    meta_labels = [-1 for _ in range(N)]
    
    parents = [(K, N, cur_state)]
    while len(parents) > 0:
        zi, xi, cur_state = parents.pop(0)
        if xi > 0:
            # print(idx2node[zi - 1], xi, cur_state)
            meta_node_id = idx2node[zi - 1] if zi > 0 else -1
            meta_labels[xi - 1] = meta_node_id
            label = int(float(meta_node_id.split(",")[0])) if zi > 0 else -1
            # labels[xi - 1] = label if cur_state == 0 else -1
            if cur_state == 1:
                labels[xi - 1] = -1
            elif cur_state == 2:
                pass # do nothing, otherwise it may overwrite previous results
            else:
                labels[xi - 1] = label
            
            if cur_state == 2:
                pass  # do nothing, otherwise it may overwrite previous results
            else:
                type_labels[xi - 1] = state2type[cur_state]
            parents.append(P[(zi, xi, cur_state)])
    min_cost = D[K, N].min()

    if return_meta_labels:
        return min_cost, labels, type_labels, meta_labels
    else:
        return min_cost, labels, type_labels

