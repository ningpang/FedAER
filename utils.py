import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class GraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))

        return adj

    def eval(self, idx, full=False):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))-torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))

        if not full:
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1, t1 = adj.topk(self.k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj*mask

        return adj

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sd_matrix(state_dict, names):

    param_vector = None
    for name in names:
        param = state_dict[name]
        if param_vector is None:
            param_vector = param.clone().detach().flatten().cpu()
        else:
            if len(list(param.size())) == 0:
                param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)), 0)
            else:
                param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)
    return param_vector



def graph_aggregate(local_models, global_model, target_names, pre_A, args):
    model_args, data_args, training_args = args
    subgraph_size = min(int(model_args.subgraph_size*model_args.num_clients), model_args.num_clients)
    param_matrix = []
    for local_model in local_models:
        param_matrix.append(sd_matrix(local_model, target_names).clone().detach())
    param_metrix = torch.stack(param_matrix)

    key_shapes = []
    for name in target_names:
        key_shapes.append(list(local_models[0][name].data.shape))

    A = generate_adj(param_matrix, args, subgraph_size).cpu().detach().numpy()



    if model_args.federated_mode == "graph":
        A = (1 - model_args.adjbeta) * pre_A + model_args.adjbeta * A

    A = normalize_adj(pre_A)
    # A = normalize_adj(A)
    A = torch.tensor(A).type(torch.float32)

    # Aggregating
    aggregated_param = torch.mm(A, param_metrix)
    # for i in range(args.layers - 1):
    #     aggregated_param = torch.mm(A, aggregated_param)
    new_param_matrix = (model_args.serveralpha * aggregated_param) + ((1 - model_args.serveralpha) * param_metrix)

    for i in range(len(local_models)):
        pointer = 0
        for k in range(len(target_names)):
            num_p = 1
            for n in key_shapes[k]:
                num_p *= n
            local_models[i][target_names[k]] = new_param_matrix[i][pointer:pointer + num_p].reshape(key_shapes[k])
            pointer += num_p

    return local_models


def generate_adj(param_metrix, args, subgraph_size):
    model_args, data_args, training_args = args
    dist_metrix = torch.zeros((len(param_metrix), len(param_metrix)))
    for i in range(len(param_metrix)):
        for j in range(len(param_metrix)):
            dist_metrix[i][j] = torch.nn.functional.pairwise_distance(
                param_metrix[i].view(1, -1), param_metrix[j].view(1, -1), p=2).clone().detach()
    dist_metrix = torch.nn.functional.normalize(dist_metrix).to(model_args.device)

    gc = GraphConstructor(model_args.num_clients, subgraph_size, 100,
                          model_args.device, model_args.adjalpha).to(model_args.device)
    idx = torch.arange(model_args.num_clients).to(model_args.device)
    optimizer = torch.optim.SGD(gc.parameters(), lr=0.01, weight_decay=0.0001)

    for e in range(model_args.gc_epochs):
        optimizer.zero_grad()
        adj = gc(idx)
        adj = torch.nn.functional.normalize(adj)

        loss = torch.nn.functional.mse_loss(adj, dist_metrix)
        loss.backward()
        optimizer.step()

    adj = gc.eval(idx).to("cpu")

    return adj