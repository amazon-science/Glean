"""
Code modified from
https://github.com/wvangansbeke/Unsupervised-Classification
"""
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from heapq import nlargest
import random

class MemoryBank(object):
    def __init__(self, args, n, dim, num_classes, temperature):
        self.args = args
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, y_pred, cluster_centers_):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included

        # compute probability that samples belong to different clusters via Student's t-distribution
        alpha =  1
        q = 1.0 / (1.0 + torch.sum((self.features.unsqueeze(1) - cluster_centers_) ** 2, dim=2) / alpha)
        q = q ** (alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, dim=1)).t()
        weight = q ** 2 / torch.sum(q, dim=0)
        p = (weight.t() / torch.sum(weight, dim=1)).t()

        # compute entropy
        entr = self.entropy(p)
        # entropy sort
        _, inx = torch.sort(entr, descending=True)

        # targets = self.targets.cpu().numpy()
        # ind, _ = self.hungray_aligment(targets, y_pred)

        # compute local inconsistency
        neighbor_pseudos = np.take(y_pred, indices[:,1:], axis=0)
        anchor_pseudos = np.repeat(y_pred.reshape(-1,1), topk, axis=1)
        temp = []
        for i in range(neighbor_pseudos.shape[0]):
            count = 0
            for j in range(neighbor_pseudos.shape[1]):
                    if neighbor_pseudos[i][j] != anchor_pseudos[i][j]:
                        count += 1
            temp.append(count)
        temp = torch.tensor(temp)
        # local inconsistency degree sort
        _, index = torch.sort(temp, descending=True)


        cluster_assignment = y_pred

        if self.args.sampling_strategy == 'loop':
            # default: all b samples with highest entropy & local inconsistency degree
            query_index = []
            for i in index[:self.args.query_samples]:
                if i in inx[:self.args.query_samples]:
                    query_index.append(i)

        elif self.args.sampling_strategy == 'random':
            # randomly sample budget query samples from the dataset
            query_index = random.sample(range(n), self.args.query_samples)

        elif self.args.sampling_strategy == 'highest':
            # sample query samples with highest uncertainty
            query_index = inx[:self.args.query_samples].tolist()

        elif self.args.sampling_strategy == 'lowest':
            # sample query samples with lowest uncertainty
            query_index = inx[-self.args.query_samples:].tolist()

        elif 'curriculum_querying' in self.args.sampling_strategy:
            """
            Rank the samples based on uncertainty, then divide the samples into args.num_training_rounds bins.
            At each round, sample self.args.query_samples data based on different strategies: random, highest uncertainty, or lowest uncertainty.
            """
            # Rank the samples based on entropy in ascending order (lower entropy first)
            _, inx = torch.sort(entr, descending=False)
            
            # Determine the number of samples per bin
            samples_per_bin = len(entr) // self.args.num_training_rounds

            # Determine the start and end index for the current round
            start_index = (self.args.current_training_round - 1) * samples_per_bin
            end_index = self.args.current_training_round * samples_per_bin

            # Ensure the end index doesn't exceed the total number of samples
            end_index = min(end_index, len(entr))

            # Extract indices for the current bin
            current_bin_indices = inx[start_index:end_index].tolist()

            # Sample from the current bin based on the strategy
            if self.args.sampling_strategy == 'curriculum_querying_highest':
                query_index = current_bin_indices[-self.args.query_samples:]
            
            elif self.args.sampling_strategy == 'curriculum_querying_lowest':
                query_index = current_bin_indices[:self.args.query_samples]
            else:
                # self.args.sampling_strategy == 'curriculum_querying_random':
                query_index = random.sample(current_bin_indices, self.args.query_samples)

        elif self.args.sampling_strategy == 'equal_random':
            # sample equal number of query samples randomly from each cluster
            cluster_budget = [self.args.query_samples // self.C] * self.C
            # ensure the sum of allocations equals the budget: the remaining budget is allocated to the cluster randomly
            while sum(cluster_budget) < self.args.query_samples:
                cluster_budget[random.choice(range(self.C))] += 1
            query_index = []
            for i in range(self.C):
                query_index.extend(random.sample(inx[cluster_assignment == i].tolist(), cluster_budget[i]))

        else:
            if self.args.sampling_strategy == 'equal_highest':
                # equal allocation of query samples to each cluster
                cluster_budget = [self.args.query_samples // self.C] * self.C
                # ensure the sum of allocations equals the budget: the remaining budget is allocated to the cluster randomly
                while sum(cluster_budget) < self.args.query_samples:
                    cluster_budget[random.choice(range(self.C))] += 1

            elif self.args.sampling_strategy == 'one_highest':
                # take all the sample from a random cluster
                cluster_budget = [0] * self.C
                
            # for each cluster, sample query samples with highest uncertainty based on the cluster budget
            query_index = []
            for i in range(self.C):
                if cluster_budget[i] == 0:
                    continue
                query_index.extend(inx[cluster_assignment == i][:cluster_budget[i]].tolist())
        

        # print('\nQuery Index:', query_index)

        return indices, query_index, p
    

    def allocate_query_samples(self, difficulty_scores, budget, degree):
        degree = float(degree)
        difficulty_scores = torch.tensor(difficulty_scores, dtype=torch.float32)
        K = len(difficulty_scores)
        
        weights = torch.pow(difficulty_scores, degree)
        
        normalized_weights = weights / weights.sum()
        allocations = (normalized_weights * budget).floor().int()
        
        # Ensure the sum of allocations equals the budget
        while allocations.sum() < budget:
            diff = budget - allocations.sum()
            _, indices = torch.topk(weights, diff)
            allocations[indices] += 1
        
        return allocations.tolist()

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')

    def hungray_aligment(self, y_true, y_pred):
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D))
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
        return ind, w

    def entropy(self, x, eps=1e-5):
        p = F.softmax(x, dim=-1)
        entropy = -torch.sum(p * torch.log(p), 1)
        return entropy



@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):

        batch = tuple(t.cuda(non_blocking=True) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
        feature = model(X, output_hidden_states=True)["hidden_states"]

        memory_bank.update(feature, label_ids)
        if i % 20 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))