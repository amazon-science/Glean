import os
import csv
import json
import copy
import random
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import gc
from sentence_transformers import SentenceTransformer, util
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    size = min(y_pred.size, y_true.size)
    for i in range(size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred, known_lab):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    ind_map = {j: i for i, j in ind}
    
    old_acc = 0
    total_old_instances = 0
    for i in known_lab:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances
    
    new_acc = 0
    total_new_instances = 0
    for i in range(len(np.unique(y_true))):
        if i not in known_lab:
            new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])
    if total_new_instances == 0:
        new_acc = 0
    else:
        new_acc /= total_new_instances
    return (round(acc*100, 2), round(old_acc*100, 2), round(new_acc*100, 2))

def clustering_score(y_true, y_pred, known_lab):
    Acc, Known, Novel = clustering_accuracy_score(y_true, y_pred, known_lab)
    return {
            'Acc': Acc,
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'H-Score': round(2 * Known * Novel / (Known + Novel), 2),
            'Known': Known,
            'Novel': Novel
            }


def mask_tokens(inputs, tokenizer,\
    special_tokens_mask=None, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs==0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class view_generator:
    def __init__(self, tokenizer, rtr_prob, seed):
        set_seed(seed)
        self.tokenizer = tokenizer
        self.rtr_prob = rtr_prob
    
    def random_token_replace(self, ids):
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        ids, _ = mask_tokens(ids, self.tokenizer, mlm_probability=0.25)
        random_words = torch.randint(len(self.tokenizer), ids.shape, dtype=torch.long)
        indices_replaced = torch.where(ids == mask_id)
        ids[indices_replaced] = random_words[indices_replaced]
        return ids

    def shuffle_tokens(self, ids):
        view_pos = []
        for inp in torch.unbind(ids):
            new_ids = copy.deepcopy(inp)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            np.random.shuffle(inds)
            shuffled_inds = sent_tokens_inds[inds]
            inp[sent_tokens_inds] = new_ids[shuffled_inds]
            view_pos.append(new_ids)
        view_pos = torch.stack(view_pos, dim=0)
        return view_pos

def measure_interpretability(predictions, references, args):
    ## Compute Similarity Matrix
    # Load the Sentence Transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Compute embeddings for both lists of sentences
    prediction_embeddings = model.encode(predictions, convert_to_tensor=True)
    reference_embeddings = model.encode(references, convert_to_tensor=True)

    # Compute pairwise cosine similarity
    similarity_matrix = util.pytorch_cos_sim(prediction_embeddings, reference_embeddings)

    # Convert similarity matrix to a numpy array for easier handling if necessary
    similarity_matrix = similarity_matrix.cpu().numpy()

    # Cleanup to free GPU memory
    del model, prediction_embeddings, reference_embeddings
    torch.cuda.empty_cache()
    gc.collect()

    # return the index of the maximum value in each row
    max_indices = np.argmax(similarity_matrix, axis=1).tolist()
    print('Similarity Matrix: ', similarity_matrix)
    print('Matching Index: ', max_indices)

    ## Compute Diverse Metrics
    K = len(references)
    # Coverage Score: The percentage of unique items in the list
    coverage_score = len(set(max_indices)) / K
    print('Coverage Score: ', coverage_score)

    # Uniformity Score: how evenly the list covers all the items, max score is 1
    counts = [max_indices.count(i) for i in range(K)] # num of times each cateogry are mapped to
    ratio = [count / len(max_indices) for count in counts] # ratio of each category are mapped to
    uniformity_score = -sum([r * np.log(r) for r in ratio if r > 0]) / np.log(K)  # calculate entropy, only for non-zero ratios
    print('Uniformity Score: ', uniformity_score)

    # Semantic Matching Score: how well the list matches the reference list in terms of semantic similarity
    max_scores = np.max(similarity_matrix, axis=1)
    semantic_matching_score = np.mean(max_scores)
    print('Semantic Matching Score: ', semantic_matching_score)

    # Informativeness Score: consider both the semantic matching score and the uniformity score
    informativeness_score = semantic_matching_score * uniformity_score
    print('Informativeness Score: ', informativeness_score)

    print('\n### More Detailed Interpretability Results ###')
    print(f'[#Unique Mapped Categories/#Total Categories]: [{len(set(max_indices))}/{K}]')
    print('Unique Mapped Categories: ', set(max_indices))
    print('Counts of Each Mapped Categories: ', counts)
    
    # Show some good and bad cases: good cases: top 5 highest similarity scores, bad cases: top 5 lowest similarity scores
    top_k = 10
    print('Top K Highest Similarity Scores and References and Corresponding Predictions: ')
    top_k_indices, top_k_scores = zip(*sorted(enumerate(max_scores), key=lambda x: x[1], reverse=True)[:top_k])
    for i, (index, score) in enumerate(zip(top_k_indices, top_k_scores)):
        print(f'{i+1}. Similarity Score: {score:.3f}, Reference: {references[max_indices[index]]}, Prediction: {predictions[index]}')
    print('Top K Lowest Similarity Scores and References and Corresponding Predictions: ')
    bottom_k_indices, bottom_k_scores = zip(*sorted(enumerate(max_scores), key=lambda x: x[1], reverse=False)[:top_k])
    for i, (index, score) in enumerate(zip(bottom_k_indices, bottom_k_scores)):
        print(f'{i+1}. Similarity Score: {score:.3f}, Reference: {references[max_indices[index]]}, Prediction: {predictions[index]}')

    # Save the interpretability scores
    interpretability_scores = {'Coverage Score': coverage_score, 'Uniformity Score': uniformity_score, 'Semantic Matching Score': semantic_matching_score, 'Informativeness Score': informativeness_score}

    save_results_path = './analysis/interpretability'
    file_name = f'interpretability_score_{args.experiment_name}.csv'
    results_path = os.path.join(save_results_path, file_name)

    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    names = ['interpret_sampling_strategy', 'interpret_num_representatives', 'llm', 'label_setting', 'labeled_ratio', 'labeled_shot', 'known_cls_ratio', 'evaluation_epoch', 'experiment_name', 'running_method']
    var = [args.interpret_sampling_strategy, args.interpret_num_representatives, args.llm, args.label_setting, args.labeled_ratio, args.labeled_shot, args.known_cls_ratio, args.evaluation_epoch, args.experiment_name, args.running_method]
    print('Key Hyperparameters and Values:')
    for i in range(len(names)):
        print(names[i], ':', var[i])
    vars_dict = {k:v for k,v in zip(names, var)}
    results = dict(interpretability_scores,**vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        # df1 = df1.append(new,ignore_index=True)
        df1 = pd.concat([df1, new], ignore_index=True)
        df1.to_csv(results_path,index=False)
    
    print('Interpretability Scores Saved to ', results_path)

    return coverage_score, uniformity_score, semantic_matching_score, informativeness_score