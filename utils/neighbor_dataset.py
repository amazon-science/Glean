import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from heapq import nlargest
import openai
import time
import re
import json

class NeighborsDataset(Dataset):
    def __init__(self, args, dataset, indices, query_index, pred, p, cluster_name=None, num_neighbors=None,
                di_all=None, di_all_pos_cluster_idx=None, di_all_neg_cluster_idx=None):
        super(NeighborsDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k]) => [len(dataset) x (k+1)]
        self.query_index = query_index
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.pred = pred
        self.count = 0
        self.di = {}
        self.di_pos_cluster_idx = {}
        self.di_neg_cluster_idx = {}
        
        self.di_all = di_all
        self.di_all_pos_cluster_idx = di_all_pos_cluster_idx
        self.di_all_neg_cluster_idx = di_all_neg_cluster_idx
        
        if args.running_method == 'no_llm_neighbor_refinement':
            openai.api_key = None
            self.api_key = None
        else:
            openai.api_key = args.api_key
            self.api_key = args.api_key     
        assert(self.indices.shape[0] == len(self.dataset))

        self.p = p
        self.cluster_name = cluster_name 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = list(self.dataset.__getitem__(index))
        neighbor_pred = np.take(self.pred, self.indices[index, :])

        # unique prediction
        res = [neighbor_pred[0]]
        for i in neighbor_pred[1:]:
            if i not in res:
                res.append(i)
                break


        pos_cluster_idx = None
        neg_cluster_idx = None
        if self.args.running_method not in ['Loop', 'GCD', 'SimGCD', 'BaCon']:
            ## Ours
            if index not in self.query_index:
                if self.di_all.get(index, -1) == -1:
                    # For the unselected samples, randomly select a sample from their neighbors
                    neighbor_index = np.random.choice(self.indices[index], 1)[0]
                else:
                    # If they have been queried in previous rounds, use the LLM selected neighbors
                    neighbor_index = self.di_all[index]
                    if self.args.weight_cluster_instance_cl > 0:
                        pos_cluster_idx = self.di_all_pos_cluster_idx[index]
                        neg_cluster_idx = self.di_all_neg_cluster_idx[index]

            else:
                # For the selected samples, query llm to select the most similar sample from the neighboring clusters
                anchor_text = self.tokenizer.decode(anchor[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                if self.di.get(index, -1) == -1:
                    prob_tensor = self.p[index, :]
                    if self.args.component_ablation == 'wo_instance_feedback' or self.args.component_ablation == 'wo_both_feedback':
                        neighbor_index = np.random.choice(self.indices[index], 1)[0]
                        self.di[index] = neighbor_index
                        self.di_all[index] = neighbor_index
                    else:
                        topk_probs, topk_indices = torch.topk(prob_tensor, self.args.options, dim=-1)
                        qs = [np.random.choice(np.where(self.pred==topk_indices[i].item())[0], 1)[0] for i in range(self.args.options)]
                        neighbor_index = self.query_llm_gen(index, qs)
                        self.di[index] = neighbor_index
                        self.di_all[index] = neighbor_index

                    if self.args.weight_cluster_instance_cl > 0:
                        # query llm to assign the anchor to one of the topk clusters based on category names and descriptions
                        k = int(np.floor(self.args.options_cluster_instance_ratio * len(self.cluster_name)))
                        topk_probs, topk_indices = torch.topk(prob_tensor, k, dim=-1)
                        topk_cluster_name = [self.cluster_name[i.item()] for i in topk_indices]
                        pos_cluster_idx = self.query_llm_cluster_instance(anchor_text, topk_cluster_name, topk_indices)
                        neg_cluster_idx = topk_indices[topk_indices != pos_cluster_idx]

                        if self.count < 6:
                            print(f"\nAnchor: {anchor_text} \nPositive Cluster Name: {self.cluster_name[pos_cluster_idx]}")
                        
                        self.di_all_pos_cluster_idx[index] = pos_cluster_idx
                        self.di_all_neg_cluster_idx[index] = neg_cluster_idx

                    self.di_pos_cluster_idx[index] = pos_cluster_idx
                    self.di_neg_cluster_idx[index] = neg_cluster_idx
                    self.count += 1

                else:
                    neighbor_index = self.di[index]
                    if self.args.weight_cluster_instance_cl > 0:
                        pos_cluster_idx = self.di_pos_cluster_idx[index]
                        neg_cluster_idx = self.di_neg_cluster_idx[index]

        else:
            ## Generalized Loop
            # For the unselected samples, randomly select a sample from their neighbors
            if len(res) == 1 or index not in self.query_index or self.args.running_method in ['GCD', 'SimGCD']:
                neighbor_index = np.random.choice(self.indices[index], 1)[0]
            else:
                # For the selected samples, randomly select a sample from its top neighboring clusters
                # Generalize to the case # querying neighbors / options >= 2
                qs = [np.random.choice(self.indices[index, np.where(neighbor_pred==res[i])][0], 1)[0] for i in range(self.args.options)]

                if self.di.get(index, -1) == -1:
                    # Generalize to the case # querying neighbors / options >= 2
                    neighbor_index = self.query_llm_gen(index, qs)
                    
                    self.di[index] = neighbor_index
                    self.count += 1
                else:
                    neighbor_index = self.di[index]
        
    
        neighbor = self.dataset.__getitem__(neighbor_index)
        output['anchor'] = anchor[:3]
        output['neighbor'] = neighbor[:3]
        output['possible_neighbors'] = torch.from_numpy(self.indices[index]) # used for neighbor contrastive learning
        output['target'] = anchor[-1]
        output['index'] = index
        output['pos_cluster_idx'] = pos_cluster_idx
        output['neg_cluster_idx'] = neg_cluster_idx

        return output
    
    
    def query_llm_gen(self, q, qs):
        s = self.tokenizer.decode(self.dataset.__getitem__(q)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        sqs = [self.tokenizer.decode(self.dataset.__getitem__(q)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True) for q in qs]
        
        # Construct the base of the prompt
        prompt = (
            f"Select the customer utterance that better corresponds with the Query in terms of {self.args.task}. "
            "Please respond in the format 'Choice [number]' without explanation, e.g., 'Choice 1', 'Choice 2', etc."
            "\nQuery: " + s
        )
        
        # Add each choice dynamically
        for i, sq in enumerate(sqs):
            prompt += f"\nChoice {i + 1}: {sq}"

        if self.count < 5:
            print(f"\nPositive Neighbor Selection Prompt Example: {self.count}\n", prompt)

        if self.api_key is None:
            return qs[0]
        if self.args.running_method == 'GCDLLMs_w_cluster_alignment':
            return qs[0]
        try:
            if 'llama' in self.args.llm:
                import boto3
                request = {
                    "prompt": prompt,
                    "max_gen_len": 50,
                    "temperature": 0,
                    "top_p": 1.0,
                }
                bedrock_rt = boto3.client("bedrock-runtime", "us-west-2")
                # Convert the native request to JSON.
                request = json.dumps(request)
                response = bedrock_rt.invoke_model(modelId=self.args.llm, body=request)
                # Decode the response body.
                model_response = json.loads(response["body"].read())
                # Extract and print the response text.
                response_text = model_response["generation"]
                choices_content = response_text
            elif 'claude' in self.args.llm:
                import boto3
                import time
                from botocore.exceptions import ClientError
                bedrock_rt = boto3.client("bedrock-runtime", "us-west-2")
                retries = 0
                max_retries = 10  # maximum number of retries
                backoff_time = 1  # initial backoff time in seconds

                while retries < max_retries:
                    try:
                        response = bedrock_rt.invoke_model(
                            modelId=self.args.llm,
                            body=json.dumps(
                                {
                                    "anthropic_version": "bedrock-2023-05-31",
                                    "max_tokens": 50,
                                    "system": "You are a helpful assistant.",
                                    "messages": [{"role": "user", "content": prompt}],
                                    "temperature": 0,
                                    "top_p": 1.0,
                                }
                            )
                        )
                        response_body = json.loads(response.get('body').read())

                        # Ensure the response structure is as expected
                        if "content" in response_body and isinstance(response_body["content"], list) and len(response_body["content"]) > 0:
                            choices_content = response_body["content"][0].get("text", "")
                            if isinstance(choices_content, str) and choices_content.strip():  # Check if the text is a non-empty string
                                break  # Valid response received, exit the retry loop
                            else:
                                print("Empty or invalid text returned by Claude. Retrying.")
                        else:
                            print("Unexpected response structure from Claude. Retrying.")
                        
                    except ClientError as e:
                        if e.response['Error']['Code'] == 'ThrottlingException':
                            retries += 1
                            print(f"ThrottlingException encountered. Retry {retries}/{max_retries}. Waiting for {backoff_time} seconds.")
                            time.sleep(backoff_time)
                            backoff_time *= 2  # Exponential backoff
                        else:
                            raise e  # Re-raise other exceptions
                else:
                    # If the loop exits without breaking, it means all retries failed
                    print("Failed to get a valid response from Claude after multiple retries. Using fallback.")
                    choices_content = None
            else:
                completion = openai.ChatCompletion.create(
                    model=self.args.llm,  # 'gpt-4o-mini', # "gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Set to 0 to remove randomness
                    top_p=1.0,        # Use top_p sampling with the full range of tokens
                    n=1,               # Number of responses to generate
                    max_tokens=50     # Set a lower max_tokens value to limit response length and avoid timeout
                )
                # Check the choices dynamically
                choices_content = completion.choices[0].message['content']
            if self.count < 5:
                print(f"\nPositive Neighbor Selection Completion Example: {self.count}\n", choices_content)
            for i in range(len(sqs)):
                choice_str = f'Choice {i + 1}'
                # Match ' Choice X' at the start or followed by a colon and any characters
                if re.search(rf'\b{choice_str}\b(:\s*\w*)?', choices_content):
                    result = qs[i]
                    break
            else:
                result = qs[0]  # Default return value if no choice matches
            return result

        except Exception as e:
            print(e)  # This will print the actual exception message
            return qs[0]


    def query_llm_cluster_instance(self, anchor_text, topk_cluster_name, topk_cat_indices):

        # Construct the base of the prompt
        prompt = (
            f"Select the category that better corresponds with the Query in terms of {self.args.task}. "
            "Please respond in the format 'Choice [number]' without explanation, e.g., 'Choice 1', 'Choice 2', etc."
            "\nQuery: " + anchor_text
        )

        # Add each choice dynamically
        for i, cluster_name in enumerate(topk_cluster_name):
            prompt += f"\nChoice {i + 1}: {cluster_name}"

        if self.count < 5:
            print(f"\nCluster Description Selection Prompt Example: {self.count}\n ", prompt)
        
        if self.api_key is None:
            return topk_cat_indices[0]
        try:
            if 'llama' in self.args.llm:
                import boto3
                request = {
                    "prompt": prompt,
                    "max_gen_len": 50,
                    "temperature": 0,
                    "top_p": 1.0,
                }
                bedrock_rt = boto3.client("bedrock-runtime", "us-west-2")
                # Convert the native request to JSON.
                request = json.dumps(request)
                response = bedrock_rt.invoke_model(modelId=self.args.llm, body=request)
                # Decode the response body.
                model_response = json.loads(response["body"].read())
                # Extract and print the response text.
                response_text = model_response["generation"]
                choices_content = response_text
            elif 'claude' in self.args.llm:
                import boto3
                import time
                from botocore.exceptions import ClientError
                bedrock_rt = boto3.client("bedrock-runtime", "us-west-2")
                retries = 0
                max_retries = 10  # maximum number of retries
                backoff_time = 1  # initial backoff time in seconds

                while retries < max_retries:
                    try:
                        response = bedrock_rt.invoke_model(
                            modelId=self.args.llm,
                            body=json.dumps(
                                {
                                    "anthropic_version": "bedrock-2023-05-31",
                                    "max_tokens": 50,
                                    "system": "You are a helpful assistant.",
                                    "messages": [{"role": "user", "content": prompt}],
                                    "temperature": 0,
                                    "top_p": 1.0,
                                }
                            )
                        )
                        response_body = json.loads(response.get('body').read())

                        # Ensure the response structure is as expected
                        if "content" in response_body and isinstance(response_body["content"], list) and len(response_body["content"]) > 0:
                            choices_content = response_body["content"][0].get("text", "")
                            if isinstance(choices_content, str) and choices_content.strip():  # Check if the text is a non-empty string
                                break  # Valid response received, exit the retry loop
                            else:
                                print("Empty or invalid text returned by Claude. Retrying.")
                        else:
                            print("Unexpected response structure from Claude. Retrying.")
                        
                    except ClientError as e:
                        if e.response['Error']['Code'] == 'ThrottlingException':
                            retries += 1
                            print(f"ThrottlingException encountered. Retry {retries}/{max_retries}. Waiting for {backoff_time} seconds.")
                            time.sleep(backoff_time)
                            backoff_time *= 2  # Exponential backoff
                        else:
                            raise e  # Re-raise other exceptions
                else:
                    # If the loop exits without breaking, it means all retries failed
                    print("Failed to get a valid response from Claude after multiple retries. Using fallback.")
                    choices_content = None
            else:
                completion = openai.ChatCompletion.create(
                    model=self.args.llm,  # 'gpt-4o-mini', # "gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Set to 0 to remove randomness
                    top_p=1.0,        # Use top_p sampling with the full range of tokens
                    n=1,               # Number of responses to generate
                    max_tokens=50     # Set a lower max_tokens value to limit response length and avoid timeout
                )
                # Check the choices dynamically
                choices_content = completion.choices[0].message['content']
            if self.count < 5:
                print(f"\nCluster Description Selection Completion Example: {self.count} \n", choices_content)
            for i in range(len(topk_cat_indices)):
                choice_str = f'Choice {i + 1}'
                # Match ' Choice X' at the start or followed by a colon and any characters
                if re.search(rf'\b{choice_str}\b(:\s*\w*)?', choices_content):
                    result = topk_cat_indices[i]
                    break
            else:
                result = topk_cat_indices[0]  # Default return value if no choice matches
            return result

        except Exception as e:
            print(e)  # This will print the actual exception message
            return topk_cat_indices[0] 