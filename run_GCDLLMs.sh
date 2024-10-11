#!/usr/bin/env bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

for dataset in 'banking' 'clinc' 'stackoverflow' 
do
	for seed in 42
	do
		for known_cls_ratio in 0.1 # 0.05 0.1 0.25 0.5 
		do
			for labeled_ratio in 0.1
			do
				for query_samples in 500 # 1 50 100 300 500 1000 2000
				do
					for sampling_strategy in 'highest' # 'random' 'highest' 'loop'
					do
						for options in 5 # 1 2 3 5 10 20 50
						do	
							for options_cluster_instance_ratio in 0.5 # 0.1 0.25, 0.5, 0.75 1
							do
								for weight_cluster_instance_cl in 0.05 0.1 # 0 0.001 0.05 0.1 0.5 1
								do		
									for llm in gpt-4o-mini # gpt-3.5-turbo gpt-4o-mini gpt-4o gpt-4-turbo gpt-4
									# gpt-3.5-turbo gpt-4o-mini gpt-4o gpt-4-turbo gpt-4
									# meta.llama3-1-70b-instruct-v1:0 meta.llama3-1-8b-instruct-v1:0 
									# anthropic.claude-3-5-sonnet-20240620-v1:0 anthropic.claude-3-opus-20240229-v1:0 anthropic.claude-3-sonnet-20240229-v1:0 anthropic.claude-3-haiku-20240307-v1:0
									do		
										echo "Dataset: ${dataset}"
										echo "Running with known_cls_ratio=${known_cls_ratio}, labeled_ratio=${labeled_ratio}, query_samples=${query_samples}"
										echo "Running with sampling_strategy=${sampling_strategy}, weight_cluster_instance_cl=${weight_cluster_instance_cl}"
										echo "Running with options=${options}, options_cluster_instance_ratio=${options_cluster_instance_ratio}"
										echo "Running with llm=${llm}"

										python GCDLLMs.py \
											--data_dir data \
											--dataset $dataset \
											--known_cls_ratio $known_cls_ratio \
											--labeled_ratio $labeled_ratio \
											--seed $seed \
											--num_train_epochs 25 \
											--lr '1e-5' \
											--save_results_path 'outputs' \
											--view_strategy 'rtr' \
											--save_premodel \
											--experiment_name 'GCDLLMs_ablation_llm_banking' \
											--running_method 'GCDLLMs' \
											--architecture 'Loop' \
											--update_per_epoch 5 \
											--ce_weight 1 \
											--cl_weight 1 \
											--sup_weight 1 \
											--weight_cluster_instance_cl $weight_cluster_instance_cl \
											--weight_ce_unsup 0 \
											--query_samples $query_samples \
											--options $options \
											--train_batch_size 48 \
											--pretrain_batch_size 48 \
											--sampling_strategy $sampling_strategy \
											--options_cluster_instance_ratio $options_cluster_instance_ratio \
											--llm $llm \
											--api_key 'add_your_openai_api_key_here'
									done
								done
							done
						done
					done
				done
			done
		done
	done
done


