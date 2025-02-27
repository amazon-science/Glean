from argparse import ArgumentParser

def init_model():
    parser = ArgumentParser()
    
    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    
    parser.add_argument("--api_key", type=str, default=None,
                        help="The key for Openai API.")
    
    parser.add_argument("--save_results_path", type=str, default='outputs',
                        help="The path to save results.")

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="The path or name for the pre-trained bert model.")

    parser.add_argument("--tokenizer", default="bert-base-uncased", type=str,
                        help="The path or name for the tokenizer")
    
    parser.add_argument("--feat_dim", default=768, type=int,
                        help="Bert feature dimension.")
    
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Warmup proportion for optimizer.")
    
    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id.")

    parser.add_argument("--save_premodel", action="store_true", help="Save Pretrained model.")

    parser.add_argument("--save_model", action="store_true", help="Save Pretrained model.")

    parser.add_argument("--disable_pretrain", action="store_true", help="Disable Pretrain.")

    parser.add_argument("--save_model_path", default='./model', type=str,
                        help="Path to save model checkpoints. Set to None if not save.")
    
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="Name of dataset.")
                        
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed.")

    parser.add_argument("--method", type=str, default='CLNN',
                        help="The name of method.")
    
    parser.add_argument("--rtr_prob", default=0.25, type=float,
                        help="Probability for random token replacement")

    parser.add_argument("--pretrain_batch_size", default=64, type=int,
                        help="Batch size for pre-training")

    parser.add_argument("--train_batch_size", default=96, type=int,
                        help="Batch size for training.")    # original: 128
    
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--wait_patient", default=20, type=int,
                        help="Patient steps for Early Stop in pretraining.") 

    parser.add_argument("--num_pretrain_epochs", default=100, type=int,
                        help="The pre-training epochs.")

    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="The training epochs.")

    parser.add_argument("--lr_pre", default=5e-5, type=float,
                        help="The learning rate for pre-training.")
    
    parser.add_argument("--lr", default=1e-5, type=float,
                        help="The learning rate for training.")
        
    parser.add_argument("--temp", default=0.07, type=float,
                        help="Temperature for contrastive loss")

    parser.add_argument("--view_strategy", default="rtr", type=str,
                        help="Choose from rtr|shuffle|none")

    parser.add_argument("--report_pretrain", action="store_true",
                        help="Enable reporting performance right after pretrain")

    parser.add_argument("--topk", default=50, type=int,
                        help="Select topk nearest neighbors")

    parser.add_argument("--grad_clip", default=1, type=float,
                        help="Value for gradient clipping.")
    
    # DistillLoss
    parser.add_argument('--memax_weight', type=float, default=1)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=5, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument("--ce_weight", default=1, type=float, help="The weight of cross entropy loss.")
    parser.add_argument("--cl_weight", default=1, type=float, help="The weight of contrative loss.")
    parser.add_argument('--sup_weight', type=float, default=1, help='The weight for supervised loss.')
    parser.add_argument("--weight_ce_unsup", default=0, type=float, help="The weight of supervised pseudo oracle cross entropy loss.")

    parser.add_argument('--print_freq', default=10, type=int)

    # Method
    parser.add_argument("--architecture", default="SimGCD", type=str, help="Choose method and architecture from SimGCD|Loop")
    
    parser.add_argument("--running_method", default=" ", type=str, help="Specify the running method.")

    parser.add_argument("--experiment_name", default=" ", type=str, help="Specify massive experiment name.")

    # Setting

    parser.add_argument("--known_cls_ratio", default=0.1, type=float, required=True, help="The ratio of known classes.")

    parser.add_argument("--label_setting", default="ratio", type=str, help="Choose from shot|ratio")

    parser.add_argument('--labeled_shot', default=10, type=int)

    parser.add_argument("--labeled_ratio", default=0.1, type=float, help="The ratio of labeled samples.")

    parser.add_argument('--query_samples', default=500, type=int)

    parser.add_argument("--options", default=5, type=int, help="# Options in querying llm.")
    
    parser.add_argument("--update_per_epoch", default=1, type=int,
                        help="Update neighbor contrastive learning data after certain amount of epochs")

    # Cluster Interpritability
    parser.add_argument("--interpret_sampling_strategy", default="nearest_center", type=str, help="Choose from random|nearest_center|nearest_sub_kmeans_centriods")
    parser.add_argument("--interpret_num_representatives", default=10, type=int, help="# representatives from each cluster to query llm.")
    parser.add_argument("--llm", default="gpt-4o-mini", type=str, help="Choose from gpt-3.5-turbo|gpt-4o-mini|gpt-4o|gpt-4-turbo|gpt-4")


    # Self-Adaptive Querying
    parser.add_argument("--sampling_strategy", default="highest", type=str, help="Choose from random|highest|equal_highest|adaptive_difficulty")
    parser.add_argument("--allocation_degree", default=1, type=float, help="degree of convexity or concavity for the allocation function")

    # Cluster Instance Alignment Learning
    parser.add_argument("--weight_cluster_instance_cl", default=0, type=float, help="The weight of alignment loss.")    
    parser.add_argument("--options_cluster_instance_ratio", default=0.5, type=float, help="# Options in querying llm.")

    # LLM Feedback Caching & Replaying
    parser.add_argument("--feedback_cache", action="store_true", help="Save all feedback.")

    # LLM Feedback Enhancement and Filtering for Instance-Level Feedback
    parser.add_argument("--flag_demo", action="store_true", help="Enable demo in prompt.")      # default: False
    parser.add_argument("--known_demo_num_per_class", default=5, type=int, help="# demo per known class.")
    parser.add_argument("--flag_filtering", action="store_true", help="Enable feedback filtering.")   # default: False
    parser.add_argument("--filter_threshold", default=0.8, type=float, help="Threshold for filtering feedback.")

    # LLM Feedback Enhancement and Filtering for Cluster-Level Feedback
    parser.add_argument("--flag_demo_c", action="store_true", help="Enable demo in prompt.")      # default: False
    parser.add_argument("--known_demo_num_per_class_c", default=2, type=int, help="# demo per known class.")
    parser.add_argument("--flag_filtering_c", action="store_true", help="Enable feedback filtering.")   # default: False
    parser.add_argument("--filter_threshold_c", default=0.85, type=float, help="Threshold for filtering feedback.")

    # args.flag_demo = True
    # args.known_demo_num_per_class = 5
    # args.flag_filtering = True
    # args.filter_threshold = 0.8

    # args.flag_demo_c = True
    # args.known_demo_num_per_class_c = 2
    # args.flag_filtering_c = True
    # args.filter_threshold_c = 0.8

    # Ablation
    parser.add_argument("--prompt_ablation", default="full", type=str, help="Choose from full|wo_demo|wo_name|wo_description")
    parser.add_argument("--component_ablation", default="full", type=str, help="Choose from full|wo_ce|wo_cl_1|wo_cl_all|wo_instance_feedback|wo_cluster_feedback|wo_both_feedback")

    return parser
