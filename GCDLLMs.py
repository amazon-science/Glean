import math
from model import CLBert
from init_parameter import init_model
from dataloader import Data
from mtp import PretrainModelManager
from utils.tools import *
from utils.memory import MemoryBank, fill_memory_bank
from utils.neighbor_dataset import NeighborsDataset
from model import BertForModel
from model import DistillLoss
from transformers import logging, WEIGHTS_NAME
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import warnings
from scipy.spatial import distance as dist
from sklearn.neighbors import NearestNeighbors
import re
import openai
warnings.filterwarnings('ignore')
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class ModelManager:
    def __init__(self, args, data, pretrained_model=None):
        set_seed(args.seed)
        self.args = args
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = data.num_labels
        self.model = CLBert(args, args.bert_model, device=self.device, num_labels=data.num_labels)
        # self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        
        # Pretraining
        if pretrained_model is None:
            pretrained_model = BertForModel(args.pretrain_dir, num_labels=data.n_known_cls)
            # if os.path.exists(args.pretrain_dir):
            #     pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model
        self.load_pretrained_model()
        
        if args.cluster_num_factor > 1:
            self.num_labels = self.predict_k(args, data) 
        else:
            self.num_labels = data.num_labels
        
        self.num_train_optimization_steps = int(len(data.train_semi_dataset) / args.train_batch_size) * args.num_train_epochs
        
        self.optimizer, self.scheduler = self.get_optimizer(args)
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)
        args.num_training_rounds = math.ceil(args.num_train_epochs / args.update_per_epoch)
        args.current_training_round = 0
        print('\nNumber of Training Rounds: ', args.num_training_rounds)

    def custom_collate(self, batch):
        batch_dict = {}
        for key in batch[0]:
            if key in ['pos_cluster_idx', 'neg_cluster_idx']:
                batch_dict[key] = [d[key] for d in batch]
            else:
                batch_dict[key] = default_collate([d[key] for d in batch])
        return batch_dict

    def get_neighbor_dataset(self, args, data, indices, query_index, pred, p, cluster_name=None, init=False):
        if init or not args.feedback_cache:
            self.di_all, self.di_all_pos_cluster_idx, self.di_all_neg_cluster_idx = {}, {}, {}
        else:
            print('\nLoad LLM feedback from cache')
            self.di_all = self.dataset.di_all
            self.di_all_pos_cluster_idx = self.dataset.di_all_pos_cluster_idx
            self.di_all_neg_cluster_idx = self.dataset.di_all_neg_cluster_idx
        # print the number of keys in di_all
        self.num_cached_feedback = len(self.di_all)
        print('\n Number of Loaded LLM feedback: ', len(self.di_all))

        dataset = NeighborsDataset(args, data.train_semi_dataset, indices, query_index, pred, p, cluster_name=cluster_name,
                                   di_all=self.di_all, di_all_pos_cluster_idx=self.di_all_pos_cluster_idx, di_all_neg_cluster_idx=self.di_all_neg_cluster_idx)
        self.train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=self.custom_collate)
        self.dataset = dataset

    def get_neighbor_inds(self, args, data, km):
        memory_bank = MemoryBank(args, len(data.train_semi_dataset), args.feat_dim, len(data.all_label_list), 0.1)
        fill_memory_bank(data.train_semi_dataloader, self.model, memory_bank)
        indices, query_index, p = memory_bank.mine_nearest_neighbors(args.topk, km.labels_, km.cluster_centers_)
        return indices, query_index, p
    
    def get_adjacency(self, args, inds, neighbors, targets):
        """get adjacency matrix"""
        adj = torch.zeros(inds.shape[0], inds.shape[0])
        for b1, n in enumerate(neighbors):
            adj[b1][b1] = 1
            for b2, j in enumerate(inds):
                if j in n:
                    adj[b1][b2] = 1 # if in neighbors
                # if (targets[b1] == targets[b2]) and (targets[b1]>=0) and (targets[b2]>=0):
                if (targets[b1] == targets[b2]) and (inds[b1] <= args.num_labeled_examples) and (inds[b2] <= args.num_labeled_examples):
                    adj[b1][b2] = 1 # if same labels
                    # this is useful only when both have labels
                    # how to ensure there is no label leakage for unlabeled data? inds[b1] <= args.num_labeled_examples?
        return adj

    def evaluation(self, args, data, save_results=True, plot_cm=True):
        """final clustering evaluation on test set"""
        print('\n### Evaluation ###\n')
        # get features
        feats_test, labels, logits = self.get_features_labels(data.test_dataloader, self.model, args, return_logit=True)
        feats_test = feats_test.cpu().numpy()

        # clustering result
        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats_test)
        y_pred = km.labels_
        y_true = labels.cpu().numpy()
        results = clustering_score(y_true, y_pred, data.known_lab)
        print('results',results)
        self.test_results = results

        # save results
        if save_results:
            self.save_results(args, result_source='clustering')

    def train(self, args, data):
        args.evaluation_epoch = 0
        # self.evaluation(args, data, save_results=False, plot_cm=False)

        if isinstance(self.model, nn.DataParallel):
            criterion = self.model.module.loss_cl
            ce = self.model.module.loss_ce
        else:
            criterion = self.model.loss_cl
            ce = self.model.loss_ce

        if args.weight_ce_unsup > 0:
            cluster_criterion = DistillLoss(
                                args.warmup_teacher_temp_epochs,
                                args.num_train_epochs,
                                args.warmup_teacher_temp,
                                args.teacher_temp,
                            )
            
        # Ablation Study: full|wo_ce|wo_cl_1|wo_cl_all|wo_instance_feedback|wo_cluster_feedback|wo_both_feedback
        if args.component_ablation == 'wo_ce':
            args.ce_weight = 0
            args.sup_weight = 0
            args.weight_ce_unsup = 0
        elif args.component_ablation == 'wo_cl_1':
            args.cl_weight = 0
        elif args.component_ablation == 'wo_cl_all':
            args.cl_weight = 0
            args.weight_cluster_instance_cl = 0
        elif args.component_ablation == 'wo_instance_feedback':
            print('wo_instance_feedback, disabled instance feedback in the neighbor_dataset')
        elif args.component_ablation == 'wo_cluster_feedback':
            args.weight_cluster_instance_cl = 0
        elif args.component_ablation == 'wo_both_feedback':
            args.weight_cluster_instance_cl = 0
            print('wo_both_feedback, disabled instance feedback in the neighbor_dataset')
        else:
            print('Full components in GCDLLMs enabled')
            
        # Obtain initial features, labels, logits
        feats_gpu, labels, logits = self.get_features_labels(data.train_semi_dataloader, self.model, args, return_logit=True)
        feats = feats_gpu.cpu().numpy()

        # Perform K-Means Clustering and extract cluster centers
        km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)

        # Category Characterization
        if self.args.weight_cluster_instance_cl > 0:
            cluster_name = self.category_characterization(data, km, feats_gpu)
            # cluster_name = None
            # cluster_name = ['(Category Name: exchange_rate_inquiry, Description: This category involves inquiries or concerns related to the current exchange rate and discrepancies in the exchange rates provided.)', '(Category Name: [statement_discrepancy], Description: [Category covers issues related to discrepancies or irregularities in statements or transaction records.])', '(Category Name: Supported_Cards, Description: Inquiries about the availability and usability of Visa and Mastercard with the service provider.)', '(Category Name: [transfer_processing_time], Description: [Inquiries about the processing duration for various types of transfers.])', '(Category Name: [card_usage_queries], Description: [Questions related to using cards for transactions, withdrawals, and inquiries.])', "(Category Name: pending_transfer, Description: Handling inquiries related to transfers that have not yet been processed or displayed in the recipient's account.)", '(Category Name: Verify_Top-Up, Description: Concerns related to verifying or understanding the verification process for top-ups.)', '(Category Name: virtual_card_requests, Description: Handling requests for virtual cards including regular and disposable ones)', "(Category Name: Account Details Update, Description: Handling inquiries related to editing or changing personal details on the user's account.)", '(Category Name: declined_card_transactions, Description: Inquiries about the reasons for card transactions being declined.)', '(Category Name: top_up_failed, Description: Inquiry about unsuccessful top-up transactions)', '[Category Name: Exchange Currency in App, Description: Inquiries related to the process of exchanging currencies within the app.]', '(Category Name: extra_charge_inquiries, Description: Inquiries about unexpected fees or charges for various financial transactions)', "(Category Name: extra_charge_on_statement, Description: Addressing unexpected or additional charges appearing on the user's account statement related to transactions)", '(Category Name: pin_change, Description: Inquiries related to changing or resetting personal identification numbers.)', '(Category Name: Pending_Transaction_Inquiry, Description: Handling user queries related to transactions that are in a pending state or not reflected in the account balance immediately after initiation.)', '(Category Name: Currency Exchange, Description: Handling queries related to currency exchange rates, costs, and potential additional charges.)', '(Category Name: Security Concerns, Description: Inquiries related to theft or loss of personal belongings containing sensitive information.)', '(Category Name: Source of Funds, Description: Addressing inquiries related to identifying the origin of funds in the account.)', '(Category Name: [fee_inquiry], Description: [Questions related to charges or fees associated with transfers or transactions.])', '(Category Name: Payment Processing, Description: Handling issues related to card payments, pending transactions, and failed payment attempts.)', '(Category Name: reported_lost_or_missing_card, Description: Handling inquiries and providing assistance related to lost or missing cards.)', '(Category Name: exchange_rate_issues, Description: Issues related to incorrect exchange rates for transactions or cash withdrawals abroad)', "(Category Name: why_verify_identity, Description: Issues related to verifying or proving one's identity.)", '(Category Name: [usage_restrictions], Description: [Questions related to limitations or restrictions on the use of disposable virtual cards.])', '(Category Name: card_activation, Description: Inquiries related to activating a new card.)', '(Category Name: missing_pin, Description: Issues related to the unavailability or loss of card PINs.)', '(Category Name: pending_transfer, Description: Handling issues related to pending transfers or delayed refunds)', '(Category Name: pin_related_issues, Description: Discussions related to issues with personal identification numbers (PINs) such as account blockages and resets.)', '(Category Name: top_up_methods, Description: Inquiries related to the various methods or options available for topping up an account or service.)', '(Category Name: card_ordering, Description: Inquiries and requests related to ordering physical or virtual cards.)', '(Category Name: top_up_with_cheque, Description: Category related to inquiries about topping up or adding funds using cheques.)', '(Category Name: virtual_card_inquiry, Description: Inquiries about obtaining or accessing virtual cards)', '(Category Name: why_verify_identity, Description: Inquiry about the reasons behind the verification of identity.)', '(Category Name: Fraudulent Transactions, Description: Handling and investigating unauthorized or suspicious transactions on an account.)', '(Category Name: transaction_fees, Description: Inquiries about the charges or fees associated with making transfers.)', '(Category Name: account_for_minors, Description: Inquiries related to opening accounts for children or minors.)', '(Category Name: [card_activation], Description: [Assistance with activating or tracking a card])', "(Category Name: extra_charge_on_statement, Description: Addresses inquiries about additional charges appearing on the user's financial statement.)", '(Category Name: [disposable_card_limits], Description: [Inquiring about the number of transactions allowed with a disposable card.])', '(Category Name: [payment_processing], Description: [Handling issues related to transactions, transfers, and payments that are pending, declined, or not processed successfully.])', '(Category Name: Extra_Charge_on_Card_Payment, Description: Inquiries about additional charges incurred when using a card for payment.)', '(Category Name: card_ordering, Description: Information related to ordering a new card.)', '(Category Name: [declined_card_payment], Description: [Category for inquiries regarding declined card payments])', '(Category Name: Refund_Inquiry, Description: Inquiries related to requesting refunds or returns for items purchased.)', '(Category Name: Cash Withdrawal Issues, Description: Handling issues related to discrepancies in cash withdrawal amounts from ATMs.)', '(Category Name: [atm_usage], Description: [Inquiries related to the use of cards at ATMs, including changing PINs and accepted ATM locations.])', '(Category Name: [disputed_transaction], Description: [Category for issues related to disputed or unclear transactions, including charges or withdrawals that the user does not recognize or that have not gone through successfully.])', '(Category Name: pending_transfer, Description: Inquiries about transfers that are still pending)', '(Category Name: card_fees, Description: Questions related to fees associated with physical cards, including top-up fees and charges for physical card issuance.)', "(Category Name: account_closure, Description: Managing user's request to close their account.)", '(Category Name: [currency_preferences], Description: [Handling requests related to receiving salary or funds in different currencies.])', '(Category Name: [top_up_options], Description: [Concerns and inquiries related to top-up functionality such as limits and customization.])', '(Category Name: Unauthorized Transactions, Description: Handling issues related to unauthorized transactions or payments on the account.)', '(Category Name: [exchange_rates_inquiry], Description: [Inquiries about how the platform determines its exchange rates.])', '(Category Name: Transfer Duration, Description: Inquiries about the time taken for transfers to be completed.)', '(Category Name: Transfer Issues, Description: Handling inquiries and providing explanations for failed or declined transfers.)', '(Category Name: top_up_methods, Description: Questions related to various payment methods for topping up an account.)', "(Category Name: Deposits and Transfers, Description: Handling inquiries related to depositing or transferring money into the user's account.)", '(Category Name: supported_cards_and_currencies, Description: Inquiring about the various fiat currencies and types of cards supported.)', '(Category Name: why_verify_identity, Description: Inquiries about the necessity and process of verifying identity)', '(Category Name: [transaction_fees], Description: [Inquiries related to charges, fees, or additional costs associated with transactions.])', '(Category Name: card_issue, Description: Issues related to card functionality and operation.)', '(Category Name: card_delivery_inquiry, Description: Inquiries regarding the time it will take to receive a new card)', '(Category Name: Pending Transfers, Description: Category related to inquiries about pending or declined transfers.)', '(Category Name: [card_activation_and_top_up], Description: [Questions related to activating new cards and topping up using cards.])', '(Category Name: refund_status_inquiry, Description: Inquiries about missing or not visible refunds in statements)', '(Category Name: adding_funds_via_transfer, Description: Inquiries related to adding money to the account using bank transfers or electronic fund transfers.)', '(Category Name: pending_transfer, Description: Dealing with delayed or pending payment transfers and transactions.)', "(Category Name: pending_cash_deposit, Description: Inquiries about cash deposits that have not yet reflected in the user's account)", '(Category Name: Payment Issues, Description: Category related to problems or concerns with making payments using cards such as declined transactions, missing funds after a top-up, or issues with online purchases.)', '(Category Name: Account Eligibility, Description: Inquiries related to the minimum age requirement for opening an account.)', '(Category Name: refund_request, Description: Handling requests from users who want to get a refund for a purchase they made but have not received the item or encountered issues with the transaction.)', '(Category Name: pending_status_inquiry, Description: Addressing inquiries about pending transactions)', '(Category Name: Update Personal Information, Description: Handling requests to modify personal details such as address or contact information.)', '(Category Name: Verification_Code_Issues, Description: Handling inquiries related to finding and using verification codes for account activities, such as top-up card verification.)', '(Category Name: declined_cash_withdrawal, Description: Inquiries about the inability to withdraw cash from an ATM)']
            # print('\nAll Category Names and description:', cluster_name)
            print('len(cluster_name)',len(cluster_name))

            label_names = list(args.label_map_semi.keys())
            print('label_names',label_names)   
            measure_interpretability(cluster_name, label_names, args)  
        else:
            cluster_name = None

        # Get Neighbor Dataset
        args.current_training_round += 1
        print('\nCurrent Training Round: ', args.current_training_round)
        indices, query_index, p = self.get_neighbor_inds(args, data, km)
        self.get_neighbor_dataset(args, data, indices, query_index, km.labels_, p, cluster_name=cluster_name, init=True)


        # Training
        labelediter = iter(data.train_labeled_dataloader)
        for epoch in range(int(args.num_train_epochs)):
            print(f'\n\nTraining Epoch: [{epoch+1}/{args.num_train_epochs}]')
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for _, batch in enumerate(self.train_dataloader):
                # 1. load data
                anchor = tuple(t.to(self.device) for t in batch["anchor"]) # anchor data
                neighbor = tuple(t.to(self.device) for t in batch["neighbor"]) # neighbor data
                pos_neighbors = batch["possible_neighbors"] # all possible neighbor inds for anchor
                data_inds = batch["index"] # data ind

                # 2. get adjacency matrix
                adjacency = self.get_adjacency(args, data_inds, pos_neighbors, batch["target"]) # (bz,bz)

                # 3. get augmentations
                if args.view_strategy == "rtr":
                    X_an = {"input_ids":self.generator.random_token_replace(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":self.generator.random_token_replace(neighbor[0].cpu()).to(self.device), "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                    X_an_2 = {"input_ids":self.generator.random_token_replace(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                elif args.view_strategy == "shuffle":
                    X_an = {"input_ids":self.generator.shuffle_tokens(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":self.generator.shuffle_tokens(neighbor[0].cpu()).to(self.device), "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                    X_an_2 = {"input_ids":self.generator.shuffle_tokens(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                elif args.view_strategy == "none":
                    X_an = {"input_ids":anchor[0], "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":neighbor[0], "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                    X_an_2 = {"input_ids":anchor[0], "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                else:
                    raise NotImplementedError(f"View strategy {args.view_strategy} not implemented!")
                
                # 4. compute loss and update parameters
                with torch.set_grad_enabled(True):
                    pstr = ''
                    out_an = self.model(X_an)
                    out_ng = self.model(X_ng)
                    out_an_2 = self.model(X_an_2)

                    ## Contrastive Loss
                    f_pos = torch.stack([out_an["features"], out_ng["features"]], dim=1) # shape required by SupConLoss: [bs, n_views, feat_dim]
                    loss_cl = criterion(f_pos, mask=adjacency, temperature=args.temp)

                    loss_cl_cluster_instance = 0
                    # Cluster-Instance Alignment Loss
                    if args.weight_cluster_instance_cl > 0:
                        pos_cluster_idx_noisy = batch["pos_cluster_idx"] # positive cluster description
                        neg_cluster_idx_noisy = batch["neg_cluster_idx"] # negative cluster description
                        
                        # only take valid cluster idx: None is invalid
                        pos_cluster_idx = [i for i in pos_cluster_idx_noisy if i is not None]
                        neg_cluster_idx = [i for i in neg_cluster_idx_noisy if i is not None]

                        # tokenize all cluster descriptions
                        all_cluster_des_t =  self.tokenizer(cluster_name, padding=True, truncation=True, return_tensors="pt", max_length=64) # shape: [num_clusters, seq_len]

                        # Feed all cluster descriptions at once to the model
                        X_all_cluster_des = {
                            "input_ids": all_cluster_des_t["input_ids"].to(self.device), 
                            "attention_mask": all_cluster_des_t["attention_mask"].to(self.device), 
                            "token_type_ids": all_cluster_des_t["token_type_ids"].to(self.device)
                        }
                        feat_all_cluster_des = self.model(X_all_cluster_des)["features"]

                        # only take the features corresponding to query with valid cluster descriptions
                        an_w_cluster_des_feat = out_an["features"] 
                        an_w_cluster_des_feat = an_w_cluster_des_feat[[i for i in range(len(pos_cluster_idx_noisy)) if pos_cluster_idx_noisy[i] is not None]] # torch.Size([..., 768])

                        pos_cluster_feat = [feat_all_cluster_des[i] for i in pos_cluster_idx]   # torch.Size([768]) x ...
                        neg_cluster_feat = [feat_all_cluster_des[i] for i in neg_cluster_idx]   # torch.Size([..., 768]) x ...

                        # compute cluster description and instance alignment loss
                        if len(an_w_cluster_des_feat) > 0 and len(pos_cluster_feat) > 0:              
                            for i in range(len(an_w_cluster_des_feat)):
                                # compute similarity scores
                                sim_positive = F.cosine_similarity(an_w_cluster_des_feat[i].unsqueeze(0), pos_cluster_feat[i].unsqueeze(0), dim=1) / args.temp
                                sim_negatives = F.cosine_similarity(an_w_cluster_des_feat[i].unsqueeze(0), neg_cluster_feat[i], dim=1) / args.temp
                                
                                # calculate the numerator (similarity of positive pair)
                                numerator = torch.exp(sim_positive)

                                # calculate the denominator (sum of similarities with all negatives)
                                denominator = torch.exp(sim_negatives).sum()

                                # compute the alignment loss
                                loss = -torch.log(numerator / denominator)
                                loss_cl_cluster_instance += loss
                            # normalize the loss
                            loss_cl_cluster_instance /= len(an_w_cluster_des_feat)
                    # print('loss_cl_cluster_instance:', loss_cl_cluster_instance)


                    ## Parametric Classification Loss
                    cluster_loss = 0
                    if args.weight_ce_unsup > 0:
                        # Unsupervised Self-Distillation Loss for All Data
                        student_out = torch.cat([out_an["logits"], out_an_2["logits"]], dim=0)  # shape required by DistillLoss: [n_viewsxbs, n_cls]
                        teacher_out = student_out.detach()

                        cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                        avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                        me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                        pstr += f'un_ce_loss: {cluster_loss.item():.2f} '
                        pstr += f'reg_loss: {me_max_loss.item():.4f} '
                        cluster_loss += args.memax_weight * me_max_loss


                    # Supervised Classification Loss for Labeled Data
                    try:
                        batch = labelediter.next()
                    except StopIteration:
                        labelediter = iter(data.train_labeled_dataloader)
                        batch = labelediter.next()
                    batch = tuple(t.to(self.device) for t in batch)
                    X_an = {"input_ids":batch[0], "attention_mask":batch[1], "token_type_ids":batch[2]}

                    logits = self.model(X_an)["logits"]
                    loss_ce_sup = ce(logits, batch[3])


                    loss_ce = args.sup_weight * loss_ce_sup 
                    loss_ce += args.weight_ce_unsup * cluster_loss

                    loss = loss_ce * args.ce_weight + loss_cl * args.cl_weight + loss_cl_cluster_instance * args.weight_cluster_instance_cl
                    pstr += f'sup_ce_loss: {loss_ce_sup.item():.2f} '
                    pstr += f'\t loss_ce: {loss_ce.item():.2f} '
                    pstr += f'loss_cl: {loss_cl.item():.2f} '
                    pstr += f'loss_cl_cluster_instance: {loss_cl_cluster_instance.item():.2f} ' if loss_cl_cluster_instance != 0 else ""
                    pstr += f'loss: {loss.item():.2f} '

                    tr_loss += loss.item()
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    nb_tr_examples += anchor[0].size(0)
                    nb_tr_steps += 1

                    if _ % args.print_freq == 0:
                        print(pstr)

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            self.dataset.count = 0
            
            args.evaluation_epoch = epoch
            # update neighbors every several epochs
            if ((epoch + 1) % args.update_per_epoch) == 0 and ((epoch + 1) != int(args.num_train_epochs)):
                self.evaluation(args, data, save_results=False, plot_cm=False)

                # Obtain initial features, labels, logits
                feats_gpu, labels, logits = self.get_features_labels(data.train_semi_dataloader, self.model, args, return_logit=True)
                feats = feats_gpu.cpu().numpy()

                # Perform K-Means Clustering and extract cluster centers
                km = KMeans(n_clusters = self.num_labels, random_state=args.seed).fit(feats)

                if self.args.weight_cluster_instance_cl > 0:
                    # Category Characterization
                    cluster_name = self.category_characterization(data, km, feats_gpu)
                    # print('\nAll Category Names and description:', cluster_name)
                    print('len(cluster_name)',len(cluster_name))
                    label_names = list(args.label_map_semi.keys())
                    print('label_names',label_names)  
                    measure_interpretability(cluster_name, label_names, args)  
                else:
                    cluster_name = None

                # Get Neighbor Dataset
                indices, query_index, p = self.get_neighbor_inds(args, data, km)
                self.get_neighbor_dataset(args, data, indices, query_index, km.labels_, p, cluster_name=cluster_name)



    def category_characterization(self, data, km, feats_gpu):
        print('\n\n### Category Characterization ###\n')
        print('Sampling Strategy:', self.args.interpret_sampling_strategy)
        print('Number of Representatives:', self.args.interpret_num_representatives)
        print('LLM Interpretation Model:', self.args.llm)

        # Sample representative examples from each cluster
        interpret_num_representatives = self.args.interpret_num_representatives
        cluster_centers = torch.tensor(km.cluster_centers_)

        # Ensure km.labels_ is a torch tensor
        km_labels = torch.tensor(km.labels_, dtype=torch.long)

        # 1. Sample K samples nearest to each cluster center
        if self.args.interpret_sampling_strategy == 'nearest_center':
            dis = self.EuclideanDistances(feats_gpu.cpu(), cluster_centers).T
            _, index = torch.sort(dis, dim=1)
            index = index[:, :interpret_num_representatives]
        # 2. Randomly sample K samples from each cluster
        elif self.args.interpret_sampling_strategy == 'random':
            index = []
            for i in range(self.num_labels):
                index.append(torch.where(km_labels == i)[0])
            index = [torch.randperm(len(i))[:interpret_num_representatives] for i in index]
        # 3. Perform sub-clustering using KMeans for each cluster to obtain K sub-clusters and sample 1 samples nearest to each sub-cluster center, similar to the first strategy
        elif self.args.interpret_sampling_strategy == 'nearest_sub_kmeans_centriods':
            # for each cluster, perform sub-clustering using KMeans
            index = []
            for i in range(self.num_labels):
                cluster_feats = feats_gpu[km_labels == i]
                # handle case where there are fewer samples than the number of representatives
                if cluster_feats.size(0) < interpret_num_representatives:
                    interpret_num_representatives = cluster_feats.size(0)
                sub_km = KMeans(n_clusters=interpret_num_representatives, random_state=self.args.seed).fit(cluster_feats.cpu().numpy())
                sub_cluster_centers = torch.tensor(sub_km.cluster_centers_).to(feats_gpu.device)
                dis = self.EuclideanDistances(cluster_feats, sub_cluster_centers).T
                _, sub_index = torch.sort(dis, dim=1)
                sub_indices = sub_index[:, :1].flatten()
                original_indices = torch.where(km_labels == i)[0]
                index.append(original_indices[sub_indices])
            print('Sub-Cluster Index:', index)
        else:
            raise NotImplementedError(f"Sampling strategy {self.args.interpret_sampling_strategy} not implemented!")

        # Assign Names and Description to Clusters:
        cluster_name = []
        example_count = 0   
        for i in range(len(index)):
            query = []
            for j in index[i]:
                query.append(data.train_semi_dataset.__getitem__(j)[0])
            llm_feedback = self.query_llm(query, example_count)
            cluster_name.append(llm_feedback)

            if example_count < 5:
                query_text = []
                query_labels = []
                for j in index[i]:
                    query_text.append(self.tokenizer.decode(data.train_semi_dataset.__getitem__(j)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
                    query_label_index = data.train_semi_dataset.__getitem__(j)[3].item()
                    query_label_name = args.get_label_name_semi[query_label_index]
                    query_labels.append(query_label_name)
                print(f'\nCategory Characterization Examples: {i}')
                print('Query Text:\n', query_text) 
                print('Query Ground Truth Labels:\n', query_labels)
                print('LLM Generated Category Name and Description:\n', llm_feedback)
            example_count += 1

        self.cluster_reprsentatives = index

        # print('\nAll Category Names and description:', cluster_name)
        print('Total Number of category characterization:', len(cluster_name))
        # print('Total Number of words in category characterization:', sum([len(name.split()) for name in cluster_name]))
        
        return cluster_name


    def query_llm(self, query, example_count):
        demo_name = str(list(args.label_map_train.keys()))
        utterances = [self.tokenizer.decode(utt, skip_special_tokens=True, clean_up_tokenization_spaces=True) for utt in query]

        # Construct the prompt with any number of utterances
        if args.prompt_ablation == 'wo_demo':
            prompt = f"Given the following utterances, return a category name and a short category description to summarize the common {args.task} of these utterances in the format (Category Name: [category_name], Description: [description]) without explanation. \n"
        elif args.prompt_ablation == 'wo_name':
            prompt = f"Given the following utterances and examples of some known category names, return a short category description to summarize the common {args.task} of these utterances in the format (Category Description: [description]) without explanation. \n"
            prompt += "Examples of Some Known Category Names: \n" + demo_name
        elif args.prompt_ablation == 'wo_description':
            prompt = f"Given the following utterances and examples of some known category names, return a category name to summarize the common {args.task} of these utterances in the format (Category Name: [category_name]) without explanation. \n"
            prompt += "Examples of Some Known Category Names: \n" + demo_name
        else:
            prompt = f"Given the following utterances and examples of some known category names, return a category name and a short category description to summarize the common {args.task} of these utterances in the format (Category Name: [category_name], Description: [description]) without explanation. \n"
            prompt += "Examples of Some Known Category Names: \n" + demo_name + "\n"

        for i, utterance in enumerate(utterances, 1):
            prompt += f"Utterance {i}: {utterance}\n"

        if example_count < 1:
            print(f'\nCluster Interpretation Prompt Example: {example_count}\n', prompt)

        openai.api_key = self.args.api_key
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
                return response_text
            elif 'claude' in self.args.llm:
                import boto3
                import time
                from botocore.exceptions import ClientError
                bedrock_rt = boto3.client("bedrock-runtime", "us-west-2")
                retries = 0
                max_retries = 10
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
                            output = response_body["content"][0].get("text", "")
                            if isinstance(output, str) and output.strip():  # Check if the text is a non-empty string
                                return output
                            else:
                                print("Empty or invalid text returned by Claude. Using fallback.")
                        else:
                            print("Unexpected response structure from Claude. Using fallback.")
                        
                        # If the response is not valid, use the fallback
                        fallback_text = " | ".join(utterances[:3])
                        return f"Fallback Description: {fallback_text}"

                    except ClientError as e:
                        if e.response['Error']['Code'] == 'ThrottlingException':
                            retries += 1
                            print(f"ThrottlingException encountered. Retry {retries}/{max_retries}. Waiting for {backoff_time} seconds.")
                            time.sleep(backoff_time)
                            backoff_time *= 2  # Exponential backoff
                        else:
                            raise e  # Re-raise other exceptions
                # If all retries fail, return a fallback
                fallback_text = " | ".join(utterances[:3])
                return f"Fallback Description: {fallback_text}"
            else:
                completion = openai.ChatCompletion.create(
                    model= self.args.llm, #'gpt-4o-mini', # "gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Set to 0 to remove randomness
                    top_p=1.0,        # Use top_p sampling with the full range of tokens
                    n=1,               # Number of responses to generate
                    max_tokens=50     # Set a lower max_tokens value to limit response length and avoid timeout
                )
                return completion.choices[0].message['content']
        except Exception as e:
            print(f"LLM query failed with exception: {e}")
            # Return the first three utterances as a fallback
            fallback_text = " | ".join(utterances[:3])
            return f"Fallback Description: {fallback_text}"
        

    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion*self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler
    
    def load_pretrained_model(self):
        """load the backbone of pretrained model"""
        if isinstance(self.pretrained_model, nn.DataParallel):
            pretrained_dict = self.pretrained_model.module.backbone.state_dict()
        else:
            pretrained_dict = self.pretrained_model.backbone.state_dict()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.backbone.load_state_dict(pretrained_dict, strict=False)
        else:
            self.model.backbone.load_state_dict(pretrained_dict, strict=False)

    def get_features_labels(self, dataloader, model, args, return_logit=False):
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0,self.num_labels)).to(self.device)

        for _, batch in enumerate(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                outputs = model(X, output_hidden_states=True)
                feature = outputs["hidden_states"]
                logit = outputs['logits']
                # feature = model(X, output_hidden_states=True)["hidden_states"]

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))
            total_logits = torch.cat((total_logits, logit))

        if return_logit:
            return total_features, total_labels, total_logits
        else:
            return total_features, total_labels
            
    def save_results(self, args, result_source=None):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)
        var = [args.evaluation_epoch, args.dataset, args.running_method, args.architecture, args.known_cls_ratio, args.label_setting, args.labeled_shot, args.labeled_ratio, result_source, args.seed, args.topk, args.view_strategy, args.num_train_epochs, args.ce_weight, args.cl_weight, args.sup_weight, args.weight_ce_unsup, args.options, args.query_samples, args.update_per_epoch,
               args.sampling_strategy, args.allocation_degree, 
               args.weight_cluster_instance_cl, args.options_cluster_instance_ratio,
               args.prompt_ablation, args.component_ablation, args.llm,
               args.feedback_cache, self.num_cached_feedback]
        names = ['evaluation_epoch', 'dataset', 'running_method', 'architecture', 'known_cls_ratio', 'label_setting', 'labeled_shot', 'labeled_ratio', 'result_source', 'seed', 'topk', 'view_strategy', 'num_train_epochs', 'ce_weight', 'cl_weight', 'sup_weight', 'weight_ce_unsup', 'options', 'query_samples', 'update_per_epoch',
                 'sampling_strategy', 'allocation_degree',
                 'weight_cluster_instance_cl', 'options_cluster_instance_ratio', 
                 'prompt_ablation', 'component_ablation', 'llm',
                 'feedback_cache', 'num_cached_feedback']
        vars_dict = {k:v for k,v in zip(names, var)}
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        file_name = f'results_{args.experiment_name}.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
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
        data_diagram = pd.read_csv(results_path)
        
        print('result_source:', result_source)
        print('test_results\n', data_diagram)
    
    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model
                
    def EuclideanDistances(self, a, b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

    def predict_k(self, args, data):
        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.pretrained_model.cuda(), args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels * 0.9
        print('drop',drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num',num_labels)

        return num_labels
    
if __name__ == '__main__':

    print('\nParameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    result_source = 'tba'
    # var = [args.dataset, args.running_method, args.architecture, args.known_cls_ratio, args.label_setting, args.labeled_shot, args.labeled_ratio, result_source, args.seed, args.topk, args.view_strategy, args.num_train_epochs, args.ce_weight, args.cl_weight, args.sup_weight, args.weight_ce_unsup, args.options, args.query_samples, args.update_per_epoch]
    # names = ['dataset', 'running_method', 'architecture', 'known_cls_ratio', 'label_setting', 'labeled_shot', 'labeled_ratio', 'result_source', 'seed', 'topk', 'view_strategy', 'num_train_epochs', 'ce_weight', 'cl_weight', 'sup_weight', 'weight_ce_unsup', 'options', 'query_samples', 'update_per_epoch']

    var = [args.dataset, args.running_method, args.architecture, args.known_cls_ratio, args.label_setting, args.labeled_shot, args.labeled_ratio, result_source, args.seed, args.topk, args.view_strategy, args.num_train_epochs, args.ce_weight, args.cl_weight, args.sup_weight, args.weight_ce_unsup, args.options, args.query_samples, args.update_per_epoch,
            args.sampling_strategy, args.allocation_degree,
            args.weight_cluster_instance_cl, args.options_cluster_instance_ratio]
    names = ['dataset', 'running_method', 'architecture', 'known_cls_ratio', 'label_setting', 'labeled_shot', 'labeled_ratio', 'result_source', 'seed', 'topk', 'view_strategy', 'num_train_epochs', 'ce_weight', 'cl_weight', 'sup_weight', 'weight_ce_unsup', 'options', 'query_samples', 'update_per_epoch',
                'sampling_strategy', 'allocation_degree',
                'weight_cluster_instance_cl', 'options_cluster_instance_ratio']

    print('\n### Key Hyperparameters and Values###')
    for i in range(len(names)):
        print(names[i], ':', var[i])

    print('\nData Initialization...')
    data = Data(args)

    # Pretraining
    if os.path.exists(args.pretrain_dir):
        args.disable_pretrain = True # disable internal pretrain
    else:
        args.disable_pretrain = False

    if not args.disable_pretrain:
        print('\n\nPre-training begin...')
        manager_p = PretrainModelManager(args, data)
        manager_p.train(args, data)
        print('Pre-training finished!')
        manager = ModelManager(args, data, manager_p.model)
    else:
        manager = ModelManager(args, data)
    
    if args.report_pretrain:
        method = args.method
        args.method = 'pretrain'
        manager.evaluation(args, data) # evaluate when report performance on pretrain
        args.method = method

    manager = ModelManager(args, data)

    print('\n\nTraining begin...')
    print('architecture: ', args.architecture)
    manager.train(args,data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, data)
    print('Evaluation finished!')
    if args.save_model:
        print('Saving Model ...')
        manager.model.save_backbone(args.save_model_path)
    print('\n Number of All LLM feedback: ',len(manager.di_all))
    print("Finished!")