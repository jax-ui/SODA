import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.FreeDyG import FreeDyG
from utils.losses import get_logp_boundary, calculate_bg_spp_loss
from models.modules import MergeLayer, flow_model, subnet_fc, ParallelModel
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer, get_logp
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
torch.cuda.empty_cache()
if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    # get arguments
    args = get_link_prediction_args(is_evaluation=False)
    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, train_norm_data, train_abnorm_data, mask_norm, mask_abnorm, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)
    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_norm_data_loader = get_idx_data_loader(indices_list=list(range(len(train_norm_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], [], [], []
    task_loss = nn.BCELoss()

    for run in range(args.num_runs):
        set_random_seed(seed=run)
        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'
        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f'configuration is {args}')
        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif args.model_name == 'FreeDyG':
            dynamic_backbone = FreeDyG(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
                                         num_layers=args.num_layers, dropout=args.dropout, max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        AD_layer = MergeLayer(input_dim=2*node_raw_features.shape[1], hidden_dim=node_raw_features.shape[1], output_dim=1)

        Normalize_flow = flow_model(2*node_raw_features.shape[1])

        CN_model = nn.Sequential(dynamic_backbone, Normalize_flow)
        AD_model = nn.Sequential(dynamic_backbone, AD_layer)

        logger.info(f'AD_model -> {AD_model}')
        logger.info(f'AD_model name: {args.model_name}, #parameters: {get_parameter_sizes(AD_model) * 4} B, '
                    f'{get_parameter_sizes(AD_model) * 4 / 1024} KB, {get_parameter_sizes(AD_model) * 4 / 1024 / 1024} MB.')
        logger.info(f'CN_model -> {CN_model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(AD_model) * 4} B, '
                    f'{get_parameter_sizes(AD_model) * 4 / 1024} KB, {get_parameter_sizes(AD_model) * 4 / 1024 / 1024} MB.')

        AD_optimizer = create_optimizer(model=AD_model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        CN_optimizer = create_optimizer(model=CN_model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        AD_model = convert_to_gpu(AD_model, device=args.device)
        CN_model = convert_to_gpu(CN_model, device=args.device)

        save_AD_model_folder = f"./saved_AD_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        save_CN_model_folder = f"./saved_CN_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        
        shutil.rmtree(save_AD_model_folder, ignore_errors=True)
        os.makedirs(save_AD_model_folder, exist_ok=True)
        os.makedirs(save_CN_model_folder, exist_ok=True)

        # early_stopping = EarlyStopping(patience=args.patience, save_AD_model_folder=save_AD_model_folder,
        #                                save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)
        
        num_iters = 10
        CN_num_epochs = 5
        AD_num_epochs = 5
        best_CN_val = 1e10
        for iteration in range(num_iters)
            for CN_epoch in range(CN_num_epochs):
                CN_model.train()
                if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer','FreeDyG']:
                    # training, only use training graph
                    model[0].set_neighbor_sampler(train_neighbor_sampler)
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reinitialize memory of memory-based models at the start of each epoch
                    model[0].memory_bank.__init_memory_bank__()
                # store train losses and metrics
                train_losses = []
                all_predicts, all_labels = [], []
                train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)

                for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                    train_data_indices = train_data_indices.numpy()
                    batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                        train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices],train_data.labels[train_data_indices]
                    
                    if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors)
                    elif args.model_name in ['JODIE', ' DyRep', 'TGN']:
                        
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            edge_ids=batch_edge_ids,
                                                                            edges_are_positive=True,
                                                                            num_neighbors=args.num_neighbors)
                    elif args.model_name in ['GraphMixer']:
                        
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)
                    elif args.model_name in ['DyGFormer']:
                    
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times)
                    elif args.model_name in ['FreeDyG']:
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times
                                                                                )
                    else:
                        raise ValueError(f"Wrong value for model_name {args.model_name}!")

                    edge_embeddings = torch.cat([batch_src_node_embeddings, batch_dst_node_embeddings],dim = 1)
                    z, log_jac_det = model[1](edge_embeddings)
            
                    logps = get_logp(edge_embeddings.shape[1], z, log_jac_det) 
                    logps = logps / edge_embeddings.shape[1]
                    log_sigmoid = nn.LogSigmoid()

                    if batch_labels.sum() > 0:
                        if args.focal_weighting:
                            logps_detach = logps.detach()
                            normal_logps = logps_detach[batch_labels == 0]   
                            anomaly_logps = logps_detach[batch_labels == 1]
                            nor_weights = normal_fl_weighting(normal_logps)
                            ano_weights = abnormal_fl_weighting(anomaly_logps)
                            weights = nor_weights.new_zeros(logps_detach.shape)
                            weights[batch_labels == 0] = nor_weights
                            weights[batch_labels == 1] = ano_weights
                            ml_loss = -log_sigmoid(logps[batch_labels == 0]) * nor_weights 
                            ml_loss = torch.mean(ml_loss)
                        else:
                            ml_loss = -log_sigmoid(logps[batch_labels == 0])
                            ml_loss = torch.mean(ml_loss)
                        boundaries = get_logp_boundary(logps, batch_labels)
                        if args.focal_weighting:
                            loss_n_con, loss_a_con = calculate_bg_spp_loss(logps, batch_labels, boundaries, weights=weights)
                        else:
                            loss_n_con, loss_a_con = calculate_bg_spp_loss(logps, batch_labels, boundaries)

                        loss = 0.5*ml_loss + (loss_n_con + loss_a_con)
                    else:
                        if args.focal_weighting:
                            normal_weights = normal_fl_weighting(logps.detach())
                            ml_loss = -log_sigmoid(logps) * normal_weights
                            loss = ml_loss.mean()      
                        else:
                            loss = -log_sigmoid(logps).mean()
                    if math.isnan(loss.item()):
                        train_losses = train_losses
                    else:
                        train_losses.append(loss.item())
            
                    CN_optimizer.zero_grad()
                    loss.backward()
                    CN_optimizer.step()
                    train_idx_data_loader_tqdm.set_description(f'CN_Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                        model[0].memory_bank.detach_memory_bank()

                # if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                #     # backup memory bank after training so it can be used for new validation nodes
                #     train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                val_losses = evaluate_CN_model_link_prediction(model_name=args.model_name,
                                                                        model=model,
                                                                        neighbor_sampler=full_neighbor_sampler,
                                                                        evaluate_idx_data_loader=val_idx_data_loader,
                                                                        evaluate_data=val_data,
                                                                        num_neighbors=args.num_neighbors,
                                                                        time_gap=args.time_gap)

                if val_losses < best_CN_val:
                    logger.info(f"save model {os.path.join(save_CN_model_folder, f"{args.save_model_name}.pkl")}")
                    torch.save(CN_model.state_dict(), os.path.join(save_CN_model_folder, f"{args.save_model_name}.pkl"))
                    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        torch.save(CN_model[0].memory_bank.node_raw_messages, 
                        os.path.join(save_CN_model_folder, f"{args.save_model_name}_nonparametric_data.pkl"))
                                                    
           
            CN_model.load_state_dict(torch.load(os.path.join(save_CN_model_folder, f"{args.save_model_name}.pkl"), map_location=map_location))
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                CN_model[0].memory_bank.node_raw_messages = torch.load(os.path.join(save_CN_model_folder, f"{args.save_model_name}_nonparametric_data.pkl"), map_location=map_location)

            #generate embeddings anomaly sample
            log_likelihoods = []
            CN_model.eval()
            with torch.no_grad():
                for data in val_data_loader:
                    inputs = data['normal_samples'].to(device)
                    log_likelihoods.append(model.log_prob(inputs))
            all_log_likelihoods = torch.cat(log_likelihoods)
         
            threshold = torch.quantile(all_log_likelihoods, 0.05).item()
            
            synthetic_anomalies = generate_synthetic_anomalies(CN_model, num_samples=500)


            AD_model[0].load_state_dict(CN_model[0].state_dict())


            for AD_epoch in range(AD_num_epochs):

                AD_model.train()
                if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer','FreeDyG']:
                    # training, only use training graph
                    model[0].set_neighbor_sampler(train_neighbor_sampler)
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reinitialize memory of memory-based models at the start of each epoch
                    model[0].memory_bank.__init_memory_bank__()

                # store train losses and metrics
                train_losses = []
                all_predicts, all_labels = [], []
        
                train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)

                for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                    train_data_indices = train_data_indices.numpy()
                    batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                        train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices],train_data.labels[train_data_indices]
                    if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                           num_neighbors=args.num_neighbors)
                    elif args.model_name in ['JODIE', ' DyRep', 'TGN']:
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            edge_ids=batch_edge_ids,
                                                                            edges_are_positive=True,
                                                                            num_neighbors=args.num_neighbors)
                    elif args.model_name in ['GraphMixer']:
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)
                    elif args.model_name in ['DyGFormer']:
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times)
                    elif args.model_name in ['FreeDyG']:
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times)
                    else:
                        raise ValueError(f"Wrong value for model_name {args.model_name}!")

                    edge_embeddings = torch.cat([batch_src_node_embeddings, batch_dst_node_embeddings],dim = 1)
                    predicts = model[1](edge_embeddings)
                    labels = torch.from_numpy(batch_labels).float().to(predicts.device)
                    loss = task_loss(input=predicts, target=labels)
                    train_losses.append(loss.item())
                    all_predicts.append(predicts)
                    all_labels.append(labels)
                    AD_optimizer.zero_grad()
                    loss.backward()
                    AD_optimizer.step()

                    train_idx_data_loader_tqdm.set_description(f'AD_Epoch: {AD_epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
                    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                        model[0].memory_bank.detach_memory_bank()

                all_predicts = torch.cat(all_predicts)
                all_labels = torch.cat(all_labels)
                train_metrics = []
                train_metrics.append(get_link_prediction_metrics(predicts=all_predicts, labels=all_labels))

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # backup memory bank after training so it can be used for new validation nodes
                    train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                        model=AD_model,
                                                                        neighbor_sampler=full_neighbor_sampler,
                                                                        evaluate_idx_data_loader=val_idx_data_loader,
                                                                        evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                        evaluate_data=val_data,
                                                                        loss_func=task_loss,
                                                                        num_neighbors=args.num_neighbors,
                                                                        time_gap=args.time_gap)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                    val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
                    # reload validation memory bank for testing nodes or saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)
                
                logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
                for metric_name in train_metrics[0].keys():
                    logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
                logger.info(f'validate loss: {np.mean(val_losses):.4f}')
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f'validate {metric_name}, {metric_value:.4f}')
                
                # perform testing once after test_interval_epochs
                if (epoch + 1) % args.test_interval_epochs == 0:
                    test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                            model=AD_model,
                                                                            neighbor_sampler=full_neighbor_sampler,
                                                                            evaluate_idx_data_loader=test_idx_data_loader,
                                                                            evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                            evaluate_data=test_data,
                                                                            loss_func=task_loss,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)

                    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # reload validation memory bank for new testing nodes
                        model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                    logger.info(f'test loss: {np.mean(test_losses):.4f}')
                    for metric_name, metric_value in test_metrics.items():
                        logger.info(f'test {metric_name}, {metric_value:.4f}')
                # select the best model based on all the validate metrics
                val_metric_indicator = []
                for metric_name, metric_value in val_metrics.items():
                    val_metric_indicator.append((metric_name, metric_value, True))
                early_stop = early_stopping.step(val_metric_indicator, model)
                if early_stop:
                    break               
               # load the best model
        early_stopping.load_checkpoint(AD_model)
        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=AD_model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=task_loss,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)
        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
            val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=AD_model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=task_loss,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap)

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name,average_val_metric in val_metrics.items():
                
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name,average_test_metric in test_metrics.items():
           
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)  
        test_metric_all_runs.append(test_metric_dict)
        
        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            }   
        result_json = json.dumps(result_json, indent=4)
        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}/"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

       
    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    
    sys.exit()


                        
                    
                    
                