

python E2E_train_AD_boundary.py --dataset_name wikipedia --model_name TCL --num_runs 5 --gpu 1 --val_ratio 0.2 --test_ratio 0.3

python sup_train_link_AD.py --dataset_name wikipedia --model_name TCL --num_runs 5 --gpu 7 --val_ratio 0.2 --test_ratio 0.3 --AD_layer MLP --loss_type Focal --train_rule DRW


python sup_train_link_AD.py --dataset_name reddit --model_name TCL --num_runs 5 --gpu 3 --val_ratio 0.1 --test_ratio 0.2 --AD_layer MLP


--loss_type Focal --train_rule DRW



