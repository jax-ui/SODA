import os
import itertools

datasets = ["wikipedia", "reddit"]
# datasets = ["reddit"]
models = ["JODIE"]
# models = ["DyRep"]
# models = ["TGN"]
# models = ["TGAT"]
# models  = ["GraphMixer"]
# models = ["TCL"]
# models  = ["DyGFormer"]
# models = ["CAWN"]
num_runs = 5
gpu = 1

for dataset, model in itertools.product(datasets, models):

    if dataset == 'wikipedia':
        val_ratio = 0.2
        test_ratio = 0.3
    else:
        val_ratio = 0.1
        test_ratio = 0.2
   
    command = (
        f"nohup python sup_train_link_AD.py --dataset_name {dataset} --model_name {model} "
        f"--num_runs {num_runs} --gpu {gpu} --val_ratio {val_ratio} --test_ratio {test_ratio} --loss_type Focal >/dev/null 2>&1 &"
    )
    
    print(f"Running command: {command}")

    exit_code = os.system(command)
    

    if exit_code != 0:
        print(f"Command failed with exit code {exit_code}. Stopping further execution.")
        break




