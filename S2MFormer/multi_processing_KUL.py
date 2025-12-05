import sys
sys.path.append('/data/wjq/AAD/OpenAAD/')
import torch
from multiprocessing import Process
from model_SNN_DTU_KUL_loop_unify_framework import main
import tools.utils as util
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


if __name__ == "__main__":
    # torch.set_default_dtype(torch.float64)
    process = []
    path = "/data/wjq/AAD/KULDataset"  # ./DTUDataset  ./KULDataset

    # sub_names = ['S' + str(i+1) for i in range(0, 8)]
    sub_names = ['S' + str(i+1) for i in range(8, 16)]
    # sub_names = ['S' + str(i+1) for i in range(0, 16)]
    # sub_names = ['S' + str(i+1) for i in range(0, 16)]
    """
    Dataset
    """
    dataset = "No_vanilla_250204_Cz_46080"

    seed = 200

    window_lengths = [0.1] # 0.1/ 1 / 2  eg: 16-32： 2s占用10580，1s占用6016
    multiple = len(sub_names)

    branch = 2
    is_CSP = True
    is_DE = True
    use_image = True
    logger = None

    is_cross_trials = True
    if is_cross_trials:
        model_name = "S2MFormer_cross_trials"
    else:
        model_name = "S2MFormer_within_trials"

    for window_length in window_lengths:
        for sub in sub_names:
            print(sub)
            p = Process(target=main, args=(sub, path, dataset, is_cross_trials, model_name, window_length, seed,branch, is_CSP, is_DE, use_image, logger))
            p.start()
            process.append(p)
            util.monitor(process, multiple, 60)

