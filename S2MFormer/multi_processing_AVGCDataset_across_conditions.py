import sys
sys.path.append('/data/wjq/AAD/OpenAAD/')

from multiprocessing import Process
from model_SNN_DTU_KUL_loop_unify_framework import main

import tools.utils as util
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


if __name__ == "__main__":
    # torch.set_default_dtype(torch.double)
    process = []
    path = "/data/wjq/AAD/AVGCDataset_across_conditions"  # ./DTUDataset  ./KULDataset

    # 1s

    subjects = [4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # S1 to S16

    sub_names = ['S' + str(i) for i in subjects]

    """
    Dataset
    """
    dataset = "No_vanilla_128"

    seed = 200

    window_lengths = [0.1] # 这里应该就是传入的 0.1/ 1 / 2
    multiple = len(sub_names)

    branch = 2
    is_CSP = True
    is_DE = True
    use_image = True
    logger = None

    is_cross_trials = True
    if is_cross_trials:
        model_name = "cross_trials"
    else:
        model_name = "within_trials"

    # Multiprocessing
    for window_length in window_lengths:
        for sub in sub_names:
            print(sub)
            p = Process(target=main, args=(sub, path, dataset, is_cross_trials, model_name, window_length, seed,branch, is_CSP, is_DE, use_image, logger))
            p.start()
            process.append(p)
            util.monitor(process, multiple, 60)
