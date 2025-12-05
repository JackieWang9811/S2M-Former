import sys
sys.path.append('/data/wjq/AAD/OpenAAD/')

from multiprocessing import Process
from model_SNN_DTU_KUL_loop_unify_framework_loso import main

import tools.utils as util
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


if __name__ == "__main__":
    # torch.set_default_dtype(torch.double)
    process = []
    # path = "data/wjq/AAD/KULDataset"  # ./DTUDataset  ./KULDataset
    path = "/data/wjq/AAD/DTUDataset" # ./DTUDataset  ./KULDataset

    # 0.1s
    sub_names = ['S' + str(i+1) for i in range(0, 1)]
    multiple = len(sub_names)
    """
    Dataset
    """
    dataset = "No_vanilla_128" #  No_vanilla 是 64h
    seed = 200

    window_lengths = [1] # 0.1/ 1 / 2  eg: 16-32： 2s占用10580，1s占用6016
    batch_size = 128

    branch = 2
    is_CSP = True
    is_DE = True
    use_image = True
    logger = None

    split = "loso"

    model_name = "S2MFormer_cross_subjects_loso"

    for window_length in window_lengths:
        for sub in sub_names:
            print(sub)
            p = Process(target=main, args=(sub,  path, dataset, batch_size, model_name, window_length, seed,branch, is_CSP, is_DE, split, logger))
            p.start()
            process.append(p)
            util.monitor(process, multiple, 60)

