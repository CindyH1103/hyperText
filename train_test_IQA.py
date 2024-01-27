import os
import argparse
import random
import numpy as np
from HyerIQASolver import HyperIQASolver
from hypertext import HyperIQASolver_Text
from hyperall import HyperIQASolver_All, HyperIQASolver_All_2023


os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main(config):
    folder_path = {
        'live': '/home/ssl/Database/databaserelease2/',
        'csiq': '/home/ssl/Database/CSIQ/',
        'tid2013': '/home/ssl/Database/TID2013/',
        'livec': '/home/ssl/Database/ChallengeDB_release/ChallengeDB_release/',
        'koniq-10k': '/home/ssl/Database/koniq-10k/',
        'bid': '/home/ssl/Database/BID/',
        'AGIQA-3k': '/home/huangyixin/hw1/homework1/AGIQA-3K/',
        'AIGCIQA2023': '/home/huangyixin/hw1/homework1/AIGCIQA2023/'
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
        'AGIQA-3k': list(range(0, 2982)),
        'AIGCIQA2023': list(range(0, 2400)),
    }
    sel_num = img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=np.float64)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float64)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i+1))
        # Randomly select 80% images for training and the rest for testing
        random.shuffle(sel_num)
        if config.hyper_text:
            if config.dataset == "AIGCIQA2023":
                sel_num = list(range(0, 24))
                random.shuffle(sel_num)
                train_index = sel_num[0: 22]
                test_index = sel_num[22:]
            else:
                train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
                test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
            solver = HyperIQASolver_Text(config, folder_path[config.dataset], train_index, test_index)
        elif config.hyper_all:
            train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
            test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
            if config.dataset == "AIGCIQA2023":
                solver = HyperIQASolver_All_2023(config, folder_path[config.dataset], train_index, test_index)
            else:
                solver = HyperIQASolver_All(config, folder_path[config.dataset], train_index, test_index)
        else:
            train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
            test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
            solver = HyperIQASolver(config, folder_path[config.dataset], train_index, test_index)
        solver.train()

    # print(srcc_all)
    # print(plcc_all)
    # srcc_med = np.median(srcc_all)
    # plcc_med = np.median(plcc_all)
    #
    # print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))

    # return srcc_med, plcc_med


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='AGIQA-3k', help='Support datasets: AGIQA-3k|AIGCIQA2023')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=1, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=4e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=30, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches, original 224')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')
    parser.add_argument('--hyper text', dest='hyper_text', type=bool, default=False, help='whether to use hyper Text solver')
    parser.add_argument('--hyper all', dest='hyper_all', type=bool, default=False, help='whether to use hyper all network')

    config = parser.parse_args()
    main(config)

