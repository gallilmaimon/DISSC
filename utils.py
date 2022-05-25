import shutil
import tensorflow as tf
import random, os
import numpy as np
import torch


def seed_everything(seed: int):
    if seed == -1:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # needed for gaussian blur operation deterministically
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def init_loggers(path: str):
    if os.path.exists(path + '/train'): shutil.rmtree(path + '/train')
    if os.path.exists(path + '/val'): shutil.rmtree(path + '/val')
    if os.path.exists(path + '/best_model.pth'): os.remove(path + '/best_model.pth')
    train_logger = tf.summary.create_file_writer(path + '/train')
    val_logger = tf.summary.create_file_writer(path + '/val')

    return train_logger, val_logger

def log_metrics(logger, value_dict: dict, epoch: int,  name: str = 'train'):
    out_str = ''
    with logger.as_default():
        for k, v in value_dict.items():
            tf.summary.scalar(k, v, step=epoch)
            out_str += f'{name}_{k}: {v:.5f}, '
    print(out_str)