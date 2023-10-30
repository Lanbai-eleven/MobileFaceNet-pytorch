import datetime
import logging
import os

import torch
from termcolor import colored

from datasets.init_webface import CASIAWebFaceDataset
from datasets.ini_webface_memory_friendly import CASIAWebFaceMemoryFriendlyDataset
from datasets.init_val_webface import VALCASIAWebFaceDataset
from datasets.init_lfw import LFWDataset


def get_datas(batch_size=32):
    train_dataset = CASIAWebFaceMemoryFriendlyDataset("train.txt")
    valid_dataset = LFWDataset("pairs.txt")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)

    return train_dataloader, valid_dataloader, train_dataset.classes_num()

    # optimizer


def check_task_folder():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    folder_name = "MobileFaceNet_" + timestamp
    folder_path = os.path.join("saved_models", folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    init_log(folder_path)

    return folder_path


def save_model(model, folder_path):
    torch.save(model.state_dict(), folder_path)
    print(colored("Model saved to %s" % folder_path, "red"))


def init_log(task_folder):
    log_path = os.path.join(task_folder, "log.txt")
    logging.basicConfig(filename=log_path, level=logging.INFO)


def log(msg):
    logging.info(msg)



