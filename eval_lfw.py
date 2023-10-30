import csv

import numpy as np
import torch

from datasets.init_eval_lfw import LFWEvalDataset
from datasets.init_lfw import LFWDataset
from models.MobileFaceNet import MobileFaceNet
from models.OtherMobileFaceNet import OtherMobileFacenet
from train import cal_similarity, eval_performance


def eval_lfw(model, dataloader, device, threshold):
    model.eval().to(device)
    left_features_list = None
    right_features_list = None
    predicts = []

    for left_datas, flip_left_datas, right_datas, flip_right_datas in dataloader:
        left_datas = left_datas.to(device)
        flip_left_datas = flip_left_datas.to(device)
        right_datas = right_datas.to(device)
        flip_right_datas = flip_right_datas.to(device)

        left_features = model(left_datas).cpu().detach().numpy()
        flip_left_features = model(flip_left_datas).cpu().detach().numpy()
        right_features = model(right_datas).cpu().detach().numpy()
        flip_right_features = model(flip_right_datas).cpu().detach().numpy()

        left_features = np.hstack((left_features, flip_left_features))
        right_features = np.hstack((right_features, flip_right_features))

        left_features_list = left_features if left_features_list is None else np.vstack(
            (left_features_list, left_features))
        right_features_list = right_features if right_features_list is None else np.vstack(
            (right_features_list, right_features))

    for i in range(len(left_features_list)):
        left_feature = left_features_list[i]
        right_feature = right_features_list[i]
        sim = cal_similarity(left_feature, right_feature)
        predicts.append(1 if sim >= threshold else -1)

    print("nums of 1: %d, nums of 0: %d" % (predicts.count(1), predicts.count(0)))

    csv_file = "datas/lfwdata/test_lst.csv"
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == len(predicts):
        for i in range(len(rows)):
            rows[i].append(predicts[i])

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MobileFaceNet().to(device)
    model.load_state_dict(torch.load("saved_models/best_model.pth"))

    test_dataset = LFWDataset("pairs.txt")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    cur_val_acc, cur_val_threshold = eval_performance(model, test_dataloader, device)
    print("cur_val_acc: %f, cur_val_threshold: %f" % (cur_val_acc, cur_val_threshold))

    eval_dataset = LFWEvalDataset("test_lst.csv")
    eval_dataloader = torch.utils.data.dataloader.DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=2)

    eval_lfw(model, eval_dataloader, device, cur_val_threshold)
