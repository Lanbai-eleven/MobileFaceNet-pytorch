import time

import numpy as np
import torch
import torch.nn as nn
from termcolor import colored
from torch import optim
from torch.nn import DataParallel

from models.MobileFaceNet import MobileFaceNet
from models.Arcloss import ArcMarginProduct
from utils import get_datas, check_task_folder, save_model, log
from torch.optim import SGD, lr_scheduler


def cal_similarity(left_features, right_features):
    return np.dot(left_features, right_features) / (np.linalg.norm(left_features) * np.linalg.norm(right_features))


def get_features(model, dataloader, device):
    left_features_list = None
    right_features_list = None
    labels_list = None

    model.eval().to(device)

    for left_datas, flip_left_datas, right_datas, flip_right_datas, labels in dataloader:
        left_datas = left_datas.to(device)
        flip_left_datas = flip_left_datas.to(device)
        right_datas = right_datas.to(device)
        flip_right_datas = flip_right_datas.to(device)

        left_features = model(left_datas).cpu().detach().numpy()
        flip_left_features = model(flip_left_datas).cpu().detach().numpy()
        right_features = model(right_datas).cpu().detach().numpy()
        flip_right_features = model(flip_right_datas).cpu().detach().numpy()

        # print(left_features.shape, flip_left_features.shape, right_features.shape, flip_right_features.shape)
        left_features = np.hstack((left_features, flip_left_features))
        right_features = np.hstack((right_features, flip_right_features))

        # print(left_features.shape, right_features.shape)
        # print(labels.shape)

        left_features_list = left_features if left_features_list is None else np.vstack(
            (left_features_list, left_features))
        right_features_list = right_features if right_features_list is None else np.vstack(
            (right_features_list, right_features))
        labels_list = labels.cpu().numpy() if labels_list is None else np.hstack((labels_list, labels.cpu().numpy()))

        # print(left_features_list.shape, right_features_list.shape, labels_list.shape)

    return left_features_list, right_features_list, labels_list


def cal_acc(sim, labels):
    sim, labels = np.asarray(sim), np.asarray(labels)
    best_accuracy = 0
    best_threshold = 0
    for i in range(len(sim)):
        threshold = sim[i]
        accuracy = ((sim > threshold) == labels).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_accuracy, best_threshold


def eval_performance(model, dataloader, device):
    left_features_list, right_features_list, labels = get_features(model, dataloader, device)

    sim = [cal_similarity(left_features_list[i], right_features_list[i]) for i in range(len(labels))]
    sim = np.array(sim)

    accuracy, threshold = cal_acc(sim, labels)

    return accuracy, threshold


if __name__ == "__main__":
    # config
    batch_size = 512
    epochs = 60

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, valid_dataloader, num_classes = get_datas(batch_size=batch_size)
    print("num_classes:", num_classes)

    model = MobileFaceNet().to(device)
    arc_loss = ArcMarginProduct(128, num_classes + 1).to(device)

    prelu_params_list = []
    default_params_list = []

    for name, param in model.layers[:-2].named_parameters():
        if 'prelu' in name:
            prelu_params_list.append(param)
        else:
            default_params_list.append(param)

    optimizer = optim.SGD([
        {'params': prelu_params_list, 'weight_decay': 0},
        {'params': default_params_list, 'weight_decay': 4e-5},
        {'params': model.layers[-1].parameters(), 'weight_decay': 4e-4},
        {'params': model.layers[-2].parameters(), 'weight_decay': 4e-4},
        {'params': arc_loss.weight, 'weight_decay': 4e-4}],
        lr=0.1, momentum=0.9, nesterov=True)

    lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[36, 52, 58], gamma=0.1)

    # Wrap the model with DataParallel
    model = DataParallel(model)
    arc_loss = arc_loss.to(device)

    check_task_folder()

    max_val_acc = 0

    for epoch in range(epochs):
        print(colored("Epoch {}/{}".format(epoch, epochs - 1), "yellow"))
        log("Epoch {}/{}".format(epoch, epochs - 1))

        train_loss = 0.0
        train_correct = 0
        train_data_num = 0

        start_time = time.time()

        model.train()
        for i, batch_datas in enumerate(train_dataloader):
            inputs, labels = batch_datas
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            outputs = arc_loss(outputs, labels)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_data_num += batch_size
            _, preds = torch.max(outputs.data, 1)
            train_correct += preds.eq(labels).sum().item()

        print(colored("Train Loss: {:.4f} Acc: {:.4f} Time: {:.4f}".format(train_loss / (train_data_num / batch_size),
                                                                           train_correct / train_data_num,
                                                                           time.time() - start_time), "magenta"))
        print(colored("Iterations per epoch:{}".format(train_data_num / batch_size), "magenta"))
        log("Train Loss: {:.4f} Acc: {:.4f} Time: {:.4f}".format(train_loss, train_correct / train_data_num,
                                                                 time.time() - start_time))

        start_time = time.time()

        model.eval()
        cur_val_acc, cur_val_threshold = eval_performance(model.module, valid_dataloader, device)

        if cur_val_acc > max_val_acc:
            max_val_acc = cur_val_acc
            save_model(model.module, "saved_models/best_model.pth")

        print(colored("Valid Acc: {:.4f} Threshold: {:.4f} Time: {:.4f}".format(cur_val_acc, cur_val_threshold,
                                                                                time.time() - start_time), "magenta"))
        print(colored("-----------------------------------------", "yellow"))

        log("Valid Acc: {:.4f} Threshold: {:.4f} Time: {:.4f}".format(cur_val_acc, cur_val_threshold,
                                                                      time.time() - start_time))
        log("-----------------------------------------")

        lr_scheduler.step()