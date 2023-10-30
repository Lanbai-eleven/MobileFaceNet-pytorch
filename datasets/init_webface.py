import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CASIAWebFaceDataset(Dataset):
    def __init__(self, file_name):
        self.root = "./datas/CASIA-WebFace"
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0])
        ])

        self.file = os.path.join(self.root, file_name)
        self.datas = []
        self.labels = []

        with open(self.file, "r") as f:
            for line in f.readlines():
                line = line.strip().split(" ")
                image_path, label = line
                print(image_path, label)
                image_path = os.path.join(self.root, image_path)

                # read image and add to datas, labels
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)
                self.datas.append(image)
                self.labels.append(int(label))

        self.datas = torch.stack(self.datas)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)

    def classes_num(self):
        return torch.unique(self.labels).shape[0]
