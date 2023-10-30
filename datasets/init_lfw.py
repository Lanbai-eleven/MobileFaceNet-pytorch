import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LFWDataset(Dataset):
    def __init__(self, file_name="pairs.txt"):
        self.root = "./datas/lfw"
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0])
        ])

        self.file = os.path.join(self.root, file_name)
        self.datas = []
        self.labels = []

        with open(self.file, "r") as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                # print(line)
                if len(line) == 3:
                    left_image_name, left_image_index, right_image_index = line
                    left_path = os.path.join(self.root, left_image_name, left_image_name + "_" +
                                             left_image_index.zfill(4) + ".jpg")
                    right_path = os.path.join(self.root, left_image_name, left_image_name + "_" +
                                              right_image_index.zfill(4) + ".jpg")

                    self.datas.append((left_path, right_path))
                    self.labels.append(1)
                elif len(line) == 4:
                    left_image_name, left_image_index, right_image_name, right_image_index = line
                    left_path = os.path.join(self.root, left_image_name, left_image_name + "_" +
                                             left_image_index.zfill(4) + ".jpg")
                    right_path = os.path.join(self.root, right_image_name, right_image_name + "_" +
                                              right_image_index.zfill(4) + ".jpg")

                    self.datas.append((left_path, right_path))
                    self.labels.append(0)

        print("LFWDataset init done")

    def __getitem__(self, index):
        left_image_path, right_image_path = self.datas[index]
        left_image = Image.open(left_image_path).convert("RGB")
        right_image = Image.open(right_image_path).convert("RGB")

        left_image = self.transform(left_image)
        right_image = self.transform(right_image)

        left_image = (left_image-127.5)/128.0
        right_image = (right_image-127.5)/128.0

        flipped_left_image = torch.flip(left_image, dims=[2])
        flipped_right_image = torch.flip(right_image, dims=[2])

        return left_image, flipped_left_image, right_image, flipped_right_image, self.labels[index]

    def __len__(self):
        return len(self.datas)

    def classes_num(self):
        return torch.unique(self.labels).shape[0]
