import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VALCASIAWebFaceDataset(Dataset):
    def __init__(self, file_name):
        self.root = "./datas/CASIA-WebFace"
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.file = os.path.join(self.root, file_name)
        self.datas = []
        self.label = []

        with open(self.file, "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                left_line = lines[i].strip().split(" ")
                right_line = lines[i + 1].strip().split(" ")
                left_image_path, left_label = left_line
                right_image_path, right_label = right_line
                print(f"left: {left_image_path}, {left_label}, right: {right_image_path}, {right_label}")
                left_image_path = os.path.join(self.root, left_image_path)
                right_image_path = os.path.join(self.root, right_image_path)
                self.datas.append((left_image_path, right_image_path))
                self.label.append(1)

            # produce negative pairs
            for i in range(0, len(self.datas)):
                right_index = random.randint(0, len(self.datas) - 1)
                while right_index == i:
                    right_index = random.randint(0, len(self.datas) - 1)

                left_line = lines[i].strip().split(" ")
                right_line = lines[right_index].strip().split(" ")
                left_image_path, left_label = left_line
                right_image_path, right_label = right_line
                print(f"left: {left_image_path}, {left_label}, right: {right_image_path}, {right_label}")
                left_image_path = os.path.join(self.root, left_image_path)
                right_image_path = os.path.join(self.root, right_image_path)
                self.datas.append((left_image_path, right_image_path))
                self.label.append(0)

        print("VALCASIAWebFaceDataset init done, {} samples".format(len(self.datas)))

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

        return left_image, flipped_left_image, right_image, flipped_right_image, self.label[index]

    def __len__(self):
        return len(self.datas)

