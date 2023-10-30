import torch.utils.data
from datasets.init_webface import CASIAWebFaceDataset
from datasets.init_lfw import LFWDataset


def test_load_webface():
    train_dataset = CASIAWebFaceDataset("train_all.txt")
    valid_dataset = LFWDataset("pairs.txt")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)

    for i, (data, label) in enumerate(train_dataloader):
        print(i, data.shape, label.shape)
        break


if __name__ == "__main__":
    test_load_webface()
