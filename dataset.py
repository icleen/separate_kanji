import os, sys, json
from os.path import join

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms


class KanjiDataset(Dataset):
    def __init__(self, config, train=True):
        super(KanjiDataset, self).__init__()
        datap = config['data']
        datap = join(datap, 'train') if train else join(datap, 'test')
        self.data = torchvision.datasets.ImageFolder(
            datap, transform=transforms.Compose(
                [transforms.Resize(config['model']['img_size']), transforms.ToTensor()] ) )

    def __getitem__(self,index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    config = 'configs/config.json'
    with open(config, 'r') as f:
        config = json.load(f)
    dataset = KanjiDataset(config)
    print(len(dataset))
    x, y = dataset[0]
    print(x.size())
    print(y)
    x, y = dataset[-1]
    print(x.size())
    print(y)
