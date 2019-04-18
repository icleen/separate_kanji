import os, sys, json
from os.path import join

import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, config):
        super(ConvNet, self).__init__()

        in_channels = config['channels']
        img_size = config['img_size']

        last_layer = in_channels
        layer_size = config['layer_size']
        layers = []
        for l in range(config['layers']):
            layers += [
                nn.Conv2d(last_layer, layer_size, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(layer_size, layer_size, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ]
            if l < config['layers']-1:
                layers.append(nn.MaxPool2d(kernel_size=2))
            last_layer = layer_size
            layer_size *= 2
        self.layers = nn.Sequential(*layers)

        num_pools = config['layers']-1

        outshape = int(img_size / pow(2, num_pools))
        self.linear_size = outshape*outshape*last_layer
        self.fc = nn.Linear(self.linear_size, config['classes'])


    def forward(self, x):
        """x shape: (batch, channels, height, width)"""
        out = self.layers(x)
        out = out.view(-1, self.linear_size)
        # does the same thing, flattening out starting with the first dimension:
        # out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    config = sys.argv[1] if len(sys.argv) > 1 else 'configs/config.json'
    with open(config, 'r') as f:
        config = json.load(f)

    model = ConvNet(config['model'])
    print(model)
    print('\ntesting on a data example:')
    from dataset import KanjiDataset
    dataset = KanjiDataset(config)
    x, y = dataset[0]
    x = torch.unsqueeze(x, 0)
    print('input shape:', x.size())
    print('class:', y)
    preds = model(x)
    print('output shape:', preds.size())
