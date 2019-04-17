import os, sys, json
from os.path import join
import numpy as np
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ConvNet
from dataset import KanjiDataset

class Wrapper(object):
    """docstring for Wrapper."""

    def __init__(self, config, cont=None):
        super(Wrapper, self).__init__()
        with open(config, 'r') as f:
            config = json.load(f)
        self.config = config
        self.model = ConvNet(config['model'])
        if cont is not None:
            print('loading in weights')
            self.load_model(cont)

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            print('using cuda')
            self.model.cuda()


    def train(self, cont=False):
        model = self.model
        config = self.config
        trainloader = DataLoader(
            KanjiDataset(self.config, train=True),
                batch_size=config['train']['batch_size'], pin_memory=True)
        self.valloader = DataLoader(
            KanjiDataset(self.config, train=False),
                batch_size=config['train']['batch_size'], pin_memory=True)
        objective = nn.CrossEntropyLoss()
        self.objective = objective
        optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

        bestloss = float('Inf')
        for e in range(config['train']['epochs']):
            avgloss = 0.0
            for i, (x, y) in enumerate(trainloader):
                if self.cuda:
                    x = x.cuda(async=True)
                    y = y.cuda(async=True)

                optimizer.zero_grad()
                preds = model(x)
                loss = objective(preds, y)
                avgloss += loss.item()
                loss.backward()
                optimizer.step()

                preds = None
                gc.collect()
            avgloss /= len(trainloader)
            vloss = self.valid()
            if e%5==0:
                print('epoch: {}, loss: {:.4f}, val_loss: {:.4f}'
                    .format(e+1, avgloss, vloss ) )
                # print('epoch: {}, loss: {:.4f}, val_loss: {:.4f}, memory: {:.4f}'
                #     .format(e+1, avgloss, vloss, torch.cuda.memory_allocated(0) / 1e9 ) )
            if e%20==0:
                acc, conf = self.eval()
                print('acc:', acc)
                print('conf:', conf)
            if vloss < bestloss:
                self.save_model('{:.4f}'.format(vloss))
                bestloss = vloss

        self.valloader = None
        acc, conf = self.eval()
        print('acc:', acc)
        print('conf:', conf)
        return

    def valid(self):
        loss = 0.0
        for (x, y) in self.valloader:
            if self.cuda:
                x = x.cuda(async=True)
                y = y.cuda(async=True)
            loss += self.objective(self.model(x), y).item()
        return loss/len(self.valloader)

    def eval(self):
        validset = KanjiDataset(self.config, train=False)
        acc = 0
        conf = np.zeros((self.config['model']['classes'],
            self.config['model']['classes']))
        for (x, y) in validset:
            pred = self.predict(x)
            acc += (pred == y)
            conf[pred, y] += 1
        return acc/len(validset), conf

    def predict(self, image):
        image = torch.unsqueeze(image, 0)
        if self.cuda:
            image = image.cuda(async=True)
        pred = self.model(image)
        pred = torch.argmax(pred[0])
        return pred.item()

    def save_model(self, itr):
        path = str(self.config['model']['model_save_path'] + self.config['name']
        + '_model_' + str(itr) + '.pt')
        torch.save( self.model.state_dict(), path )
        print('save:', path)

    def load_model(self, cont):
        if cont == 'cont':
            pass
        else:
            self.model.load_state_dict( torch.load(
                join(self.config['model']['model_save_path'], cont) ) )


if __name__ == '__main__':
    wrapper = Wrapper('configs/config.json')