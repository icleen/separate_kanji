import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

from mwrapper import Wrapper
from dataset import KanjiDataset

def main():
    tolabel = sys.argv[1] if len(sys.argv) > 1 else './data/test/'
    config = sys.argv[2] if len(sys.argv) > 2 else 'configs/config.json'
    cont = sys.argv[3] if len(sys.argv) > 3 else 'cont'
    wrapper = Wrapper(config, cont=cont)
    dataset = KanjiDataset(wrapper.config, datapath=tolabel)
    output = []
    acc = 0
    conf = np.zeros((wrapper.config['model']['classes'],
        wrapper.config['model']['classes']), dtype=np.int32)
    for i, (x, y) in enumerate(dataset):
        pred = wrapper.predict(x)
        file = dataset.get_file(i)
        output.append( (pred, file) )
        # image = x.numpy()
        # print(image.shape)
        # image = np.transpose(image, (1, 2, 0))
        # print(image.shape)
        # print('pred: {}, label: {}, file: {}'.format(pred, y, file))
        # plt.imshow(image)
        # plt.show()
        acc += (pred == y)
        conf[y, pred] += 1
    acc = acc/len(dataset)
    print('acc:', acc)
    print('conf:\n', conf)

    with open('labels.txt', 'w') as f:
        for (pred, file) in output:
            f.write(str(pred) + ',' + str(file) + '\n')

if __name__ == '__main__':
    main()
