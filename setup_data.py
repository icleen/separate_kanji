
import os
from os.path import join
from random import shuffle
from shutil import copyfile

def main():
    datapath = './data/'
    folders = ['rightleft', 'single', 'topbot', 'unreadable', 'undecided', 'outin_side']
    datal = []
    for cls, fol in enumerate(folders):
        fol = join(datapath, fol)
        for fil in os.listdir(fol):
            if fil != '.' and fil != '/':
                fil = join(fol, fil)
                datal.append( (cls, fil) )

    shuffle(datal)
    train = int(0.8 * len(datal))
    dpath = join(datapath, 'train')
    for inst in datal[:train]:
        path = join(dpath, str(inst[0]))
        if not os.path.exists(path):
            os.mkdir(path)
        copyfile(inst[1], join(path, inst[1].split('/')[-1]))

    dpath = join(datapath, 'test')
    for inst in datal[train:]:
        path = join(dpath, str(inst[0]))
        if not os.path.exists(path):
            os.mkdir(path)
        copyfile(inst[1], join(path, inst[1].split('/')[-1]))

    # with open(join(datapath, 'train.txt'), 'w') as f:
    #     for inst in datal[:train]:
    #         f.write(str(inst[0]) + ',' + str(inst[1]) + '\n')
    #
    # with open(join(datapath, 'valid.txt'), 'w') as f:
    #     for inst in datal[train:]:
    #         f.write(str(inst[0]) + ',' + str(inst[1]) + '\n')

if __name__ == '__main__':
    main()
