
import os, sys
from os.path import join
from random import shuffle
from shutil import copyfile

def setup_data():
    datapath = './data/'
    folders = ['rightleft', 'single', 'topbot', 'unreadable', 'undecided', 'outin_side']
    datal = []
    for fol in folders:
        cls = fol
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

def setup_char_data():
    datapath = './data/chars/'
    folders = [f for f in os.listdir(datapath) if f != '.' and f != '/']
    datal = []
    clsses = []
    for fol in folders:
        cls = fol
        if cls not in clsses:
            clsses.append(cls)
        fol = join(datapath, fol)
        for fil in os.listdir(fol):
            if fil != '.' and fil != '/':
                fil = join(fol, fil)
                datal.append( (cls, fil) )

    shuffle(datal)
    train = int(0.8 * len(datal))
    datapath = './data/'

    dpath = join(datapath, 'char_train')
    for cls in clsses:
        path = join(dpath, cls)
        if not os.path.exists(path):
            os.mkdir(path)
    for inst in datal[:train]:
        path = join(dpath, str(inst[0]), inst[1].split('/')[-1])
        copyfile(inst[1], path)

    dpath = join(datapath, 'char_test')
    for cls in clsses:
        path = join(dpath, cls)
        if not os.path.exists(path):
            os.mkdir(path)
    for inst in datal[train:]:
        path = join(dpath, str(inst[0]), inst[1].split('/')[-1])
        copyfile(inst[1], path)


def main():
    doChars = sys.argv[1] if len(sys.argv) > 1 else False
    if doChars is not False:
        setup_char_data()
    else:
        setup_data()

if __name__ == '__main__':
    main()
