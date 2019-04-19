import os, sys, json
from os.path import join
from shutil import copyfile

with open('labels.txt', 'r') as f:
    files = [fil.strip() for fil in f]

path = 'data/labeled/'
dict = {'0':'rls', '1':'single', '2':'topbot', '3':'unreadable'}
for fil in files:
    label, file = fil.split(',')
    label = dict[label]
    end = file.split('/')[-1]
    copyfile(file, join(path, str(label), end))
