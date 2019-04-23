#!/bin/bash

rm -r data/chars/*

# cp -r ~/Documents/tops/0*/ data/chars/
# cp -r ~/Documents/tops/1*/ data/chars/
# cp -r ~/Documents/tops/2*/ data/chars/
mkdir data/chars/nums/
cp -r ~/Documents/tops/0*/* data/chars/nums/
cp -r ~/Documents/tops/1*/* data/chars/nums/
cp -r ~/Documents/tops/2*/* data/chars/nums/
cp -r ~/Documents/tops/groups/*/ data/chars/

cp -r ~/Documents/tops/rls/ data/chars/

cp -r ~/Documents/tops/singls/ data/chars/

cp -r ~/Documents/tops/outside_inside/ data/chars/

cp -r ~/Documents/tops/undecided/ data/chars/

cp -r ~/Documents/tops/unreadables/ data/chars/

rm -r data/char_train/*
rm -r data/char_test/*
python setup_data.py true
