#!/bin/bash

cp ~/Documents/tops/0*/* data/topbot/
cp ~/Documents/tops/1*/* data/topbot/
cp ~/Documents/tops/2*/* data/topbot/
cp ~/Documents/tops/groups/*/* data/topbot/

cp ~/Documents/tops/rls/* data/rightleft/

cp ~/Documents/tops/singls/* data/single/

cp ~/Documents/tops/outside_inside/* data/outin_side/

cp ~/Documents/tops/undecided/* data/undecided/

cp ~/Documents/tops/unreadables/* data/unreadable/

rm -r data/train/*
rm -r data/test/*
python setup_data.py
