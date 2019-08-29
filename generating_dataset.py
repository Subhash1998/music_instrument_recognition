#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:50:25 2019

@author: subhash
"""

###################################import libraries################################
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
################################import from libraries##############################
from os import walk
from sklearn.preprocessing import LabelEncoder
from mfcc import find_mfcc

classes = ['cello','clarinet','flute','guitar','saxophone','trumpet','violin']
path = '/home/subhash/Desktop/musical instrument recognition/data/'
    
files=[]
labels=[]
prev=0
for classname in classes:
    for root,dirnames,filenames in walk(path+classname):
        for filename in fnmatch.filter(filenames,'*.wav'):
            files.append(filename)
            labels.append(classname)

i = 0
mfcc_data = []
for file in files:
    location = path+labels[i]+'/'+file
    mfcc_data.append(find_mfcc(location,labels[i]))
    i+=1

df = pd.DataFrame(mfcc_data)
csv_data = df.to_csv('/home/subhash/Desktop/musical instrument recognition/dataset.csv')