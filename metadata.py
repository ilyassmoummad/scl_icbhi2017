import pandas as pd
import numpy as np
import glob
import os
from args import args

files_wav = [file for file in glob.glob(os.path.join(args.datapath,'*.wav'))]
files_txt = [file for file in glob.glob(os.path.join(args.datapath,'[1-2]*.txt'))]
split_txt = pd.read_csv(os.path.join(args.datapath,'ICBHI_challenge_train_test.txt'), delimiter='\t', header=None, names=['patient','mode'])
split_txt['patient'] = split_txt['patient'].str[:-8]

dfs = []
for i in range(len(files_txt)):
    txtfilename = files_txt[i]
    filename = txtfilename[:-4]
    filenamewithoutpath = filename.split('/')[-1]
    audiofilename = filenamewithoutpath + '.wav' 
    df = pd.read_csv(txtfilename, header=None, names=['onset','offset','crackles','wheezes'],delimiter='\t')   
    df.insert(loc=0, column='filepath', value=audiofilename) 
    df.insert(loc=0, column='patient', value=filenamewithoutpath[:3])  
    filenamewithoutpathanddevice = filenamewithoutpath[:-8]
    mode = split_txt[split_txt['patient']==filenamewithoutpathanddevice]['mode'].values[0] 
    df.insert(loc=len(df.columns), column='split', value=mode)
    df.insert(loc=len(df.columns), column='device', value=filenamewithoutpath[-8:]) 
    dfs.append(df)

merged = pd.concat(dfs)
merged.to_csv(os.path.join(args.datapath, args.metadata))