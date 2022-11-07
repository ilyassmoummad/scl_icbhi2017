import argparse

parser = argparse.ArgumentParser()

#Generic
parser.add_argument("--method", type=str, default='sl') #method in ['sl','scl','hybrid']
parser.add_argument("--device", type=str, default="cuda:0") #device to train on
parser.add_argument("--workers", type=int, default=3) #number of workers
parser.add_argument("--bs", type=int, default=128) #batch size
parser.add_argument("--epochs", type=int, default=400) #nb of epoches
parser.add_argument("--epochs2", type=int, default=50) #nb of epoches for linear classifier of scl and hybrid

#Model
parser.add_argument("--backbone", type=str, default='cnn6') #['cnn6','cnn10','cnn14]
parser.add_argument("--dropout", action='store_true') #whether to activate
parser.add_argument("--scratch", action='store_true') #train from scratch
parser.add_argument("--weightspath", type=str, default='panns') #path to cnn6, cnn10 and cnn14 weights

#Data
parser.add_argument("--datapath", type=str, default='data') # path of ICBHI dataset
parser.add_argument("--metadata", type=str, default='metadata.csv') #metadata file
parser.add_argument("--samplerate", type=int, default=16000) #sampling rate
parser.add_argument("--duration", type=int, default=8) #max duration of audio for train/test
parser.add_argument("--pad", type=str, default='circular') #audio padding in ['zero','circular']
parser.add_argument("--noweights", action='store_true') #remove class weights for cross entropy

#Optimizer
parser.add_argument("--wd", type=float, default=1e-4) #weight decay
parser.add_argument("--lr", type=float, default=1e-4) #learning rate
parser.add_argument("--lr2", type=float, default=1e-1) #learning rate for linear eval

#Data Augmentation
parser.add_argument("--freqmask", type=int, default=20) #frequency mask size for spectrogram
parser.add_argument("--freqstripes", type=int, default=2) #frequency stripes for spectrogram
parser.add_argument("--timemask", type=int, default=40) #time mask size for spectrogram
parser.add_argument("--timestripes", type=int, default=2) #time stripes for spectrogram

#Parameter
parser.add_argument("--tau", type=float, default=0.06) #temperature for nt xent loss
parser.add_argument("--alpha", type=float, default=0.5) #tradeoff between cross entropy and nt xent

args = parser.parse_args()