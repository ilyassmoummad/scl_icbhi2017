import os 
import torch
import torch.nn as nn
from torchaudio import transforms as T
import torch.nn.functional as F
from torchinfo import summary
from augmentations import SpecAugment
from models import CNN6, CNN10, CNN14, Projector, LinearClassifier
from dataset import ICBHI
from utils import Normalize, Standardize
from losses import SupConLoss, SupConCELoss
from ce import train_ce
from hybrid import train_supconce
from scl import train_scl, linear_scl
from args import args

if not(os.path.isfile(os.path.join(args.datapath, args.metadata))):
    raise(IOError(f"CSV file {args.metadata} does not exist in {args.datapath}"))

METHOD = args.method

DEFAULT_NUM_CLASSES = 4 #for cross entropy
DEFAULT_OUT_DIM = 128 #for ssl embedding space dimension
DEFAULT_NFFT = 1024
DEFAULT_NMELS = 64
DEFAULT_WIN_LENGTH = 1024
DEFAULT_HOP_LENGTH = 512
DEFAULT_FMIN = 50
DEFAULT_FMAX = 2000

# Model definition
if args.method == 'sl':
    embed_only = False
else:
    embed_only = True
    projector = Projector(name=args.backbone, out_dim=DEFAULT_OUT_DIM, device=args.device)
    classifier = LinearClassifier(name=args.backbone, num_classes=DEFAULT_NUM_CLASSES, device=args.device)
    
if args.backbone == 'cnn6':
    PATH_TO_WEIGHTS = os.path.join(args.weightspath, 'Cnn6_mAP=0.343.pth')
    model = CNN6(num_classes=DEFAULT_NUM_CLASSES, do_dropout=args.dropout, embed_only=embed_only, from_scratch=args.scratch, path_to_weights=PATH_TO_WEIGHTS, device=args.device)
elif args.backbone == 'cnn10':
    PATH_TO_WEIGHTS = os.path.join(args.weightspath, 'Cnn10_mAP=0.380.pth')
    model = CNN10(num_classes=DEFAULT_NUM_CLASSES, do_dropout=args.dropout, embed_only=embed_only, from_scratch=args.scratch, path_to_weights=PATH_TO_WEIGHTS, device=args.device)
elif args.backbone == 'cnn14':
    PATH_TO_WEIGHTS = os.path.join(args.weightspath, 'Cnn14_mAP=0.431.pth')
    model = CNN14(num_classes=DEFAULT_NUM_CLASSES, do_dropout=args.dropout, embed_only=embed_only, from_scratch=args.scratch, path_to_weights=PATH_TO_WEIGHTS, device=args.device)
s = summary(model, device=args.device)
nparams = s.trainable_params

# Spectrogram definition
melspec = T.MelSpectrogram(n_fft=DEFAULT_NFFT, n_mels=DEFAULT_NMELS, win_length=DEFAULT_WIN_LENGTH, hop_length=DEFAULT_HOP_LENGTH, f_min=DEFAULT_FMIN, f_max=DEFAULT_FMAX).to(args.device)
normalize = Normalize()
melspec = torch.nn.Sequential(melspec, normalize)
standardize = Standardize(device=args.device)

# Data transformations
specaug = SpecAugment(freq_mask=args.freqmask, time_mask=args.timemask, freq_stripes=args.freqstripes, time_stripes=args.timestripes).to(args.device)
train_transform = nn.Sequential(melspec, specaug, standardize)
val_transform = nn.Sequential(melspec, standardize)

# Dataset and dataloaders
train_ds = ICBHI(data_path=args.datapath, metadatafile=args.metadata, duration=args.duration, split='train', device=args.device, samplerate=args.samplerate, pad_type=args.pad)
val_ds = ICBHI(data_path=args.datapath, metadatafile=args.metadata, duration=args.duration, split='test', device=args.device, samplerate=args.samplerate, pad_type=args.pad)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False)

### Optimizer
if METHOD == 'sl':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif METHOD == 'scl':
    optimizer = torch.optim.Adam(list(model.parameters()) + list(projector.parameters()), lr=args.lr, weight_decay=args.wd) 
    optimizer2 = torch.optim.Adam(classifier.parameters(), lr=args.lr2, weight_decay=args.wd)
elif METHOD == 'hybrid':
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters()), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6) 

if args.noweights:
    criterion_ce = nn.CrossEntropyLoss()  
else:
    weights = torch.tensor([2063, 1215, 501, 363], dtype=torch.float32) #N_COUNT, C_COUNT, W_COUNT, B_COUNT = 2063, 1215, 501, 363 for trainset
    weights = weights / weights.sum()
    weights = 1.0 / weights
    weights = weights / weights.sum()
    weights = weights.to(args.device)    
    criterion_ce = nn.CrossEntropyLoss(weight=weights)

if METHOD == 'sl':
    history = train_ce(model, train_loader, val_loader, train_transform, val_transform, criterion_ce, optimizer, args.epochs, scheduler)
    del model

elif METHOD == 'scl':
    criterion = SupConLoss(temperature=args.tau, device=args.device)
    ssl_train_losses, model, last_checkpoint = train_scl(model, projector, train_loader, train_transform, criterion, optimizer, scheduler, args.epochs)
    history = linear_scl(model, last_checkpoint ,classifier, train_loader, val_loader, val_transform, criterion_ce, optimizer2, args.epochs2)
    del model; del projector; del classifier
    
elif METHOD == 'hybrid':
    criterion = SupConCELoss(temperature=args.tau, weights=weights, alpha=args.alpha, device=args.device)
    history = train_supconce(model, projector, classifier, train_loader, val_loader, train_transform, val_transform, criterion, criterion_ce, optimizer, args.epochs, scheduler)
    del model; del projector; del classifier

del train_ds; del val_ds
del train_loader; del val_loader
