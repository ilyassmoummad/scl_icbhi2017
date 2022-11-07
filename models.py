import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module): #for CNN10 & CNN14
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class ConvBlock5x5(nn.Module): #for CNN6
    def __init__(self, in_channels, out_channels, stride=(1,1)):
        
        super(ConvBlock5x5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(5, 5), stride=stride,
                              padding=(2, 2), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class _CNN6(nn.Module):
    def __init__(self, num_classes=4, do_dropout=False, embed_only=False):
        super(_CNN6, self).__init__()

        self.embed_only = embed_only
        self.do_dropout = do_dropout
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64, stride=(1,1))
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128, stride=(1,1))
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256, stride=(1,1))
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512, stride=(1,1))
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)

        x = torch.mean(x, dim=3) #mean over time dim
        (x1, _) = torch.max(x, dim=2) #max over freq dim
        x2 = torch.mean(x, dim=2) #mean over freq dim (after mean over time)
        x = x1 + x2

        if self.embed_only:
            return x
        return self.linear(x)

class CNN6(nn.Module):
    def __init__(self, num_classes=4, do_dropout=False, embed_only=False, from_scratch=False, path_to_weights="Cnn6_mAP=0.343.pth", device="cuda"):
        super(CNN6, self).__init__()

        self.cnn6 = _CNN6(num_classes=num_classes, do_dropout=do_dropout, embed_only=embed_only).to(device)
        if not from_scratch:
            weights = torch.load(path_to_weights, map_location=device)['model']
            state_dict = {k: v for k, v in weights.items() if k in self.cnn6.state_dict().keys()}
            self.cnn6.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.cnn6(x)

class _CNN10(nn.Module):
    def __init__(self, num_classes=4, do_dropout=False, embed_only=False):
        super(_CNN10, self).__init__()

        self.embed_only = embed_only
        self.do_dropout = do_dropout
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)

        x = torch.mean(x, dim=3) #mean over time dim
        (x1, _) = torch.max(x, dim=2) #max over freq dim
        x2 = torch.mean(x, dim=2) #mean over freq dim (after mean over time)
        x = x1 + x2

        if self.embed_only:
            return x
        return self.linear(x)

class CNN10(nn.Module):
    def __init__(self, num_classes=4, do_dropout=False, embed_only=False, from_scratch=False, path_to_weights="Cnn10_mAP=0.380.pth", device="cuda"):
        super(CNN10, self).__init__()

        self.cnn10 = _CNN10(num_classes=num_classes, do_dropout=do_dropout, embed_only=embed_only).to(device)
        if not from_scratch:
            weights = torch.load(path_to_weights, map_location=device)['model']
            state_dict = {k: v for k, v in weights.items() if k in self.cnn10.state_dict().keys()}
            self.cnn10.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.cnn10(x)

class _CNN14(nn.Module):
    def __init__(self, num_classes=4, do_dropout=False, embed_only=False):
        super(_CNN14, self).__init__()

        self.embed_only = embed_only
        self.do_dropout = do_dropout
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x):

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        if self.do_dropout:
            x = self.dropout(x)

        x = torch.mean(x, dim=3) #mean over time dim
        (x1, _) = torch.max(x, dim=2) #max over freq dim
        x2 = torch.mean(x, dim=2) #mean over freq dim (after mean over time)
        x = x1 + x2

        if self.embed_only:
            return x
        return self.linear(x)

class CNN14(nn.Module):
    def __init__(self, num_classes=4, do_dropout=False, embed_only=False, from_scratch=False, path_to_weights="/users/local/Cnn14_mAP=0.431.pth", device="cuda"):
        super(CNN14, self).__init__()

        self.cnn14 = _CNN14(num_classes=num_classes, do_dropout=do_dropout, embed_only=embed_only).to(device)
        if not from_scratch:
            weights = torch.load(path_to_weights, map_location=device)['model']
            state_dict = {k: v for k, v in weights.items() if k in self.cnn14.state_dict().keys()}
            self.cnn14.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.cnn14(x)

class Projector(nn.Module):
    def __init__(self, name='cnn6', out_dim=128, apply_bn=False, device="cpu"):
        super(Projector, self).__init__()
        _, dim_in = model_dict[name]
        self.linear1 = nn.Linear(dim_in, dim_in)
        self.linear2 = nn.Linear(dim_in, out_dim)
        self.bn = nn.BatchNorm1d(dim_in)
        self.relu = nn.ReLU()
        if apply_bn:
            self.projector = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
        else:
            self.projector = nn.Sequential(self.linear1, self.relu, self.linear2)
        self.projector = self.projector.to(device)

    def forward(self, x):
        return self.projector(x)

class LinearClassifier(nn.Module):
    def __init__(self, name='cnn6', num_classes=4, device="cpu"):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes).to(device)

    def forward(self, features):
        return self.fc(features)

def cnn6(**kwargs):
    return CNN6(**kwargs)

def cnn10(**kwargs):
    return CNN10(**kwargs)

def cnn14(**kwargs):
    return CNN14(**kwargs)
    
model_dict = {
    'cnn6' : [cnn6, 512],
    'cnn10' : [cnn10, 512],
    'cnn14' : [cnn14, 2048],
}