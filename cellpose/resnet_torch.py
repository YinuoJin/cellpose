import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime


from . import transforms, io, dynamics, utils
from . import cgru

sz = 3

def convbatchrelu(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )  

def batchconv(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )  

def batchconv0(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )  

class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj  = batchconv0(in_channels, out_channels, 1)
        for t in range(4):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz))
                
    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x

class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz))
                
    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x

class downsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase)-1):
            if residual_on:
                self.down.add_module('res_down_%d'%n, resdown(nbase[n], nbase[n+1], sz))
            else:
                self.down.add_module('conv_down_%d'%n, convdown(nbase[n], nbase[n+1], sz))
            
    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n>0:
                y = self.maxpool(xd[n-1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd
    
class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.concatenation = concatenation
        if concatenation:
            self.conv = batchconv(in_channels*2, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels*2)
        else:
            self.conv = batchconv(in_channels, out_channels, sz)
            self.full = nn.Linear(style_channels, out_channels)
        
    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None:
            if self.concatenation:
                x = torch.cat((y, x), dim=1)
            else:
                x = x + y
        feat = self.full(style)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat.unsqueeze(-1).unsqueeze(-1)).to_mkldnn()
        else:
            y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y
    
class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, concatenation=concatenation))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.proj  = batchconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
        return x
    
class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, concatenation=concatenation))
        
    def forward(self, x, y, style, mkldnn=False):
        x = self.conv[1](style, self.conv[0](x), y=y)
        return x
    
class make_style(nn.Module):
    def __init__(self):
        super().__init__()
        #self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        #style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2],x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5

        return style
    
class upsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, concatenation=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1,len(nbase)):
            if residual_on:
                self.up.add_module('res_up_%d'%(n-1), 
                    resup(nbase[n], nbase[n-1], nbase[-1], sz, concatenation))
            else:
                self.up.add_module('conv_up_%d'%(n-1), 
                    convup(nbase[n], nbase[n-1], nbase[-1], sz, concatenation))

    def forward(self, style, xd, mkldnn=False):
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up)-2,-1,-1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x
    
class CPnet(nn.Module):
    def __init__(self, nbase, nout, sz,
                residual_on=True, style_on=True, 
                concatenation=False, mkldnn=False,
                is_szmodel=False, do_cgru=False,
                diam_mean=30.):

        torch.manual_seed(0)

        super(CPnet, self).__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation)
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], nout, 1)
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.style_on = style_on

        self.is_szmodel = is_szmodel
        self.path = '/home/yinuo/Desktop/azizi_lab/cell-segmentation/results/cpnet/'
        self.do_cgru = do_cgru

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # ConvBGRU class requires device def. a priori.

        # DEBUG: try CGRU for each cascade level
        self.cgru = CGRU(in_channels=3)

        # DEBUG: layers for global weights
        self._w_s = nn.Parameter(torch.ones((1, 5), device=device)*1/5, requires_grad=True)

    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()
        T0    = self.downsample(data)

        # Cross-slice contextual info. learning;
        # BN layer dim: T0[-1].shape -> [B, C=256, Y=28, X=28]

        mid_idx = T0[-1].shape[0] // 2
        if self.do_cgru:
            for i, T0_lvl in enumerate(T0[:-1]):
                T0[i] = T0_lvl[mid_idx:mid_idx+1]
            T0[-1] = self.cgru(T0[-1])

        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0

        T0 = self.upsample(style, T0, self.mkldnn)
        T0 = self.output(T0)
        if self.mkldnn:
            T0 = T0.to_dense()

        # DEBUG: implement simple global kernel (W1, ..., Wn) + N-to-1
        # if self.do_cgru:
        #    T0 = torch.einsum("wz,zcyx->wcyx", self.w_s, T0)

        return T0, style0

    @property
    def w_s(self):
        return F.softmax(self._w_s, dim=-1)

    def update_sz_model(self, is_sz):
        self.is_szmodel = is_sz

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device=None):
        if (device is not None) and (device.type != 'cpu'):
            state_dict = torch.load(filename, map_location=device)
        else:
            self.__init__(self.nbase,
                          self.nout,
                          self.sz,
                          self.residual_on,
                          self.style_on,
                          self.concatenation,
                          self.mkldnn,
                          self.diam_mean)
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
        self.load_state_dict(dict([(name, param) for name, param in state_dict.items()]), strict=False)


# Add CGRU architecture
# now we have 4D tensor as input & output, we treat batch size == depth for CGRU
# x_in dim: [B/Z, C=2, Y, X], x_out dim: [B/Z, C=3, Y, X]

# TODO: compare
#  - N-N vs. N-1 (without masked autoregression)
#  - add stateful CGRU (see cgru.py line 58, 103, 114)
#  - Try freezing all Cellpose & only retrain CGRU part
class CGRU(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels=256,
            ksize=5
    ):
        super(CGRU, self).__init__()

        self.enc = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.bgru = cgru.ConvBGRU(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels//2,
            kernel_size=(ksize, ksize),
            num_layers=1,
            batch_first=True,
            # combine_option='avg',
            combine_option='concat'
        )

        # DEBUG
        self.out_channels = hidden_channels//2
        
        self.dec = nn.Conv3d(hidden_channels, in_channels, kernel_size=1)
        self.mask_layer = None  # to be deleted
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # ConvBGRU class requires device def. a priori.

        self.enc, self.dec, self.bgru = self.enc.to(self.device), self.dec.to(self.device), self.bgru.to(self.device)
        self.bgru.apply(self._init_weights)

    def forward(self, x):
        x = x.to(self.device)

        """
        # CGRU attached to Cellpose 2nd to last Decoder layer
        x = torch.transpose(
            torch.unsqueeze(x, dim=0),
            1, 2
        )  # dim: [B, C, H, W] -> [1, C, B, H, W]
        x = self.enc(x)

        x_fwd = x.transpose(1, 2)  # dim: [1, C, B, H, W] -> [1, B, C, H, W] for CGRU
        x_bwd = x_fwd.flip(1)
        x_gru = self.bgru(x_fwd, x_bwd)

        x_out = self.dec(x_gru.transpose(1, 2))  # dim: [1, B, C, H, W] -> [1, C, B, H. W] for decoder
        out = torch.squeeze(
            torch.transpose(x_out, 1, 2),
            dim=0)  # dim: [1, C, B, H, W] -> [B, C, H, W]
        """

        # CGRU attached to Cellpose BN layer
        x_fwd = torch.unsqueeze(x, dim=0)  # [B, C, H, W] -> [1, B, C, H, W]
        x_bwd = x_fwd.flip(1)
        x_gru = self.bgru(x_fwd, x_bwd)
        out = torch.squeeze(x_gru, dim=0)  # [1, B, C, H, W] -> [B, C, H, W]

        # N-1 prediction [x-n,..., x+n] -> y_i, only return axis Z=mid_idx
        mid_idx = out.shape[0] // 2
        return out[mid_idx:mid_idx+1]

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
