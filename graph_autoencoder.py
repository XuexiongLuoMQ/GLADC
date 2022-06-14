# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GINConv, global_add_pool

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = True
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):

        y = torch.matmul(adj, x)

        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias

        return y

class Encoder(nn.Module):
    def __init__(self, feat_size,hiddendim,outputdim,dropout,batch):
        super(Encoder, self).__init__()
        self.gc1 = nn.Linear(feat_size, hiddendim, bias=False)
        #self.gc2 = nn.Linear(hiddendim*2, hiddendim*2, bias=False)
        #self.gc3 = nn.Linear(hiddendim*2, hiddendim, bias=False)
        self.gc4 = nn.Linear(hiddendim, outputdim, bias=False)
        self.proj_head = nn.Sequential(nn.Linear(outputdim, outputdim), nn.ReLU(inplace=True), nn.Linear(outputdim, outputdim))
        self.leaky_relu = nn.LeakyReLU(0.5)
        self.dropout = nn.Dropout(dropout)
        self.batch=batch
    

        

    def forward(self, x, adj):
        x = self.leaky_relu(self.gc1(torch.matmul(adj, x)))

        x=self.dropout(x)
        
        x = self.gc4(torch.matmul(adj, x))
        out, _ = torch.max(x, dim=1)
        #out = global_add_pool(x,self.batch)
        out=self.proj_head(out)
        
        return x,out

class attr_Decoder(nn.Module):
    def __init__(self, feat_size,hiddendim,outputdim,dropout):
        super(attr_Decoder, self).__init__()

        self.gc1 = nn.Linear(outputdim, hiddendim, bias=False)
        #self.gc2 = nn.Linear(hiddendim, hiddendim*2, bias=False)
        #self.gc3 = nn.Linear(hiddendim*2, hiddendim*2, bias=False)
        self.gc4 = nn.Linear(hiddendim, feat_size, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.5)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x, adj):
        x = self.leaky_relu(self.gc1(torch.matmul(adj, x)))
        x=self.dropout(x)
        
        x = self.gc4(torch.matmul(adj, x))
        
            
        return x    

class stru_Decoder(nn.Module):
    def __init__(self, feat_size,outputdim,dropout):
        super(stru_Decoder, self).__init__()

        #self.gc1 = nn.Linear(outputdim, outputdim, bias=False)
        self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(dropout)
    def forward(self, x, adj):

        x1=x.permute(0, 2, 1)
        x = torch.matmul(x,x1) 
        x=self.sigmoid(x)
        return x

class NetGe(nn.Module):
    def __init__(self, feat_size, hiddendim, outputdim, dropout,batch):
        super(NetGe, self).__init__()
        
        self.shared_encoder = Encoder(feat_size, hiddendim, outputdim, dropout,batch)
        self.attr_decoder = attr_Decoder(feat_size, hiddendim, outputdim, dropout)
        self.struct_decoder = stru_Decoder(feat_size, outputdim, dropout)
    
    def forward(self, x, adj):
        
        x_fake= self.attr_decoder(x, adj)
        s_fake = self.struct_decoder(x, adj)
        x2,Feat_1=self.shared_encoder(x_fake, s_fake)

            
        return x_fake,s_fake,x2,Feat_1

class NetDe(nn.Module):
    def __init__(self, feat_size, hiddendim, outputdim, dropout,batch):
        super(NetDe, self).__init__()
        
        #self.gc1 = nn.Linear(feat_size, hiddendim, bias=False)
        #self.gc2 = nn.Linear(hiddendim, outputdim, bias=False)
        self.shared_encoder = Encoder(feat_size, hiddendim, outputdim, dropout,batch)
        self.leaky_relu = nn.LeakyReLU(0.5)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.FloatTensor(outputdim, 1).cuda())
        init.xavier_uniform_(self.weight)
        self.m=nn.Sigmoid()
    
    
    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)
    
    def forward(self, x, adj):
        # encode
        #x = self.leaky_relu(self.gc1(torch.matmul(adj, x)))
        #x=self.dropout(x)
        #x = self.apply_bn(x)
        #x = self.gc2(torch.matmul(adj, x))
        #x = self.apply_bn(x)
        #Feat = torch.mean(x,dim=1).squeeze(1)
        x,Feat = self.shared_encoder(x, adj)
        out_emb=torch.mm(Feat,self.weight)
        out_emb=self.dropout(out_emb)
        pred=self.m(out_emb)
        pred=pred.view(-1, 1).squeeze(1)

       
        return pred,Feat
