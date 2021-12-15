import torch
import torch.nn as nn
import torch.nn.functional as F


class Non_Local_Block(nn.Module):
  
  def __init__(self, in_chns, out_chns, dimension=3, inter_chns=None, sub_sample=True, with_bn=True):
    super(Non_Local_Block, self).__init__()
    
    assert dimension == [1, 2, 3]
    
    if inter_chns is None:
      inter_chns = in_chns // 2
      
    if dimension == 3:
      conv_nd = nn.Conv3d
      max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
      bn = nn.BatchNorm3d
      
    elif dimension == 2:
      conv_nd = nn.Conv2d
      max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
      bn = nn.BatchNorm2d
      
    else:
      conv_nd = nn.Conv1d
      max_pool_layer = nn.MaxPool2d(kernel_size=(2))
      bn = nn.BatchNorm1d
      
    self.W_theta = conv_nd(in_chns, inter_chns, 1)
    self.W_phi = conv_nd(in_chns, inter_chns, 1)
    self.g = conv_nd(in_chns, inter_chns, 1)
    
    if with_bn:
      self.W_final = nn.Sequential(
      conv_nd(inter_chns, out_chns, 1),
      bn(out_chns)
      )
      nn.init.constant_(self.W_final.weight, 0)
      nn.init.constant_(self.W_final.bias, 0)
      
    else:
      self.W_final = nn.Sequential(
      conv_nd(inter_chns, out_chns, 1),
      bn(out_chns)
      )
    
    if sub_sample:
      self.g = nn.Sequential(self.g, max_pool_layer)
      self.W_phi = nn.Sequential(self.W_phi, max_pool_layer)
    
  def forward(self, x, type='embeded_gaussian'):
    B,C,T,H,W = x.size()
    
    x_theta = self.W_theta(x).view(B,C,-1)
    x_phi = self.W_phi(x).view(B,C,-1)
    relations = F.softmax(torch.matmul(x_theta.permute(0,2,1), x_phi), -1) # THW x THW
    
    g_x = self.g(x).view(B,C,-1) # C x THW
    y = torch.matmul(relations, g_x.permute(0,2,1))
    y = y.permute(0,2,1).contiguous
    y = y.view(B,C,T,H,W)
    
    W_y = self.W_final(y)
    return W_y + x
    

