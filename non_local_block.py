import torch
import torch.nn as nn
import torch.nn.functional as F


class Non_Local_Block(nn.Module):
  
  def __init__(self, in_chns, out_chns, inter_chns=None, kernel_size=None):
    super(Non_Local_Block, self).__init__()
    
    if inter_chns is None:
      inter_chns = in_chns // 2
      
    self.W_theta = nn.Conv3d(in_chns, inter_chns, 1)
    self.W_phi = nn.Conv3d(in_chns, inter_chns, 1)
    self.g = nn.Conv3d(in_chns, inter_chns, 1)
    
    self.conv = nn.Conv3d(inter_chns, out_chns, 1)
    
  def forward(self, x):
    B,C,T,H,W = x.size()
    
    x_theta = self.W_theta(x).view(B,C,-1)
    x_phi = self.W_phi(x).view(B,C,-1)
    relations = F.softmax(torch.matmul(x.permute(0,2,1), x), -1) # THW x THW
    
    g_x = self.g(x).view(B,C,-1) # C x THW
    res = torch.matmul(relations, g_x.permute(0,2,1))
    
    return self.conv(res)
