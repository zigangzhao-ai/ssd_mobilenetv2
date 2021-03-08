'''
code by zzg-2020-11-10
'''
import torch
from thop import profile
from ssd_fpn0 import build_ssd

input = torch.randn(1, 3, 300, 300) #模型输入的形状,batch_size=1
ssd_net = build_ssd('train', 300, 2)
model = ssd_net
flops, params = profile(model, inputs=(input, ))
print(flops/1e9,params/1e6) #flops单位G，para单位M