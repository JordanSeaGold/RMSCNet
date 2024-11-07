from torch import nn
from einops import rearrange
import torch
import torch.nn.functional as F
import torch.autograd
import numpy as np
from torch.nn import init
from thop import profile

 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        # x:(B,C,H,W)
        max_result,_=torch.max(x,dim=1,keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 最大值和对应的索引.
        avg_result=torch.mean(x,dim=1,keepdim=True)   # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 平均值
        result=torch.cat([max_result,avg_result],1)   # 在通道上拼接两个矩阵:(B,2,H,W)
        output=self.conv(result)                      # 然后重新降维为1维:(B,1,H,W)
        output=self.sigmoid(output)                   # 通过sigmoid获得权重:(B,1,H,W)
        return output
    
#感受野空间注意力
class RFA(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
 
        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)
    def forward(self, x):
        # B,C,H,W = x.shape
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        conv_data = rearrange(weighted, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

class RFSW(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=1):
        super().__init__()
        self.SpatialAttention=SpatialAttention(kernel_size=kernel_size)
        self.rfac = RFA(in_channel, out_channel, kernel_size, stride)
        self.convd = Conv(in_channel, out_channel, k=kernel_size, s=stride, p=padding)
        self.batchnorm = torch.nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                         
    def forward(self, x):
        weight1 = self.rfac(x)  #得到感受野加权后的特征图
        x3 = self.convd(x) #得到残差流
        weight2 = self.SpatialAttention(x3)
        rfca_result = weight1*x3
        SpatialAttention_result = weight2*x3
        out = rfca_result + SpatialAttention_result + x3
        return out
    
if __name__ == '__main__':
    block = RFSW(64,128,3,2,1)
    input = torch.rand(1, 64, 320, 320)
    output = block(input)
    print(input.size(), output.size())
    #获取模型GFLOPs和参数量
    flops, params = profile(block, inputs=(input,))
    gflops = flops / 1e9
    print(f"GFLOPs: {gflops}", f"Params: {params/1e6}M")