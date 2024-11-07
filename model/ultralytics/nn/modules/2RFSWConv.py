from torch import nn
from einops import rearrange
import torch
import torch.nn.functional as F
import torch.autograd
 
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
 
 
 
class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
 
        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())
 
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)
 
    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data) 
 
 
 
class RFSW(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=1):
        super().__init__()
        # self.odc = ODConv2d(in_channel, out_channel, kernel_size, stride, padding=1)  
        self.rfac = RFAConv(in_channel, out_channel, kernel_size, stride)
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=stride, p=padding)
        self.squeeze = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=1)
        self.batchnorm = torch.nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('x:', x.shape)
        x1 = self.rfac(x)
        x2 = self.conv(x)
        x2 = self.batchnorm(x2)
        bias = self.sigmoid(x2)
        # print('bias:', bias)
        # print('x1:', x1.shape)
        # print('x2:', x2.shape) 
        out = x1*bias
        return out
    
if __name__ == '__main__':
    block = RFSW(64,128,3,2,1)
    input = torch.rand(1, 64, 320, 320)
    output = block(input)
    print(input.size(), output.size())