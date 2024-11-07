import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from thop import profile
 
class MSAttention(nn.Module):#多层感知空间注意力
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(1,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.mlp=nn.Sequential(nn.Linear(2,1), nn.ReLU(inplace=True))
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        # x:(B,C,H,W)
        max_result,_=torch.max(x,dim=1,keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 最大值和对应的索引.
        avg_result=torch.mean(x,dim=1,keepdim=True)   # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 平均值
        max_conv = self.conv(max_result)
        avg_conv = self.conv(avg_result)
        result=torch.cat([max_conv,avg_conv],1)   # 在通道上拼接两个矩阵:(B,2,H,W)
        output=self.mlp(result.permute(0,3,2,1)).permute(0,3,2,1)                      # 然后重新降维为1维:(B,1,H,W)
        output=self.sigmoid(output)                   # 通过sigmoid获得权重:(B,1,H,W)
        return output
 
 
class MCRAttention(nn.Module):#多层感知通道衰减空间注意力
    '''
    alpha: 0<alpha<1
    '''
 
    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: float = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.low_channel1 = low_channel1 = int(alpha * op_channel)
        self.low_channel2 = low_channel2 = op_channel - low_channel1
        self.squeeze1 = nn.Conv2d(low_channel1, low_channel1 // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel2, low_channel2 // squeeze_radio, kernel_size=1, bias=False)
        #MLP
        self.mlp1 = nn.Sequential(nn.Linear(low_channel1 // squeeze_radio, low_channel1), nn.ReLU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(low_channel1 // squeeze_radio, low_channel2), nn.ReLU(inplace=True))
        
        # up
        self.G_extr = nn.Conv2d(op_channel, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=op_channel)
        self.advavg = nn.AdaptiveAvgPool2d(1)
 
    def forward(self, x):
        # Split
        low1, low2 = torch.split(x, [self.low_channel1,self.low_channel2], dim=1)
        low1, low2 = self.squeeze1(low1), self.squeeze2(low2)
        #MLP Transform
        mlp1, mlp2 = self.mlp1(low1.permute(0,3,2,1)), self.mlp2(low2.permute(0,3,2,1))
        # Fuse
        y = torch.cat([mlp1.permute(0,3,2,1), mlp2.permute(0,3,2,1)], dim=1)
        out = self.G_extr(y)
        out = F.softmax(self.advavg(out), dim=1)
        return out
 
class GCA(nn.Module):#空间通道注意力
    def __init__(self, in_channels, rate=4):
        super().__init__()
        out_channels = in_channels
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels/rate)
 
 
        self.linear1 = nn.Linear(in_channels, inchannel_rate)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(inchannel_rate, out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self,x):
        b, c, h, w = x.shape
        # B,C,H,W ==> B,H*W,C
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        
        # B,H*W,C ==> B,H,W,C
        x_att_permute = self.linear2(self.relu(self.linear1(x_permute))).view(b, h, w, c)
 
        # B,H,W,C ==> B,C,H,W
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        
        x_channel_att_weight = self.sigmoid(self.norm(x_channel_att))
 
        out = x * x_channel_att_weight

        return out

class GSPA(nn.Module):
    def __init__(self, dim, out_dim, k_size = 3):
        super().__init__()
        self.msa_w = MSAttention(kernel_size=k_size)
        self.mcra_w = MCRAttention(dim,group_kernel_size=k_size)
        self.gca = GCA(dim)
        # self.apply(self.init_weights) # 调用初始化函数，初始化参数    
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
        x_msa = self.msa_w(x) * x
        x_mcra = self.mcra_w(x) * x
        x_gca = self.gca(x)
        return x+ x_msa + x_mcra + x_gca

if __name__ == '__main__':
    block = GSPA(64,64)
    input = torch.rand(3, 64, 80, 80)
    output = block(input)
    print(input.size(), output.size())
    #获取模型GFLOPs和参数量(from thop import profile)
    flops, params = profile(block, inputs=(input,))
    gflops = flops / 1e9
    print(f"GFLOPs: {gflops}", f"Params: {params/1e6}M")