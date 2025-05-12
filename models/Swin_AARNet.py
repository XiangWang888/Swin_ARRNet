import torch.nn as nn
import torch

class Swin_AARNet(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(Swin_AARNet, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.fc1 = nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(int(in_channels / rate) // 2, int(in_channels / rate) // 2, kernel_size=1, bias=False)
        self.RELU = nn.ReLU(inplace=True)
        self.SSA = nn.Sequential(
            nn.Conv2d(int(in_channels / rate) // 2, int(in_channels / rate) // 2, kernel_size=7, padding=3, 
            groups=int(in_channels / rate) // 2),

            nn.Conv2d(int(in_channels / rate) // 2, int(in_channels / rate) // 2, kernel_size=7, stride=1, 
                     padding=9, groups=int(in_channels / rate) // 2, dilation=3),

            nn.Conv2d(int(in_channels / rate) // 2, int(in_channels / rate) // 2, kernel_size=1, bias=False)
        )
        self.conv2 = nn.Conv2d(int(in_channels / rate) // 2, int(in_channels / rate) // 2, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.fc2 = nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), int(in_channels / rate), kernel_size=7, padding=3, 
                     groups=int(in_channels / rate), bias=False),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x_norm = self.norm1(x)        
        x_fc1 = self.fc1(x_norm)
        x1, x2 = torch.split(x_fc1, x_fc1.size(1) // 2, dim=1)

        u1 = x1.clone()
        attn1 = self.conv1(x1)
        attn1 = self.RELU(attn1)
        a1 = self.SSA(attn1)
        out1 = a1 * attn1
        out1 = self.conv2(out1)


        out1 = u1 + self.alpha * out1
        
        u2 = x2.clone()        
        attn2 = self.conv1(x2)
        attn2 = self.RELU(attn2)
        a2 = self.SSA(attn2)
        out2 = a2 * attn2
        out2 = self.conv2(out2)
        out2 = u2 + self.beta * out2

        out = torch.cat([out1, out2], dim=1)

        x_permute = self.fc2(out)
        x_spatial_att = self.spatial_attention(x_permute).sigmoid()
        out = x_permute * x_spatial_att
 
        return out


if __name__ == '__main__':
    x = torch.randn(16, 64, 224, 224)
    b, c, h, w = x.shape
    net = Swin_AARNet(in_channels=c, out_channels=c)
    y = net(x)
    print(net)
    print("Output shape:", y.shape)
    print("Alpha value:", net.alpha.item())
    print("Beta value:", net.beta.item())