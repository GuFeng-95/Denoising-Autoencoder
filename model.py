import torch
from torch import nn
from torch.nn import init



class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        # 如果数据集最后一个batch样本数量小于定义的batch_batch大小，会出现mismatch问题。可以自己修改下，如只传入后面的shape，然后通过x.szie(0)，来输入。
        return x.view((x.size(0),)+self.shape)


class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


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
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class autoencoder_n(nn.Module):
    def __init__(self,w,h,vector_l):
        super(autoencoder_n, self).__init__()

        self.vector_l=vector_l

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # (b, 16, 384, 192)
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 192, 96)
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 8, 96, 48)
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 8, 48, 24)
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 8, 24, 12)
            nn.Conv2d(8, 4, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 4, 12, 6)
            nn.Conv2d(4, 4, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 4, 6, 3)
            Reshape(int(4*w*h/4096)),
            nn.Linear(int(4*w*h/4096),self.vector_l)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.vector_l,int(4*w*h/4096)),
            Reshape(4,int(w/64),int(h/64)),
            nn.Upsample(scale_factor=2, mode='nearest'),# (b, 4, 12, 6)
            nn.Conv2d(4, 4, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (b, 4, 24, 12)
            nn.Conv2d(4, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (b, 8, 48, 24)
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (b, 8, 96, 48)
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (b, 8, 192, 96)
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (b, 16, 384, 192)
            nn.Conv2d(16, 1, 3, stride=1, padding=1)
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class autoencoder_attention_n(nn.Module):
    def __init__(self,w,h,n):
        super(autoencoder_attention_n, self).__init__()
        self.n=n

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # (b, 16, 384, 192)
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 192, 96)
            SEAttention(16, 2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 8, 96, 48)
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 8, 48, 24)
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 8, 24, 12)
            nn.Conv2d(8, 4, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 4, 12, 6)
            nn.Conv2d(4, 4, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # (b, 4, 12, 6)
            # SEAttention(16,2),
            # nn.AdaptiveAvgPool2d(1)
            Reshape(int(4 * w * h / 4096)),
            nn.Linear(int(4 * w * h / 4096), self.n)

        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n, int(4 * w * h / 4096)),
            Reshape(4, int(w / 64), int(h / 64)),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (b, 1, 12, 6)
            nn.Conv2d(4, 4, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),# (b, 1, 12, 6)
            nn.Conv2d(4, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (b, 8, 48, 24)
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (b, 8, 96, 48)
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (b, 8, 192, 96)
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (b, 16, 384, 192)
            SEAttention(16, 2),
            nn.Conv2d(16, 1, 3, stride=1, padding=1)
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode



if __name__=='__main__':
    x = torch.ones(16,1, 384, 192)
    a=autoencoder_n(384, 192,8)

    v,y=a.forward(x)
    print(y.shape)