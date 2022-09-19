import torch
import torch.nn as nn
# import torchvision.transforms.functional as TF
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, input_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(Unet,self).__init__()
        self.downs=nn.ModuleList()
        self.ups=nn.ModuleList()
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

        for feature in features:
             self.downs.append(DoubleConv(input_channels, feature))
             input_channels=feature
        for feature in (reversed(features)):
            # print(feature)
            self.ups.append(
                nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2)
            )
            self.ups.append(DoubleConv(feature*2,feature))

        self.bottleneck=DoubleConv(features[-1],features[-1]*2)

        # self.final_conv=nn.Conv2d(features[0],out_channels,kernel_size=1)
        self.pos_output = nn.Conv2d(features[0],out_channels,kernel_size=1)
        self.cos_output = nn.Conv2d(features[0],out_channels,kernel_size=1)
        self.sin_output = nn.Conv2d(features[0],out_channels,kernel_size=1)
        self.width_output = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
        skip_connections=[]
        for down in self.downs:
            x=down(x)
            skip_connections.append(x)
            x=self.pool(x)
        x=self.bottleneck(x)
        skip_connections=skip_connections[::-1]
        for idx in range(0,len(self.ups),2):
            x=self.ups[idx](x)
            skip=skip_connections[idx//2]
            # print(x.shape,skip.shape)
            if x.shape!=skip.shape:
                diffY = skip.size()[2] - x.size()[2]
                diffX = skip.size()[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
            #     x=TF.resize(x,size=skip.shape)

            concat_skip=torch.cat([skip,x],dim=1)
            x=self.ups[idx+1](concat_skip)

        # x=self.final_conv(x)
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        return pos_output, cos_output, sin_output, width_output
    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }




if __name__ == '__main__':
    print(1)
    x=torch.rand((3,3,300,300))
    unet=Unet(input_channels=3, out_channels=1)
    # print(unet)
    pred=unet(x)
    print(x[0].shape)
    print(pred[0].shape)