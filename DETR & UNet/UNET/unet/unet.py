
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

class UNet(nn.Module):
    def __init__(self, n_class, load_pretrained_encoder_layers=False):
        super(UNet, self).__init__()
        
        self.base_model = models.resnet18(pretrained=load_pretrained_encoder_layers)
        self.base_layers = list(self.base_model.children())[:8]

        self.layer0 = nn.Sequential(*self.base_layers[:3]) 
        self.layer011 = conv_f(64, 64, 1, 0)

        self.layer1 = nn.Sequential(*self.base_layers[3:5]) 
        self.layer111 = conv_f(64, 64, 1, 0)

        self.layer2 = self.base_layers[5]  
        self.layer211 = conv_f(128, 128, 1, 0)

        self.layer3 = self.base_layers[6]  
        self.layer311 = conv_f(256, 256, 1, 0)

        self.layer4 = self.base_layers[7]  
        self.layer411 = conv_f(512, 512, 1, 0)

     
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_original_0 = conv_f(3, 64, 3, 1)
        self.conv_original_1 = conv_f(64, 64, 3, 1)
        self.conv_original_2 = conv_f(192, 64, 3, 1)
        self.conv3 = conv_f(768, 512, 3, 1)
        self.conv2 = conv_f(640, 256, 3, 1)
        self.conv1 = conv_f(320, 256, 3, 1)
        self.conv0 = conv_f(320, 128, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original=self.conv_original_0(input)
        x_original=self.conv_original_1(x_original)
        layer0=self.layer0(input)
        layer1=self.layer1(layer0)
        layer2=self.layer2(layer1)
        layer3=self.layer3(layer2)
        layer4=self.layer4(layer3)
        layer4=self.layer411(layer4)
        x=self.upsample(layer4)
        layer3=self.layer311(layer3)
        x=torch.cat([x,layer3],dim=1)
        x=self.conv3(x)
        x=self.upsample(x)
        layer2=self.layer211(layer2)
        x=torch.cat([x,layer2],dim=1)
        x=self.conv2(x)
        x=self.upsample(x)
        layer1=self.layer111(layer1)
        x=torch.cat([x,layer1],dim=1)
        x=self.conv1(x)
        x=self.upsample(x)
        layer0=self.layer011(layer0)
        x=torch.cat([x,layer0],dim=1)
        x=self.conv0(x)
        x=self.upsample(x)
        x=torch.cat([x,x_original],dim=1)
        x=self.conv_original_2(x)
    
        output = self.conv_last(x)

        return output

def conv_f(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
        nn.ReLU(inplace=True)
    )
    
class SumUNet(nn.Module):
    def __init__(self, n_class, load_pretrained_encoder_layers=False):
        super().__init__()

        resnet = torchvision.models.resnet50(pretrained=load_pretrained_encoder_layers)
        
        self.base_layers = nn.Sequential(
            resnet.conv1, 
            resnet.bn1, 
            resnet.relu, 
            resnet.maxpool)
        
        self.encoder_1 = resnet.layer1
        self.encoder_2 = resnet.layer2
        self.encoder_3 = resnet.layer3
        self.encoder_4 = resnet.layer4
        
        self.decoder_4 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_3 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_1 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=1, output_padding=0)

        self.final_layer = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, input):

        base = self.base_layers(input)
        layer_1 = self.encoder_1(base)
        layer_2 = self.encoder_2(layer_1)
        layer_3 = self.encoder_3(layer_2)
        layer_4 = self.encoder_4(layer_3)

        Upper_4 = self.decoder_4(layer_4)
        Sum_4 = Upper_4 + layer_3
        Upper_3 = self.decoder_3(Sum_4)
        Sum_3 = Upper_3 + layer_2
        Upper_2 = self.decoder_2(Sum_3)
        Sum_2 = Upper_2 + layer_1
        Upper_1 = self.decoder_1(Sum_2)
        Sum_1 = Upper_1 + base

        output = self.final_layer(Sum_1)
  
        output = torch.nn.functional.interpolate(output,size = 256)

        return output