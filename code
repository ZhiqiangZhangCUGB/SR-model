import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np


class SPP(nn.Module):
    def __init__(self, in_channels):
        super(SPP, self).__init__()
        self.out_channels = in_channels * 5  
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(4)
        self.pool3 = nn.AdaptiveAvgPool2d(8)
        self.pool4 = nn.AdaptiveAvgPool2d(16)
        self.conv = nn.Conv2d(self.out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = F.interpolate(self.pool1(x), size=x.size()[2:], mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.pool2(x), size=x.size()[2:], mode='bilinear', align_corners=False)
        x3 = F.interpolate(self.pool3(x), size=x.size()[2:], mode='bilinear', align_corners=False)
        x4 = F.interpolate(self.pool4(x), size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        return self.conv(x)

class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(torch.cat([x, x1], dim=1))))
        x3 = self.bn3(self.conv3(torch.cat([x, x1, x2], dim=1)))
        return x + x3


class EnhancedMZSRNet(nn.Module):
    def __init__(self, in_channels=64):
        super(EnhancedMZSRNet, self).__init__()
        self.input_conv = nn.Conv2d(1, in_channels, kernel_size=3, padding=1)
        self.spp = SPP(in_channels)
        self.dense_res_blocks = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(5)])
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels
                                           , kernel_size=4, stride=2, padding=1)  
        self.output_conv = nn.Conv2d(in_channels, 10, kernel_size=3, padding=1)
        self.skip_connection = nn.Conv2d(1, in_channels, kernel_size=3, padding=1)  

    def forward(self, x):
        skip = self.skip_connection(x)  
        x = F.relu(self.input_conv(x))
        x = self.spp(x)
        x = self.dense_res_blocks(x)
        x = self.upsample(x) 
        
        if skip.size() != x.size():
            skip = F.interpolate(skip, size=x.size()[2:]
                                 , mode='bilinear', align_corners=False)
        x = x + skip 
        return self.output_conv(x)


class FrequencySuperResolution(nn.Module):
    def __init__(self, in_channels=64):
        super(FrequencySuperResolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x_fft = torch.fft.fft2(x, dim=(-2, -1))  
        x_fft = torch.fft.fftshift(x_fft) 
        x_real = torch.real(x_fft).unsqueeze(1) 
        x_imag = torch.imag(x_fft).unsqueeze(1) 

        # Reshape to match 64 channels for convolution (remove extra dimension)
        x_real = x_real.squeeze(1).expand(-1, 64, -1, -1)
        x_imag = x_imag.squeeze(1).expand(-1, 64, -1, -1)


        x_real = F.relu(self.conv1(x_real))
        x_real = F.relu(self.conv2(x_real))
        x_real = self.output_conv(x_real)

        x_imag = F.relu(self.conv1(x_imag))
        x_imag = F.relu(self.conv2(x_imag))
        x_imag = self.output_conv(x_imag)


        x_fft = torch.complex(x_real, x_imag)
        x_fft = torch.fft.ifftshift(x_fft)  
        x_ifft = torch.fft.ifft2(x_fft, dim=(-2, 1)) 
        return torch.real(x_ifft)



def adaptive_interpolation(img, scale_factor):
    return F.interpolate(img, scale_factor=scale_factor, mode='bicubic', align_corners=False)


def frequency_loss(output, target):
    output_fft = torch.fft.fft2(output, dim=(-2, 1))
    target_fft = torch.fft.fft2(target, dim=(-2, 1))
    return F.mse_loss(torch.abs(output_fft), torch.abs(target_fft))


def train_self_supervised(model, original_image, scale_factor=4, num_epochs=100, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    frequency_sr = FrequencySuperResolution()
    

    lr_image = adaptive_interpolation(original_image, scale_factor=1/scale_factor)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        output_spatial = model(lr_image)
        
 
        output_frequency = frequency_sr(lr_image)
        
        # Ensure both outputs have the same size
        if output_spatial.size() != output_frequency.size():
            output_frequency = F.interpolate(output_frequency
                                             , size=output_spatial.shape[2:]
                                             , mode='bicubic'
                                             , align_corners=False)
        
  
        output_combined = (output_spatial + output_frequency) / 2
        
       
        output_resized = F.interpolate(output_combined, size=original_image.shape[2:], mode='bicubic', align_corners=False)
        
    
        mse_loss = F.mse_loss(output_resized, original_image) 
        perceptual_loss = torch.mean((output_resized - original_image) ** 2)  
        freq_loss = frequency_loss(output_resized, original_image)  
        loss = mse_loss + 0.01 * perceptual_loss + 0.01 * freq_loss  

        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")
def load_image(file_path):
    image = Image.open(file_path).convert('L')
    image_tensor = ToTensor()(image).unsqueeze(0)
    print("Loaded image with shape:", image_tensor.shape)
    return image_tensor
