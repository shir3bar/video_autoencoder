from typing import Any
import torch.nn.functional as F
import torch.nn as nn


class Autoencoder(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, color_channels):
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Conv3d(color_channels, 8, 3)
        self.conv2 = nn.Conv3d(8, 16, 3)
        self.conv3 = nn.Conv3d(16, 32, 3)
        self.conv4 = nn.Conv3d(32, 64, 3)
        self.conv5 = nn.Conv3d(64, 128, 3)
        self.maxpool1 = nn.MaxPool3d((1, 2, 2), return_indices=True)
        self.unpool1 = nn.MaxUnpool3d((1, 2, 2))
        self.convt1 = nn.ConvTranspose3d(128, 64, 3)
        self.convt2 = nn.ConvTranspose3d(64, 32, 3)
        self.convt3 = nn.ConvTranspose3d(32, 16, 3)
        self.convt4 = nn.ConvTranspose3d(16, 8, 3)
        self.convt5 = nn.ConvTranspose3d(8, color_channels, 3)

    def encoder(self, x):
        indices = []
        output_size = []
        x = F.relu(self.conv1(x))
        output_size.append(x.size())
        x, index = self.maxpool1(x)
        indices.append(index)
        x = F.relu(self.conv2(x))
        output_size.append(x.size())
        x, index = self.maxpool1(x)
        indices.append(index)
        x = F.relu(self.conv3(x))
        output_size.append(x.size())
        x, index = self.maxpool1(x)
        indices.append(index)
        x = F.relu(self.conv4(x))
        output_size.append(x.size())
        x, index = self.maxpool1(x)
        indices.append(index)
        x = F.relu(self.conv5(x))
        output_size.append(x.size())
        x, index = self.maxpool1(x)
        indices.append(index)
        return x, indices, output_size

    def decoder(self, x, indices, output_size):
        x = self.unpool1(x, indices=indices.pop(-1), output_size=output_size.pop(-1))
        x = F.relu(self.convt1(x))
        x = self.unpool1(x, indices=indices.pop(-1), output_size=output_size.pop(-1))
        x = F.relu(self.convt2(x))
        x = self.unpool1(x, indices=indices.pop(-1), output_size=output_size.pop(-1))
        x = F.relu(self.convt3(x))
        x = self.unpool1(x, indices=indices.pop(-1), output_size=output_size.pop(-1))
        x = F.relu(self.convt4(x))
        x = self.unpool1(x, indices=indices.pop(-1), output_size=output_size.pop(-1))
        x = F.relu(self.convt5(x))
        return x

    def forward(self, x):
        x, indices, output_size = self.encoder(x)
        x = self.decoder(x, indices, output_size)
        return x


class GANomaly(Autoencoder):
    def __init__(self,color_channels):
        super(GANomaly, self).__init__(color_channels)
        self.conv6 = nn.Conv3d(color_channels, 8, 3)
        self.conv7 = nn.Conv3d(8, 16, 3)
        self.conv8 = nn.Conv3d(16, 32, 3)
        self.conv9 = nn.Conv3d(32, 64, 3)
        self.conv10 = nn.Conv3d(64, 128, 3)

    def encoder2(self, x):
        indices = []
        output_size = []
        x = F.relu(self.conv6(x))
        output_size.append(x.size())
        x, index = self.maxpool1(x)
        indices.append(index)
        x = F.relu(self.conv7(x))
        output_size.append(x.size())
        x, index = self.maxpool1(x)
        indices.append(index)
        x = F.relu(self.conv8(x))
        output_size.append(x.size())
        x, index = self.maxpool1(x)
        indices.append(index)
        x = F.relu(self.conv9(x))
        output_size.append(x.size())
        x, index = self.maxpool1(x)
        indices.append(index)
        x = F.relu(self.conv10(x))
        output_size.append(x.size())
        x, index = self.maxpool1(x)
        indices.append(index)
        return x, indices, output_size

    def forward(self, x):
        z_in, indices, output_size = self.encoder(x)
        img_out = self.decoder(z_in,indices,output_size)
        z_out, _, _ = self.encoder2(img_out)
        return z_in, img_out, z_out


