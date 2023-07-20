from typing import Any
import torch.nn.functional as F
import torch.nn as nn


class Autoencoder(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, color_channels):
        super(Autoencoder, self).__init__()
        # self.conv1 = nn.Conv3d(color_channels, 8, 3) # 1x20x256x256
        # self.conv2 = nn.Conv3d(8, 16, 3) #
        # self.conv3 = nn.Conv3d(16, 32, 3)
        # self.conv4 = nn.Conv3d(32, 64, 3)
        # self.conv5 = nn.Conv3d(64, 128, 3)
        # self.maxpool1 = nn.MaxPool3d((1, 2, 2), return_indices=True)
        # self.unpool1 = nn.MaxUnpool3d((1, 2, 2))
        # self.convt1 = nn.ConvTranspose3d(128, 64, 3)
        # self.convt2 = nn.ConvTranspose3d(64, 32, 3)
        # self.convt3 = nn.ConvTranspose3d(32, 16, 3)
        # self.convt4 = nn.ConvTranspose3d(16, 8, 3)
        # self.convt5 = nn.ConvTranspose3d(8, color_channels, 3)
        initial_filters=32
        self.conv1 = nn.Conv3d(color_channels, initial_filters,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv3d(initial_filters, initial_filters*2, kernel_size=3,stride=1,padding=1) #
        self.conv3 = nn.Conv3d(initial_filters*2, initial_filters*(2 ** 2), kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv3d(initial_filters*(2 ** 2), initial_filters*(2 ** 3), kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv3d(initial_filters*(2 ** 3), initial_filters*(2 ** 4), kernel_size=3,stride=1,padding=1)
        self.conv6 = nn.Conv3d(initial_filters*(2 ** 4),initial_filters*(2 ** 5),kernel_size=3,stride=1,padding=1)
        self.conv7 = nn.Conv3d(initial_filters*(2 ** 5),512,kernel_size=3,stride=1,padding=0)
        self.convt1 = nn.ConvTranspose3d(512, initial_filters*(2 ** 5), kernel_size=3,stride=1,padding=0)
        self.convt2 = nn.ConvTranspose3d(initial_filters*(2 ** 5), initial_filters*(2 ** 4), kernel_size=3,stride=1,padding=1)
        self.convt3 = nn.ConvTranspose3d(initial_filters*(2 ** 4), initial_filters*(2 ** 3), kernel_size=3,stride=1,padding=1)
        self.convt4 = nn.ConvTranspose3d(initial_filters*(2 ** 3), initial_filters*(2 ** 2), kernel_size=3,stride=1,padding=1)
        self.convt5 = nn.ConvTranspose3d(initial_filters*(2 ** 2), initial_filters*2, kernel_size=3,stride=1,padding=1)
        self.convt6 = nn.ConvTranspose3d(initial_filters*2, initial_filters, kernel_size=3,stride=1,padding=1)
        self.convt7 = nn.ConvTranspose3d(initial_filters, color_channels, kernel_size=3,stride=1,padding=1)

    def encoder(self, x):
        indices = []
        output_size = []
        x = F.relu(self.conv1(x))
        output_size.append(x.size())
        x, index = F.max_pool3d(x,kernel_size=(2, 2, 2), stride=(1,2,2), return_indices=True)
        indices.append(index)
        x = F.relu(self.conv2(x))
        output_size.append(x.size())
        x, index = F.max_pool3d(x,kernel_size=(2, 2, 2), stride=(1,2,2),return_indices=True)
        indices.append(index)
        x = F.relu(self.conv3(x))
        output_size.append(x.size())
        x, index = F.max_pool3d(x,kernel_size=(2, 2, 2), stride=(1,2,2),return_indices=True)
        indices.append(index)
        x = F.relu(self.conv4(x))
        output_size.append(x.size())
        x, index = F.max_pool3d(x,kernel_size=(2, 2, 2),stride=(2,2,2),return_indices=True)
        indices.append(index)
        x = F.relu(self.conv5(x))
        output_size.append(x.size())
        x, index = F.max_pool3d(x,kernel_size=(2, 2, 2), stride=(2,2,2), return_indices=True)
        indices.append(index)
        x = F.relu(self.conv6(x))
        output_size.append(x.size())
        x, index = F.max_pool3d(x,kernel_size=(2, 2, 2), stride=(2,2,2), return_indices=True)
        indices.append(index)
        x = self.conv7(x)
        # output_size.append(x.size())
        # x, index = F.max_pool3d(x,kernel_size=(2, 2, 2), stride=(2,2,2), return_indices=True)
        # indices.append(index)
        return x, indices, output_size

    def decoder(self, x, indices, output_size):
        # x = F.max_unpool3d(x, indices=indices.pop(-1),kernel_size=(2, 2, 2), stride=(2,2,2), output_size=output_size.pop(-1))
        x = self.convt1(x)
        x = F.max_unpool3d(x, indices=indices.pop(-1),kernel_size=(2, 2, 2), stride=(2,2,2), output_size=output_size.pop(-1))
        x = F.relu(self.convt2(x))
        x = F.max_unpool3d(x, indices=indices.pop(-1),kernel_size=(2, 2, 2), stride=(2,2,2), output_size=output_size.pop(-1))
        x = F.relu(self.convt3(x))
        x = F.max_unpool3d(x, indices=indices.pop(-1),kernel_size=(2, 2, 2), stride=(2,2,2), output_size=output_size.pop(-1))
        x = F.relu(self.convt4(x))
        x = F.max_unpool3d(x, indices=indices.pop(-1), kernel_size=(2, 2, 2), stride=(1,2,2), output_size=output_size.pop(-1))
        x = F.relu(self.convt5(x))
        x = F.max_unpool3d(x, indices=indices.pop(-1), kernel_size=(2, 2, 2), stride=(1,2,2), output_size=output_size.pop(-1))
        x = F.relu(self.convt6(x))
        x = F.max_unpool3d(x, indices=indices.pop(-1), kernel_size=(2, 2, 2), stride=(1,2,2), output_size=output_size.pop(-1))
        x = F.relu(self.convt7(x))
        return x

    def forward(self, x):
        z, indices, output_size = self.encoder(x)
        x = self.decoder(z, indices, output_size)
        return x

class BN_Autoencoder(Autoencoder):
    def __init__(self,color_channels=1):
        super(BN_Autoencoder,self).__init__(color_channels)
        self.conv2 = nn.Sequential(self.conv2,nn.BatchNorm3d(self.conv2.out_channels))
        self.conv3 = nn.Sequential(self.conv3, nn.BatchNorm3d(self.conv3.out_channels))
        self.conv4 = nn.Sequential(self.conv4,nn.BatchNorm3d(self.conv4.out_channels))
        self.conv5 = nn.Sequential(self.conv5, nn.BatchNorm3d(self.conv5.out_channels))
        self.convt5 = nn.Sequential(self.convt5, nn.BatchNorm3d(self.convt5.out_channels))
        self.convt4 = nn.Sequential(self.convt4, nn.BatchNorm3d(self.convt4.out_channels))
        self.convt3 = nn.Sequential(self.convt3, nn.BatchNorm3d(self.convt3.out_channels))
        self.convt2 = nn.Sequential(self.convt2, nn.BatchNorm3d(self.convt2.out_channels))


class GANomaly(Autoencoder):
    def __init__(self,color_channels=1):
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

class BN_GANomaly(BN_Autoencoder):
    def __init__(self,color_channels=1):
        super(BN_GANomaly, self).__init__(color_channels)
        self.conv6 = nn.Conv3d(color_channels, 8, 3)
        self.conv7 = nn.Sequential(nn.Conv3d(8, 16, 3), nn.BatchNorm3d(16))
        self.conv8 = nn.Sequential(nn.Conv3d(16, 32, 3), nn.BatchNorm3d(32))
        self.conv9 = nn.Sequential(nn.Conv3d(32, 64, 3), nn.BatchNorm3d(64))
        self.conv10 = nn.Sequential(nn.Conv3d(64, 128, 3), nn.BatchNorm3d(128))

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
        return img_out, z_in,  z_out


class AutoencoderB(nn.Module):
    """ Following the basic C3D implementation from:
    https://github.com/okankop/Efficient-3DCNNs/blob/master/models/c3d.py
    """
    def __init__(self,color_channels):
        super(AutoencoderB, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(color_channels,64,kernel_size=3,padding=1),
                                   nn.ReLU())
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(1,2,2),return_indices=True)
        self.conv2 = nn.Sequential(nn.Conv3d(64,128,kernel_size=3,padding=1),
                                   nn.ReLU())
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2),return_indices=True)
        self.conv3 = nn.Sequential(nn.Conv3d(128,256,kernel_size=3,padding=1),
                                   nn.ReLU(),
                                   nn.Conv3d(256,256,kernel_size=3,padding=1),
                                   nn.ReLU())
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2),return_indices=True)
        self.conv4 = nn.Sequential(nn.Conv3d(256,256,kernel_size=3,padding=1),
                                   nn.ReLU(),
                                   nn.Conv3d(256,256,kernel_size=3,padding=1),
                                   nn.ReLU())
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2),return_indices=True)
        self.conv5 = nn.Sequential(nn.Conv3d(256,256,kernel_size=3,padding=1),
                                   nn.ReLU(),
                                   nn.Conv3d(256,256,kernel_size=3, padding=1),
                                   nn.ReLU())
        self.maxpool5 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(2,2,2),padding=(0,1,1), return_indices=True)
        self.unpool1 = nn.MaxUnpool3d(kernel_size =(1,2,2),stride=(2,2,2),padding=(0,1,1))
        self.convt0 = nn.Sequential(nn.ConvTranspose3d(256,256,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.ConvTranspose3d(256,256,kernel_size=3, padding=1),
                                    nn.ReLU())
        self.unpool0 = nn.MaxUnpool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.convt1 = nn.Sequential(nn.ConvTranspose3d(256, 256,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.ConvTranspose3d(256,256,kernel_size=3,padding=1),
                                    nn.ReLU())
        self.unpool2 = nn.MaxUnpool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.convt2 = nn.Sequential(nn.ConvTranspose3d(256,256,kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.ConvTranspose3d(256,128,kernel_size=3, padding=1),
                                    nn.ReLU())
        self.unpool3 = nn.MaxUnpool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.convt3 = nn.Sequential(nn.ConvTranspose3d(128,64,kernel_size=3,padding=1),
                                    nn.ReLU())
        self.unpool4 = nn.MaxUnpool3d(kernel_size=(2,2,2),stride=(1,2,2))
        self.convt4 = nn.Sequential(nn.ConvTranspose3d(64,1,kernel_size=3,padding=1),
                                    nn.ReLU())

    def encoder(self,x):
        indices = []
        output_size = []
        convs = [self.conv1,self.conv2,self.conv3,self.conv4,self.conv5]
        pools = [self.maxpool1, self.maxpool2,self.maxpool3,self.maxpool4,self.maxpool5]
        for conv, pool in zip(convs,pools):
            x = conv(x)
            output_size.append(x.size())
            x, index = pool(x)
            indices.append(index)
        return x, indices, output_size

    def decoder(self,x,indices,output_size):
        convts = [self.convt0, self.convt1,self.convt2,self.convt3,self.convt4]
        unpools = [self.unpool1, self.unpool0, self.unpool2,self.unpool3,self.unpool4]
        for convt, unpool in zip(convts, unpools):
            x = unpool(x, indices=indices.pop(-1), output_size=output_size.pop(-1))
            x = F.relu(convt(x))
        return x

    def forward(self,x):
        x, indices, output_size = self.encoder(x)
        print(x.shape)
        x = self.decoder(x, indices, output_size)
        return x


