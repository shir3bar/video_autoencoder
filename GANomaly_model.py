import torch.nn.functional as F
import torch.nn as nn
### An vanilla implementation of Ackay et al's GANomaly model to 3d convolutions
class GAN_Autoencoder(nn.Module):
    def __init__(self,color_channels=1,initial_filters=32,nz=2048):
        super(GAN_Autoencoder, self).__init__()
        main = nn.Sequential()

        self.conv1 = nn.Conv3d(color_channels,initial_filters, kernel_size=4,stride=(1,2,2),padding=1)
        self.conv2 = nn.Conv3d(initial_filters, initial_filters*2, kernel_size=4, stride=(1,2,2),padding=1)
        self.conv3 = nn.Conv3d(initial_filters*2,initial_filters*2**2, kernel_size=4, stride=(1,2,2), padding=1)
        self.conv4 = nn.Conv3d(initial_filters*2**2,initial_filters*2**3,kernel_size=4,stride=2,padding=1)
        self.conv5 = nn.Conv3d(initial_filters*2**3, initial_filters*2**4,kernel_size=4,stride=2,padding=1)
        self.conv6 = nn.Conv3d(initial_filters*2**4, initial_filters*2**5, kernel_size=4, stride=2, padding=1)
        self.conv7= nn.Conv3d(initial_filters*2**5,nz,kernel_size=4,stride=1,padding=0)
        self.convt1 = nn.ConvTranspose3d(nz, initial_filters*2**5, kernel_size=4,stride=1,padding=0)
        self.convt2 = nn.ConvTranspose3d(initial_filters*2**5, initial_filters*2**4,kernel_size=4,stride=2,padding=1)
        self.convt3 = nn.ConvTranspose3d(initial_filters*2**4, initial_filters*2**3, kernel_size=4, stride=2, padding=1)
        self.convt4 = nn.ConvTranspose3d(initial_filters*2**3,initial_filters*2**2, kernel_size=4, stride=2, padding=1)
        self.convt5 = nn.ConvTranspose3d(initial_filters*2**2, initial_filters*2, kernel_size=4, stride=(1,2,2), padding=1)
        self.convt6 = nn.ConvTranspose3d(initial_filters*2, initial_filters, kernel_size=4, stride=(1,2,2), padding=1)
        self.convt7 = nn.ConvTranspose3d(initial_filters, color_channels, kernel_size=4, stride=(1,2,2), padding=1)

    def encoder(self,x):
        convs = [self.conv1,self.conv2,self.conv3,self.conv4,self.conv5, self.conv6]#,self.conv7]
        for conv in convs:
            x = conv(x)
            x = F.leaky_relu(x,0.2,inplace=True)
        x = self.conv7(x)
        return x

    def decoder(self,x):
        convts = [self.convt1,self.convt2,self.convt3,self.convt4,self.convt5, self.convt6, self.convt7]
        for convt in convts:
            x = convt(x)
            x = F.relu(x,inplace=True)
        return x

    def forward(self,x):
        z_in = self.encoder(x)
        img_out = self.decoder(z_in)
        return img_out


class Encoder(nn.Module):
    def __init__(self,color_channels=1, image_size=256, num_frames=74,ndf=64, nz=256, batchnorm=True, DEPTH=4):
        super(Encoder, self).__init__()
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"
        main = nn.Sequential()
        main.add_module(f'initial_conv{color_channels}-{ndf}',
                        nn.Conv3d(color_channels, ndf, kernel_size=4, stride=(1,2,2), padding=1, bias=False))
        main.add_module(f'initial-relu{ndf}',
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = image_size/2, ndf

        while csize > DEPTH:
            in_feat = cndf
            out_feat = cndf*2
            if csize > image_size/6 and num_frames<129:
                # see note below
                stride = (1,2,2)
            else:
                stride = 2
            main.add_module(f'pyramid-{in_feat}-{out_feat}-conv',
                            nn.Conv3d(in_feat,out_feat,kernel_size=4,stride=stride,padding=1,bias=False))
            if batchnorm:
                main.add_module(f'pyramid-{out_feat}-batchnorm',
                                nn.BatchNorm3d(out_feat))
            main.add_module(f'pyramid{out_feat}-relu',
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf*2
            csize = csize/2
        # with the original setting of kernel size=4, stride=2, padding=1, for a 70x256x256 clip size:
        # after initial conv: channels (1,64) imsize 35x128x128 | if 129 frms: 128x128x128
        # after second conv (64,128) imsize 17x64x64 | 64x64x64
        # after third conv (128,256) imsize 8x32x32 | 32x32x32
        # after fourth conv (256, 512) imsize 4x16x16  | 13X16x16
        # after fifth conv (512, 1024) imsize 2x8x8 | 8X8X8
        # after sixth conv  (1024, 2048) imsize 1x4x4 | 4x4x4
        # since we need our data to be 4x4x4 to apply to final conv of (2048,512) with kernel=4 (stride=1,padding=0)
        # I changed the stride of the first two convolutions to (1,2,2) instead of (2,2,2), so that input at the end
        # of the sixth convolution is 2048x4x4x4
        main.add_module(f'final-{cndf}-{nz}-conv',
                        nn.Conv3d(cndf,nz, kernel_size=4, stride=1,padding=0,bias=False))
        self.main = main

    def forward(self,x):
        return self.main(x)


class Decoder(nn.Module):

    def __init__(self,color_channels=1, image_size=256, num_frames=74, ngf=64, nz=256, batchnorm=True, DEPTH=4):
        super(Decoder, self).__init__()
        assert image_size % 16 == 0, "image_size has to be a multiple of 16"

        # get the number of filter at the first layer after the bottleneck:
        cngf, tisize = ngf // 2, DEPTH
        while tisize != image_size:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module(f'initial-{nz}-{cngf}',
                        nn.ConvTranspose3d(nz, cngf, kernel_size=4, stride=1, padding=0, bias=False))
        if batchnorm:
           main.add_module(f'initial-{cngf}-batchnorm',
                            nn.BatchNorm3d(cngf))
        main.add_module(f'initial-{cngf}-relu',
                       nn.ReLU(True))
        csize = DEPTH
        while csize < image_size//2:
            if csize*2 > image_size//8 and num_frames<129:
                stride = (1,2,2)
            else:
                stride = 2
            main.add_module(f'pyramid-{cngf}-{cngf//2}-convt',
                            nn.ConvTranspose3d(cngf,cngf//2,kernel_size=4,stride=stride,padding=1,bias=False))
            if batchnorm:
                main.add_module(f'pyramid-{cngf//2}-batchnorm',
                                nn.BatchNorm3d(cngf//2))
            main.add_module(f'pyramind-{cngf//2}-relu',
                            nn.ReLU(True))
            cngf = cngf//2
            csize = csize*2

        main.add_module(f'final-{cngf}-{color_channels}-convt',
                        nn.ConvTranspose3d(cngf,color_channels,kernel_size=4,stride=(1,2,2),padding=1,bias=False))
        main.add_module(f'final-{color_channels}-tanh',
                        nn.Tanh())
        self.main = main

    def forward(self,x):
        return self.main(x)


class Net(nn.Module):
    """ What is referred to as a generator network in the original repository"""

    def __init__(self, color_channels=1, image_size=256, num_frames=129, initial_filters=32, nz=4096, batchnorm=True):
        super(Net, self).__init__()
        DEPTH = 4
        self.encoder1 = Encoder(color_channels,image_size,num_frames,initial_filters,nz,batchnorm,DEPTH)
        self.decoder = Decoder(color_channels,image_size,num_frames,initial_filters,nz,batchnorm,DEPTH)
        #self.encoder2 = Encoder(color_channels,image_size,num_frames,initial_filters,nz,batchnorm, DEPTH)

    def forward(self,x):
        z_in = self.encoder1(x)
        recon = self.decoder(z_in)
        #z_out = self.encoder2(recon)
        return recon #, z_in,  z_out