from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.initializers import HeNormal
# import chainer.Variable as V


class ResBlock1(Chain):
    def __init__(self, in_channels, out_channels, z_channels, dilate):
        super(ResBlock1, self).__init__()
        with self.init_scope():
            self.convT = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 2), dilate=(0, dilate))
            self.convS = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 2), dilate=(0, dilate))
            self.conv = L.Convolution2D(out_channels, out_channels, 1)
            self.out_channels = out_channels
            self.dilate = dilate
        
    def __call__(self, x):
        h = F.tanh(self.convT(x)) * F.sigmoid(self.convS(x))
        h = self.conv(h)
        return h

class Compressor(Chain):
    def __init__(self):
        super(Compressor, self).__init__()
        with self.init_scope():
            self.resBlocks =[]
            self.embedid = L.EmbedID(256, 32)
            for i in range(9):
                x=ResBlock1(32, 32, 8, 2**(i+1)-1)
                x.to_gpu()
                self.resBlocks.append(x)
            self.l0 = L.Linear(16,8)
            self.conv0 = L.ConvolutionND(1, 32, 32, 2)
            self.conv1 = L.ConvolutionND(1, 32, 32, 1)
            self.conv2 = L.ConvolutionND(1, 32, 8, 1)

    def __call__(self, x):
        
        # x = x.reshape(x.shape[0], x.shape[1], 1, x.shape[2])
        # print(self.embedid(x).shape)
        h = F.transpose(self.embedid(x),axes=(0,1,3,2)).reshape(x.shape[0],32,-1)

        h = F.reshape(self.conv0(h), (x.shape[0], 32, 1, -1))
        t = self.resBlocks[0](h)
        residual = t + h[:,:,:,:-1]
        for c in self.resBlocks[1:]:
            t = c(residual)
            n=h.shape[-1]-t.shape[-1]
            h = t + h[:,:,:,:-n]
            residual = t + residual[:,:,:,:-c.dilate]
        h = F.reshape(h, (x.shape[0], 32, -1))
        h = F.relu(self.conv1(h))
        h = self.conv2(h)
        h = F.average(h,axis=2)
        return h


class ResBlock(Chain):
    def __init__(self, in_channels, out_channels, z_channels, dilate):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.convT = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 2), dilate=(0, dilate))
            self.convS = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 2), dilate=(0, dilate))
            self.conv = L.Convolution2D(out_channels, out_channels, 1)
            self.linearT = L.Linear(z_channels, out_channels)
            self.linearS = L.Linear(z_channels, out_channels)
            self.out_channels = out_channels
            self.dilate = dilate
        
    def __call__(self, x, z):
        zT = F.tile(F.reshape(self.linearT(z),(x.shape[0],self.out_channels,1,1)),(x.shape[3]-self.dilate))
        zS = F.tile(F.reshape(self.linearS(z),(x.shape[0],self.out_channels,1,1)),(x.shape[3]-self.dilate))
        h = F.tanh(self.convT(x) + zT) * F.sigmoid(self.convS(x) + zS)
        h = self.conv(h)
        return h



class Generator(Chain):
    def __init__(self, compressor):
        super(Generator, self).__init__()
        with self.init_scope():
            self.convBlock=compressor
            self.resBlocks =[]
            for i in range(8):
                x=ResBlock(32, 32, 8, 2**(i+1)-1)
                x.to_gpu()
                self.resBlocks.append(x)
            self.l0 = L.Linear(16,8)
            self.conv0 = L.ConvolutionND(1, 32, 32, 2)
            self.conv1 = L.ConvolutionND(1, 32, 64, 1)
            self.conv2 = L.ConvolutionND(1, 64, 256, 1)
            self.embedid = L.EmbedID(256, 32)

    def __call__(self, x, i, o):
        # x = x.reshape(x.shape[0], x.shape[1], 1, x.shape[2])
        z = self.convBlock(i)-self.convBlock(o)
        h = F.transpose(self.embedid(x),axes=(0,1,3,2)).reshape(x.shape[0],32,-1)
        h = F.reshape(self.conv0(h), (x.shape[0], 32, 1, -1))
        t = self.resBlocks[0](h, z)
        residual = t + h[:,:,:,:-1]
        for c in self.resBlocks[1:]:
            t = c(residual, z)
            n=h.shape[-1]-t.shape[-1]
            h = t + h[:,:,:,:-n]
            residual = t + residual[:,:,:,:-c.dilate]
        h = F.reshape(h, (x.shape[0], 32, -1))
        h = F.relu(self.conv1(h))
        h = self.conv2(h)
        return h


class Discriminator(Chain):
    def __init__(self, compressor):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.convBlock=compressor
            self.l1=L.Linear(8, 8, initialW=HeNormal())
            self.l2=L.Linear(8, 4, initialW=HeNormal())
            self.l3=L.Linear(4, 2, initialW=HeNormal())

    def __call__(self, x, c):
        h = self.convBlock(x)-self.convBlock(c)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h

