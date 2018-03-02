from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.initializers import HeNormal
import cupy as xp
import concurrent.futures

# import chainer.Variable as V
class GAU(Chain):
    def __init__(self, in_channels, out_channels, dilate):
        super(GAU, self).__init__()
        with self.init_scope():
            self.dilate = dilate
            if not dilate:
                self.convT = L.Convolution2D(in_channels, out_channels, ksize=(1, 3), pad=(0, 1))
                self.convS = L.Convolution2D(in_channels, out_channels, ksize=(1, 3), pad=(0, 1))
            else:
                self.convT = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 3), dilate=(0, dilate), pad=(0, dilate))
                self.convS = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 3), dilate=(0, dilate), pad=(0, dilate))
    
    def __call__(self, x):
        return F.tanh(self.convT(x)) * F.sigmoid(self.convS(x))
        

class ResBlock1(Chain):
    def __init__(self, in_channels, out_channels, resSize=8):
        super(ResBlock1, self).__init__()
        with self.init_scope():
            for i in range(8):
                self.add_link(f"gau{i}", GAU(in_channels, out_channels //8 , i))
            self.conv = L.Convolution2D(out_channels, out_channels, 1)
            self.out_channels = out_channels
            # self.dilate = dilate
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
            self.resSize = resSize
        
    def __call__(self, x):
        task = []
        for i in range(8):
            task.append(self.executor.submit(self[f"gau{i}"], x))
        h = [p.result() for p in task]
        h = F.concat(h, axis=1)
        h = self.conv(h)
        a, b, c, d = x.shape
        residual = h + F.concat((x[:,:self.resSize], xp.zeros((a, b - self.resSize, 1, d),dtype=xp.float32)),axis=1)
        return h, residual


class Compressor(Chain):
    def __init__(self):
        super(Compressor, self).__init__()
        with self.init_scope():
            self.embedid = L.EmbedID(256, 32)
            for i in range(4):
                self.add_link(f"resBlock{i}", ResBlock1(32, 32))
            self.l0 = L.Linear(16,8)
            self.conv0 = L.ConvolutionND(1, 1, 32, 1)
            self.conv1 = L.ConvolutionND(1, 32, 32, 1)
            self.conv2 = L.ConvolutionND(1, 32, 8, 1)
        self.w = xp.arange(256).astype(xp.float32).reshape(256,1)/255

    def __call__(self, x):
        h = F.embed_id(x, self.w)
        h = F.transpose(h, axes=(0,1,3,2)).reshape(x.shape[0],1,-1)
        
        h = F.reshape(F.relu(self.conv0(h)), (x.shape[0], 32, 1, -1))[:,:,:,:32768]
        for i in range(4):
            _, h=self[f"resBlock{i}"](h)
            h = F.max_pooling_2d(h, ksize=(1, 2))
        h = F.reshape(h, (x.shape[0], 32, -1))
        h = F.relu(self.conv1(h))
        h = self.conv2(h)
        h = F.average(h,axis=2)
        return h


class ResBlock(Chain):
    def __init__(self, in_channels, out_channels, z_channels, dilate, resSize=8):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.convT = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 2), dilate=(0, dilate))
            self.convS = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 2), dilate=(0, dilate))
            self.conv = L.Convolution2D(out_channels, out_channels, 1)
            self.linearT = L.Linear(z_channels, out_channels)
            self.linearS = L.Linear(z_channels, out_channels)
            self.out_channels = out_channels
            self.dilate = dilate
            self.resSize = resSize
        
    def __call__(self, x, z):
        zT = F.tile(F.reshape(self.linearT(z),(x.shape[0],self.out_channels,1,1)),(x.shape[3]-self.dilate))
        zS = F.tile(F.reshape(self.linearS(z),(x.shape[0],self.out_channels,1,1)),(x.shape[3]-self.dilate))
        h = F.tanh(self.convT(x) + zT) * F.sigmoid(self.convS(x) + zS)
        h = self.conv(h)
        a, b, c, d = x.shape
        # print(x[:self.resSize,:,:,self.dilate:].shape)
        # print(b - self.resSize)
        residual = h + F.concat((x[:,:self.resSize,:,self.dilate:],xp.zeros((a, b - self.resSize, 1, d - self.dilate),dtype=xp.float32)),axis=1)
        # residual[:self.resSize,:,:,:] += x
        return h, residual



class Generator(Chain):
    def __init__(self, compressor):
        super(Generator, self).__init__()
        with self.init_scope():
            self.convBlock=compressor
            self.resBlocks =[]
            for i in range(12):
                x=ResBlock(32, 32, 8, 2**(i+1)-1)
                x.to_gpu()
                self.resBlocks.append(x)
            self.l0 = L.Linear(16,8)
            self.conv0 = L.ConvolutionND(1, 1, 32, 2)
            self.conv1 = L.ConvolutionND(1, 32, 64, 1)
            self.conv2 = L.ConvolutionND(1, 64, 256, 1)
            # self.embedid = L.EmbedID(256, 32)
            # self.conv0

    def __call__(self, x, i, o):
        z = self.convBlock(i)-self.convBlock(o)
        # h = F.transpose(self.embedid(x),axes=(0,1,3,2)).reshape(x.shape[0],32,-1)
        h = F.reshape(self.conv0(x), (x.shape[0], 32, 1, -1))
        h, h_ = self.resBlocks[0](h, z)
        # residual = t + h[:,:,:,:-1]
        for c in self.resBlocks[1:]:
            t, h_ = c(h_, z)
            # n=h.shape[-1]-t.shape[-1]
            # print(n)
            # print(c.dilate)
            h = t + h[:,:,:,c.dilate:]
            # residual = residual[:,:,:,:-c.dilate]
            # residual += t
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

