from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.initializers import HeNormal
import cupy as xp
import concurrent.futures

class Swish(Chain):
    def __init__(self, in_channels, out_channels, dilate):
        super(Swish, self).__init__()
        with self.init_scope():
            self.dilate = dilate
            if not dilate:
                self.conv = L.Convolution2D(in_channels, out_channels, ksize=(1, 3), pad=(0, 1))
            else:
                self.conv = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 3), dilate=(0, dilate), pad=(0, dilate))
    
    def __call__(self, x):
        h = self.conv(x)
        return h * F.sigmoid(h)
        

class ResBlock1(Chain):
    def __init__(self, in_channels, out_channels, resSize=8, executor=None):
        super(ResBlock1, self).__init__()
        with self.init_scope():
            for i in range(8):
                self.add_link(f"swish{i}", Swish(in_channels, out_channels // 8 , i))
            self.conv = L.Convolution2D(out_channels, out_channels, 1)
            self.out_channels = out_channels
            # self.dilate = dilate
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8) if executor == None else executor
            self.resSize = resSize
        
    def __call__(self, x):
        task = []
        for i in range(8):
            task.append(self.executor.submit(self[f"swish{i}"], x))
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
            # self.embedid = L.EmbedID(256, 32)
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
            for i in range(6):
                self.add_link(f"resBlock{i}", ResBlock1(64, 64, executor=self.executor))
            self.l0 = L.Linear(16,8)
            self.conv0 = L.ConvolutionND(1, 1, 64, 1)
            self.conv1 = L.ConvolutionND(1, 64, 64, 1)
            self.conv2 = L.ConvolutionND(1, 64, 8, 1)
            # self.w = xp.arange(256).astype(xp.float32).reshape(256,1)/255

    def __call__(self, x):
        # h = F.embed_id(x, self.w)
        # h = F.transpose(h, axes=(0,1,3,2)).reshape(x.shape[0],1,-1)
        h = x.reshape(x.shape[0],1,-1)
        h = F.reshape(F.relu(self.conv0(h)), (x.shape[0], 64, 1, -1))[:,:,:,:32768]
        for i in range(6):
            _, h=self[f"resBlock{i}"](h)
            h = F.max_pooling_2d(h, ksize=(1, 4))
        # print(h.shape)
        h = F.reshape(h, (x.shape[0], 64, -1))
        # print(h.shape)
        h = F.relu(self.conv1(h))
        h = self.conv2(h)
        h = F.average(h,axis=2)
        return h


class ResBlock(Chain):
    def __init__(self, in_channels, out_channels, z_channels, dilate, resSize=8):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.conv = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 2), dilate=(0, dilate))
            # self.convS = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 2), dilate=(0, dilate))
            # self.conv = L.Convolution2D(out_channels, out_channels, 1)
            self.linear = L.Linear(z_channels, out_channels)
            # self.linearS = L.Linear(z_channels, out_channels)
            self.out_channels = out_channels
            self.dilate = dilate
            self.resSize = resSize
        
    def __call__(self, x, z):
        z = F.tile(F.reshape(self.linear(z),(x.shape[0],self.out_channels,1,1)),(x.shape[3]-self.dilate))
        # zS = F.tile(F.reshape(self.linearS(z),(x.shape[0],self.out_channels,1,1)),(x.shape[3]-self.dilate))
        h = self.conv(x) + z
        h = h * F.sigmoid(h)
        # h = self.conv(h)
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
            for i in range(11):
                self.add_link(f"resBlock{i}", ResBlock(64, 64, 8, 2**(i+1)-1))
            self.l0 = L.Linear(16,16)
            self.conv0 = L.ConvolutionND(1, 1, 64, 2)
            self.conv1 = L.ConvolutionND(1, 104, 64, 1)
            self.conv2 = L.ConvolutionND(1, 64, 1, 1)
            self.l1 =L.Linear(16, 8)
            # self.embedid = L.EmbedID(256, 32)
            # self.conv0
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __call__(self, x, i, o, test=False):
        dataLen = x.shape[-1]-4084 if test else 32768
        # z = self.convBlock(i)-self.convBlock(o)
        z = (self.executor.submit(self.convBlock,i),self.executor.submit(self.convBlock,o))
        z = F.concat((z[0].result(), z[1].result()),axis=-1)
        z = F.relu(self.l0(z))
        z = F.relu(self.l1(z))
        # h = F.transpose(self.embedid(x),axes=(0,1,3,2)).reshape(x.shape[0],32,-1)
        h_ = F.reshape(self.conv0(x), (x.shape[0], 64, 1, -1))
        # _, h_ = self[f"resBlock0"](h, z)
        h = []
        # residual = t + h[:,:,:,:-1]
        for i in range(11):
            t, h_ = self[f"resBlock{i}"](h_, z)
            if i==10:
                h.append(t)
                break
            h.append(t[:,-4:,:,-dataLen:])
            # h = t + h[:,:,:,self[f"resBlock{i}"].dilate:]
            # t.append(h[:,-:,-32768:])
        h = F.concat(h)
        # print(h.shape)
        h = F.reshape(h, (x.shape[0], 104, -1))
        h = self.conv1(h)
        h = h * F.sigmoid(h)
        h = self.conv2(h)
        return h

    def test(self, x, i, o):
        # z = self.convBlock(i)-self.convBlock(o)
        z = (self.executor.submit(self.convBlock,i),self.executor.submit(self.convBlock,o))
        z = F.concat((z[0].result(), z[1].result()),axis=-1)
        z = F.relu(self.l0(z))
        z = F.relu(self.l1(z))
        # h = F.transpose(self.embedid(x),axes=(0,1,3,2)).reshape(x.shape[0],32,-1)
        h_ = F.reshape(self.conv0(x), (x.shape[0], 64, 1, -1))
        # _, h_ = self[f"resBlock0"](h, z)
        h = []
        # residual = t + h[:,:,:,:-1]
        for i in range(11):
            t, h_ = self[f"resBlock{i}"](h_, z)
            if i==10:
                h.append(t)
                break
            h.append(t[:,-4:,:,-32768:])
            # h = t + h[:,:,:,self[f"resBlock{i}"].dilate:]
            # t.append(h[:,-:,-32768:])
        h = F.concat(h)
        # print(h.shape)
        h = F.reshape(h, (x.shape[0], 104, -1))
        h = self.conv1(h)
        h = h * F.sigmoid(h)
        h = self.conv2(h)
        return h


class Discriminator(Chain):
    def __init__(self, compressor):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.convBlock=compressor
            self.l1=L.Linear(16, 16, initialW=HeNormal())
            self.l2=L.Linear(16, 16, initialW=HeNormal())
            self.l3=L.Linear(16, 16, initialW=HeNormal())
            self.l4=L.Linear(16, 2, initialW=HeNormal())
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __call__(self, x, c):
        h = (self.executor.submit(self.convBlock,x),self.executor.submit(self.convBlock,c))
        h = F.concat((h[0].result(), h[1].result()),axis=-1)
        # h = F.concat((self.convBlock(x),self.convBlock(c)),axis=-1)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = self.l4(h)
        return h

