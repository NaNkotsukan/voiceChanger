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
            # self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
            # for i in range(6):
            #     self.add_link(f"resBlock{i}", ResBlock1(64, 64, executor=self.executor))
            self.l0 = L.Linear(64,8)

            self.conv0 = L.Convolution2D(2, 64, 3)
            self.conv1 = L.Convolution2D(64, 64, 3)
            self.conv2 = L.Convolution2D(64, 64, 3)
            self.conv3 = L.Convolution2D(64, 64, 3)
            self.conv4 = L.Convolution2D(64, 64, 3)
            self.conv5 = L.Convolution2D(64, 64, 3)
            self.conv6 = L.Convolution2D(64, 64, 3)
            self.conv7 = L.Convolution2D(64, 64, 3)
            self.conv8 = L.Convolution2D(64, 64, 3)
            self.conv9 = L.Convolution2D(64, 64, 3)

            self.bn0 = L.BatchNormalization(64)
            self.bn1 = L.BatchNormalization(64)
            self.bn2 = L.BatchNormalization(64)
            self.bn3 = L.BatchNormalization(64)
            self.bn4 = L.BatchNormalization(64)

            # self.conv1 = L.ConvolutionND(1, 64, 64, 1)
            # self.conv2 = L.ConvolutionND(1, 64, 8, 1)
            # self.w = xp.arange(256).astype(xp.float32).reshape(256,1)/255

    def __call__(self, x):
        # h = F.embed_id(x, self.w)
        # h = F.transpose(h, axes=(0,1,3,2)).reshape(x.shape[0],1,-1)

        h = self.bn0(self.conv0(x))
        h = h * F.sigmoid(h)
        h = F.max_pooling_2d(h, 2)

        h = self.conv1(h)
        h = h * F.sigmoid(h)
        h = self.bn1(self.conv2(h))
        h = h * F.sigmoid(h)
        h = F.max_pooling_2d(h, 2)

        h = self.conv3(h)
        h = h * F.sigmoid(h)
        h = self.bn2(self.conv4(h))
        h = h * F.sigmoid(h)
        h = F.max_pooling_2d(h, 2)

        h = self.conv5(h)
        h = h * F.sigmoid(h)
        h = self.bn3(self.conv6(h))
        h = h * F.sigmoid(h)
        h = F.max_pooling_2d(h, 2)

        h = self.conv7(h)
        h = h * F.sigmoid(h)
        h = self.bn4(self.conv8(h))
        h = h * F.sigmoid(h)
        h = F.max_pooling_2d(h, 2)

        h = self.conv9(h)
        h = h * F.sigmoid(h)
        
        h = self.bn2(self.conv4(h))
        h = h * F.sigmoid(h)
        h = F.max_pooling_2d(h, 2)

        h = F.average(h, axis=(2,3))
        h = self.l0(h)

        return h


class dilatedBlock(Chain):
    def __init__(self, in_channels, out_channels, dilateV, dilateH):
        super(dilatedBlock, self).__init__()
        with self.init_scope():
            self.dilateH = dilateH
            self.dilateV = dilateV
            if dilateH == dilateV == 0:
                self.conv = L.Convolution2D(in_channels, out_channels, ksize=(3, 2), pad=(1, 0))
            else:
                self.conv = L.DilatedConvolution2D(in_channels, out_channels, ksize=(3, 2), dilate=(dilateV, dilateH), pad=(dilateV, 0))
    
    def __call__(self, x):
        return self.conv(x)


class ResBlock(Chain):
    # def __init__(self, in_channels, out_channels, z_channels, dilate, resSize=8):
    def __init__(self, in_channels, out_channels, dilation, dilate=8, resSize=8, executor=None):
        super(ResBlock, self).__init__()
        with self.init_scope():
            for i in range(dilate):
                self.add_link(f"dilatedBlock{i}", dilatedBlock(in_channels, out_channels // dilate , i, dilation))
            self.conv = L.Convolution2D(out_channels, out_channels, 1)
            self.out_channels = out_channels
            # self.dilate = dilate
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4) if executor == None else executor
            self.resSize = resSize
            self.dilate = dilate
        
    def __call__(self, x):
        task = []
        for i in range(self.dilate):
            task.append(self.executor.submit(self[f"dilatedBlock{i}"], x))
        h = [p.result() for p in task]
        h = F.concat(h, axis=1)
        # print(h.shape)
        h = h * F.sigmoid(h)
        h = self.conv(h)
        h = h * F.sigmoid(h)
        a, b, c, d = h.shape
        # print(h.shape)
        # print(x.shape)
        residual = h + F.concat((x[:,:self.resSize,:,-d:], xp.zeros((a, b - self.resSize, c, d),dtype=xp.float32)),axis=1)
        # h = h * F.sigmoid(x)
        return residual


class Generator(Chain):
    def __init__(self, compressor):
        super(Generator, self).__init__()
        with self.init_scope():
            self.convBlock=compressor
            self.add_link(f"resBlock0", ResBlock(8, 64, 1))
            for i in range(1, 3):
                self.add_link(f"resBlock{i}", ResBlock(64, 64, 2**(i+1)-1))
            self.l0 = L.Linear(16, 8)
            self.conv0 = L.Convolution2D(2, 8, (1, 2))
            self.conv1 = L.Convolution2D(64, 64, 1)
            self.conv2 = L.Convolution2D(64, 2, 1)
            self.l1 =L.Linear(16, 8)
            # self.embedid = L.EmbedID(256, 32)
            # self.conv0
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __call__(self, x, i, o, test=False):
        # dataLen = x.shape[-1]-4084 if test else 
        # z = self.convBlock(i)-self.convBlock(o)
        z = (self.executor.submit(self.convBlock,i),self.executor.submit(self.convBlock,o))
        z = F.concat((z[0].result(), z[1].result()),axis=-1)
        # z = z * F.sigmoid(z)
        z = self.l0(z)
        z = F.tile(F.reshape(z,(x.shape[0],8,1,1)),(442, x.shape[3]-1))
        # print(self.conv0(x).shape)
        # print(x.shape)
        h = self.conv0(x) + z
        # h = F.reshape(h, (x.shape[0], 64, 442, -1)) + z

        for i in range(3):
            h = self[f"resBlock{i}"](h)
        
        # h = F.reshape(h, (x.shape[0], -1, 442))

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
            self.l2=L.Linear(16, 2, initialW=HeNormal())
            # self.l3=L.Linear(16, 16, initialW=HeNormal())
            # self.l4=L.Linear(16, 2, initialW=HeNormal())
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __call__(self, x, c):
        h = (self.executor.submit(self.convBlock,x),self.executor.submit(self.convBlock,c))
        h = F.concat((h[0].result(), h[1].result()),axis=-1)
        # h = F.concat((self.convBlock(x),self.convBlock(c)),axis=-1)
        h = F.relu(self.l1(h))
        # h = F.relu(self.l2(h))
        # h = F.relu(self.l3(h))
        h = self.l2(h)
        return h

