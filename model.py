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
    
class GAU(Chain):
    def __init__(self, in_channels, out_channels, dilate):
        super(GAU, self).__init__()
        with self.init_scope():
            self.dilate = dilate
            # if not dilate:
            #     self.convT = L.Convolution2D(in_channels, out_channels, ksize=(1, 2), pad=(0, 1))
            #     self.convS = L.Convolution2D(in_channels, out_channels, ksize=(1, 2), pad=(0, 1))
            # else:
            self.convT = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 2), dilate=(0, dilate))
            self.convS = L.DilatedConvolution2D(in_channels, out_channels, ksize=(1, 2), dilate=(0, dilate))
            self.conv = L.Convolution2D(out_channels, out_channels, ksize=1)
    
    def __call__(self, x):
        h = F.tanh(self.convT(x)) * F.sigmoid(self.convS(x))
        # print(h.shape)
        h = self.conv(h)
        return h

class ResBlock(Chain):
    # def __init__(self, in_channels, out_channels, z_channels, dilate, resSize=8):
    def __init__(self, in_channels, out_channels, dilate, resSize=8, executor=None):
        super(ResBlock, self).__init__()
        with self.init_scope():
            # self.conv0 = L.DilatedConvolution2D(in_channels, out_channels, ksize=(3, 2), dilate=(dilate, 1), pad=(dilate, 0))
            # self.conv1 = L.Convolution2D(out_channels, out_channels, ksize=(1, 1), pad=(0, 0))
            self.gau=GAU(in_channels, out_channels, dilate)
            self.out_channels = out_channels
            # self.dilate = dilate
            self.resSize = resSize
            self.dilate = dilate
        
    def __call__(self, x):

        # h = h * F.sigmoid(h)
        h = self.gau(x)
        a, b, c, d = h.shape
        residual = F.concat([x[:,:self.resSize,:,-d:]+h[:,:self.resSize], h[:,self.resSize:]])
        #  h + F.concat((x[:,:self.resSize,:,-d:], xp.zeros((a, b - self.resSize, c, d),dtype=xp.float32)),axis=1)
        return residual

class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(1, 256, ksize=(1, 2))
            for i in range(10):
                self.add_link(f"resBlock{i}", ResBlock(256, 256, 2**(i+1)))
            # self.l0 = L.Linear(128, 64)
            self.conv1 = L.Convolution2D(256, 256, 1)
            self.conv2 = L.Convolution2D(256, 1, 1)
            self.id = L.EmbedID(108, 8, initialW=HeNormal())
            # self.l1 =L.Linear(16, 8)
            # self.embedid = L.EmbedID(256, 32)
            # self.conv0
            # self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __call__(self, x, c, test=False):
        # print(x.reshape(len(x), 1, 1, -1))
        # print(x.shape)
        # print(x.dtype)
        # print(type(x))
        z = F.concat((self.id(c), xp.zeros((x.shape[0], 248), dtype=xp.float32)))
        
        h = self.conv0(x.reshape(len(x), 1, 1, -1))
        h = h + F.tile(z.reshape(x.shape[0], -1, 1, 1), h.shape[-1])

        for i in range(10):
            h = self[f"resBlock{i}"](h)
        
        # h = F.reshape(h, (x.shape[0], -1, 442))
        h = F.relu(h)
        h = self.conv1(h)
        # h = h * F.sigmoid(h)
        h = F.relu(h)
        h = self.conv2(h)
        # print("=-----------------")
        return h

class Conv(Chain):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels,  ksize=(1, 4), stride=(1, 2), pad=(0, 1))
            self.bn = L.BatchNormalization(in_channels)

    def __call__(self, x):
        # print(x.dtype)
        h = self.bn(x)
        h = h * F.sigmoid(h)
        h = F.dropout(h, 0.3)
        h = self.conv(h)
        return h

# class xBlock(Chain):
#     def __init__(self, in_channels, out_channels):
#         super(xBlock, self).__init__()
#         with self.init_scope():
#             for i in range(8):
#                 self.add_link(f"dic{i}", L.DilatedConvolution2D(in_channels, out_channels/8 ,ksize=(1, 3), pad=(0, i), dilate=(0, i)))
#             self.bn = L.BatchNormalization(in_channels)
#         self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    
#     def __call__(self, x):
#         for i in range(8):
#             self.executor.submit(self[f'dic{i}'], x)
        


# class Conv2(Chain):
#     def __init__(self, in_channels, out_channels):
#         super(Conv2, self).__init__()
#         with self.init_scope():
#             L.Inception()   
#             self.conv0 = L.Convolution2D(in_channels, out_channels, ksize=(1, 8))
#             self.bn0 = L.BatchNormalization(in_channels)
#             self.bn1 = L.BatchNormalization(in_channels)

#     def __call__(self, x):
#         # print(x.dtype)
#         h = self.bn(x)
#         h = h * F.sigmoid(h)
#         h = F.dropout(h, 0.3)
#         h = self.conv(h)
#         h = self.bn(x)
#         h = h * F.sigmoid(h)
#         h = F.dropout(h, 0.3)
#         h = self.conv(h)
#         return h

class Discriminator(Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        with self.init_scope():
            # self.convBlock=compressor
            self.conv = L.Convolution2D(1, 32, ksize=(1, 4), stride=2, pad=(0, 1))
            for i in range(8):
                self.add_link(f"conv{i}", Conv(32, 32))
            
            self.add_link(f"conv0_", Conv(32, 256))
            for i in range(1, 8):
                self.add_link(f"conv{i}_", Conv(256, 256))
            self.l1=L.Linear(64, 64, initialW=HeNormal())
            self.l2=L.Linear(64, 64, initialW=HeNormal())
            self.l3=L.Linear(64, 2)
            self.l4=L.Linear(512, 512, initialW=HeNormal())
            self.l5=L.Linear(512, 256, initialW=HeNormal())
            self.l6=L.Linear(256, 111, initialW=HeNormal())
            # self.l4=L.Linear(16, 2, initialW=HeNormal())
            # self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __call__(self, x):
        # print(x.shape)
        h = self.conv(x.reshape(len(x), 1, 1, -1))
        h_ = h
        # for i in range(8):
            # h = self[f"conv{i}"](h)

        for i in range(8):
            h_ = self[f"conv{i}_"](h_)
            # # print(h.shape)
        # # h = F.concat((self.convBlock(x),self.convBlock(c)),axis=-1)

        # h = F.relu(h)
        # h = F.relu(self.l1(h))
        # h = F.relu(self.l2(h))
        # h = F.relu(self.l3(h))
        # tf = self.l3(h)
        tf = 0

        h = F.relu(h_)
        h = F.relu(self.l4(h))
        h = F.relu(self.l5(h))
        c = self.l6(h)
        return tf, c

