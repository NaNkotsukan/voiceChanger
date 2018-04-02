import chainer.functions as F
import chainer.links as L
from chainer import Variable as V
from chainer import optimizers, optimizer
from chainer.serializers import save_npz, load_npz
import numpy as np
import cupy as xp
from model import Generator, Discriminator
from data import dataset
import time
import pickle

import copy

class Train:
    def __init__(self):
        self.data=dataset()
        self.data.reset()
        self.reset()
        # self.load(1)
        self.setLR()
        self.time=time.time()
        self.dataRate = xp.float32(0.8)
        self.training(batchsize = 128)


    def reset(self):
        self.generator = None
        self.discriminator = None
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator.to_gpu()
        self.discriminator.to_gpu()

    def setLR(self, lr=0.002):
        self.gen_opt = optimizers.Adam(alpha=lr)
        self.gen_opt.setup(self.generator)
        self.gen_opt.add_hook(optimizer.WeightDecay(0.0001))
        self.dis_opt = optimizers.Adam()
        self.dis_opt.setup(self.discriminator)
        self.dis_opt.add_hook(optimizer.WeightDecay(0.0001))

    def training(self, batchsize = 1):
        for x in range(100):
            N = self.data.reset()
            for i in range(N // batchsize-1):
                res = self.batch(batchsize = batchsize)
                if not i%100:
                    print(F"{i} time:{int(time.time()-self.time)} D_Loss:{res[0]} D_Acc:{res[1]}")
                    if not i%1000:
                        save_npz(f"param/disClass__/dis_{i}.npz",self.discriminator)

    def batch(self, batchsize = 2):
        x, c = self.data.next(batchSize = batchsize,  dataSize=[1024], dataSelect=[0])
        x = x[0].reshape(batchsize, 1, 1, -1)
        # print(xp.abs(x).max(axis=-1).shape)
        x /= ((xp.random.rand(batchsize)*0.5+0.5) / xp.abs(x).max(axis=-1).reshape(-1)).reshape(-1,1,1,1)

        c=xp.asarray(c[0])

        T_tf, T_c = self.discriminator(x)
        L_dis = F.softmax_cross_entropy(T_c, c)

        self.discriminator.cleargrads()
        L_dis.backward()
        self.dis_opt.update()

        return (L_dis.data.get(), (T_c.data.argmax(axis=-1)==c).sum())
    
    def garagara(self):
        pass

if __name__ == '__main__':
    train = Train()
    train.training(batchsize = 5)
    