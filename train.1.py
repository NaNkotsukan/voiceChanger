import chainer.functions as F
import chainer.links as L
from chainer import Variable as V
from chainer import optimizers, optimizer
from chainer.serializers import save_npz, load_npz
import numpy as np
import cupy as xp
from model import Generator, Discriminator, Model_
from data import dataset
import time
import pickle

import copy

class Train:
    def __init__(self):
        self.model = Model_(10)
        self.model.to_gpu()
        self.model_opt = optimizers.MomentumSGD(lr=0.001)
        self.model_opt.setup(self.model)
        i = 0
        load_npz(f"param/model__/model{i}.npz", self.model)

        self.generator = Generator() 
        self.generator.to_gpu()
        self.gen_opt = optimizers.Adam()
        self.gen_opt.setup(self.generator)

        self.discriminator = Model_(2)
        self.discriminator.to_gpu()
        self.dis_opt = optimizers.Adam()
        self.dis_opt.setup(self.discriminator)

        # self.model_opt.add_hook(optimizer.WeightDecay(0.0001))

        self.data=dataset()
        self.data.reset()
        # self.reset()
        # self.load(1)
        # self.setLR()
        

        self.time=time.time()

        self.training(batchsize = 4)


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
                if not i%20:
                    print(F"{i} time:{int(time.time()-self.time)} Loss:{res[0]}  Acc:{res[1]}")
                    # testAcc = (np.concatenate([self.model(self.data.test[0][i*10:i*10+10], test=True).data.get().argmax(-1).flatten() for i in range(10)])==self.data.test[1]).sum()
                    if not i%100:
                        y = self.generator(xp.asarray(self.data.testData[0][:22050*4].reshape(1,1,1,-1)), xp.array([6])).data.get()
                        self.data.save(y, f"Garagara_{i}")
                        # save_npz(f"param/model/model{i}.npz",self.model)
                        save_npz(f"param/model/dis_{i}.npz",self.discriminator)
                        save_npz(f"param/model/gen_{i}.npz",self.generator)

    def batch(self, batchsize = 2):
        # 235 3267
        x, c = self.data.next(batchSize = batchsize,  dataSize=[3308], dataSelect=[0])

        x = x[0].reshape(batchsize, 1, 1, -1)
        c = xp.asarray(c[0])
        t = xp.random.randint(0,9,batchsize)
        t += t>=c
        # print(xp.abs(x).max(axis=-1).shape)
        x /= (((xp.random.rand(batchsize)*0.5+0.5) / xp.abs(x).max(axis=-1).reshape(-1)).reshape(-1,1,1,1))
        y = self.generator(x, t)
        x_ = self.generator(y, c)

        gen = self.discriminator(y)
        # c=xp.asarray(c[0])

        T_c = self.discriminator(x[:,:,:,:2285])
        L_dis0 = F.softmax_cross_entropy(gen, xp.zeros(batchsize, dtype=xp.int))
        L_dis1 = F.softmax_cross_entropy(T_c, xp.ones(batchsize, dtype=xp.int))
        L_gen0 = F.softmax_cross_entropy(gen, xp.ones(batchsize, dtype=xp.int))
        L_gen1 = F.softmax_cross_entropy(self.model(y), c)
        L_gen2 = F.mean_squared_error(x[:,:,:,1023:-1023], x_)
        L_dis = L_dis0 + L_dis1
        L_gen = L_gen0 + L_gen1 + L_gen2


        self.discriminator.cleargrads()
        L_dis.backward()
        self.dis_opt.update()

        self.generator.cleargrads()
        L_gen.backward()
        self.gen_opt.update()

        # self.model_opt.update()
        # self.model_opt.lr*=0.99
        # return 
        return (np.array([i.data.get() for i in [L_dis0, L_dis1, L_gen0, L_gen1, L_gen2]]), np.array([(T_c.data.argmax(axis=-1)==1).sum().get(), (gen.data.argmax(axis=-1)==0).sum().get()]))
    
    def garagara(self):
        pass

if __name__ == '__main__':
    train = Train()
    train.training(batchsize = 5)
    