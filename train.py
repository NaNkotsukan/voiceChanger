import chainer.functions as F
import chainer.links as L
from chainer import Variable as V
from chainer import optimizers, optimizer
from chainer.serializers import save_npz, load_npz
import numpy as np
import cupy as xp
from model import Compressor, Generator, Discriminator
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
        # n=10
        # load_npz(f"param/com/com_{n}.npz",self.compressor)
        # load_npz(f"param/gen/gen_{n}.npz",self.generator)
        # load_npz(f"param/dis/dis_{n}.npz",self.discriminator)
        self.training(batchsize = 4)


    def reset(self):
        self.compressor = None
        self.generator = None
        self.discriminator = None
        self.compressor = Compressor()
        self.generator = Generator(self.compressor)
        self.discriminator = Discriminator(self.compressor)
        self.generator.to_gpu()
        self.discriminator.to_gpu()

    def setLR(self, lr=0.001):
        self.gen_opt = optimizers.Adam(alpha=lr)
        self.gen_opt.setup(self.generator)
        self.gen_opt.add_hook(optimizer.WeightDecay(0.0001))
        self.dis_opt = optimizers.Adam(alpha=lr)
        self.dis_opt.setup(self.discriminator)
        self.dis_opt.add_hook(optimizer.WeightDecay(0.0001))
    
    # def save(self, i):
    #     with open(f"param/com/com{i}.pickle", mode='wb') as f:
    #         pickle.dump(self.compressor, f)
    #     with open(f"param/gen/gen{i}.pickle", mode='wb') as f:
    #         pickle.dump(self.generator, f)
    #     with open(f"param/dis/dis{i}.pickle", mode='wb') as f:
    #         pickle.dump(self.discriminator, f)

    # def load(self, i):
    #     with open(f"param/com/com{i}.pickle", mode='rb') as f:
    #         self.compressor = pickle.load(f)
    #     with open(f"param/gen/gen{i}.pickle", mode='rb') as f:
    #         self.generator = pickle.load(f)
    #     with open(f"param/dis/dis{i}.pickle", mode='rb') as f:
    #         self.discriminator = pickle.load(f)

    def training(self, batchsize = 1):
        for x in range(100):
            N = self.data.reset()
            # a,b,c=self.data.test()
            # d=F.argmax(self.generator(a.astype(xp.float32),b.astype(xp.int16),c.astype(xp.int16)),-2).data.get().reshape(-1)
            # print(d[25000:26000])
            # self.data.save(d, "_")
            self.batch(batchsize = 1)
            for i in range(N // batchsize-1):
                # if not i%1:
                    # self.save(i)
                    # g=copy.deepcopy(self.generator).to_cpu
                # g.to_cpu
                # print(d[25000:25100])
                res = self.batch(batchsize = batchsize)
                if not i%10:
                    print(F"{i} time:{int(time.time()-self.time)} G_Loss:{res[0][0]} {res[0][1]} D_Loss:{res[1][0]+res[1][1]} D_Acc:{res[2]}")
                    save_npz(f"param/com/com_{i}.npz",self.compressor)
                    save_npz(f"param/gen/gen_{i}.npz",self.generator)
                    save_npz(f"param/dis/dis_{i}.npz",self.discriminator)
                    if not i%100:
                        a,b,c=self.data.test()
                        d=self.generator(a[:,:,:110250],b,c,test=True).data.get().reshape(-1)
                        self.data.save(d, i)
                        # del d
                        
                    # print(res[-1][0])
                    # print(res[-1][1])
                        

    def batch(self, batchsize = 2):
        A0, a0, B0, b0, b1, b2, b3, b4= self.data.next(batchSize = batchsize)
        _ = lambda x:x
        # _ = lambda x:x/xp.float32(32768)
        A_gen = self.generator(_(A0), _(a0), _(b0))
        B_gen = self.generator(_(B0), _(b3), _(b4))

        F_dis = self.discriminator(A_gen, _(b1))
        T_dis = self.discriminator(_(B0), _(b2))

        dis_acc = (F.argmax(F_dis,axis=1).data.sum(), xp.int32(batchsize) - F.argmax(T_dis,axis=1).data.sum())
        # acc = (dis_acc[0]+dis_acc[1])/8

        # self.dataRate = self.dataRate if dis_acc[0] == dis_acc[1] else self.dataRate / xp.float32(0.99) if dis_acc[0] > dis_acc[1] else self.dataRate * xp.float32(0.99)

        # receptionSize = B0.shape[-1] - B_gen.shape[-1]
        # L_gen0 = F.softmax_cross_entropy(B_gen, B0[:,:,receptionSize:].reshape(batchsize,-1))

        L_gen0 = F.mean_squared_error(F.reshape(B_gen, (batchsize, 1, -1)), B0[:,:,-32768:])
        L_gen1 = F.softmax_cross_entropy(F_dis, xp.zeros(batchsize, dtype=np.int32))
        gen_loss=(L_gen0.data, L_gen1.data)

        L_gen = L_gen0 + L_gen1
        L_dis0 = F.softmax_cross_entropy(F_dis, xp.ones(batchsize, dtype=np.int32))
        L_dis1 = F.softmax_cross_entropy(T_dis, xp.zeros(batchsize, dtype=np.int32))
        dis_loss = (L_dis0.data.get(), L_dis1.data.get())
        # L_dis = L_dis0 * min(xp.float32(1), 1 / self.dataRate) + L_dis1 * min(xp.float32(1), self.dataRate)
        L_dis = L_dis0 + L_dis1

        self.generator.cleargrads()
        L_gen.backward()
        self.gen_opt.update()

        self.discriminator.cleargrads()
        L_dis.backward()
        self.dis_opt.update()

        self.dis_opt.alpha*=0.99983
        self.gen_opt.alpha*=0.99983
        return (gen_loss, dis_loss, dis_acc, self.dataRate, (F_dis.data, T_dis.data))
    
    def garagara(self):
        pass

if __name__ == '__main__':
    train = Train()
    train.training(batchsize = 5)