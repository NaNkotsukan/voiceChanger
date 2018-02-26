# from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
# import chainer.Variable as V
from chainer import Variable as V
from chainer import optimizers
from chainer.serializers import save_npz, load_npz
import numpy as np
import cupy as xp
from model import Compressor, Generator, Discriminator
from data import dataset
import time
import pickle

class Train:
    def __init__(self):
        self.data=dataset()
        self.data.reset()
        self.reset()
        # self.load(1)
        # for i in range(100):
        #     self.reset()
        #     res = self.batch()
        #     print(F"{i} G_Loss:{res[0][0]} {res[0][1]} D_Loss:{res[1]}")
        #     self.save(i)
        self.setLR()
        self.time=time.time()
        # self.training(batchsize = 3)
        # n=1250
        # load_npz(f"param/com/com{n}.npz",self.compressor)
        # load_npz(f"param/gen/gen{n}.npz",self.generator)
        # load_npz(f"param/dis/dis{n}.npz",self.discriminator)
        # with open(f"param/com/com{i}.pickle", mode='rb') as f:
        #     self.compressor = pickle.load(f)
        # with open(f"param/gen/gen{i}.pickle", mode='rb') as f:
        #     self.generator = pickle.load(f)
        # with open(f"param/dis/dis{i}.pickle", mode='rb') as f:
        #     self.discriminator = pickle.load(f)

    def reset(self):
        self.compressor = None
        self.generator = None
        self.discriminator = None
        self.compressor = Compressor()
        self.generator = Generator(self.compressor)
        self.discriminator = Discriminator(self.compressor)
        self.generator.to_gpu()
        self.discriminator.to_gpu()

    def setLR(self, lr=0.003):
        self.gen_opt = optimizers.Adam(alpha=lr)
        self.gen_opt.setup(self.generator)
        self.dis_opt = optimizers.Adam(alpha=lr)
        self.dis_opt.setup(self.discriminator)
    
    def save(self, i):
        with open(f"param/com/com{i}.pickle", mode='wb') as f:
            pickle.dump(self.compressor, f)
        with open(f"param/gen/gen{i}.pickle", mode='wb') as f:
            pickle.dump(self.generator, f)
        with open(f"param/dis/dis{i}.pickle", mode='wb') as f:
            pickle.dump(self.discriminator, f)

    def load(self, i):
        with open(f"param/com/com{i}.pickle", mode='rb') as f:
            self.compressor = pickle.load(f)
        with open(f"param/gen/gen{i}.pickle", mode='rb') as f:
            self.generator = pickle.load(f)
        with open(f"param/dis/dis{i}.pickle", mode='rb') as f:
            self.discriminator = pickle.load(f)

    def training(self, batchsize = 1):
        for x in range(100):
            N = self.data.reset()
            for i in range(N // batchsize):
                res = self.batch(batchsize = batchsize)
                if not i%10:
                    print(F"{i//10} time:{int(time.time()-self.time)} G_Loss:{res[0][0]} {res[0][1]} D_Loss:{res[1]}")
                    if not i%50:
                        self.save(i)
                        # save_npz(f"param/com/com{i}.npz",self.compressor)
                        # save_npz(f"param/gen/gen{i}.npz",self.generator)
                        # save_npz(f"param/dis/dis{i}.npz",self.discriminator)
                        a,b,c=self.data.test()
                        # print(a.shape)
                        # print(b.shape)
                        # print(c.shape)
                        res=F.argmax(self.generator(a.astype(xp.int16),b.astype(xp.int16),c.astype(xp.int16)),-2).data.get().reshape(-1)
                        # print(f"res:{res.shape}")
                        self.data.save(res, i)

    def batch(self, batchsize = 2):
        A0, a0, B0, b0, b1, b2, b3, b4= self.data.next(batchSize = batchsize)
        _ = lambda x:x.astype(xp.float32)/255
        A_gen = self.generator(A0, a0, b0)
        B_gen = self.generator(B0, b3, b4)
        F_dis = self.discriminator(F.argmax(A_gen, -2).reshape(batchsize, 1, -1), b1)
        T_dis = self.discriminator(B0, b2)
        # print(B_gen.data[0])
        # print(A_gen.data[0])
        # print(F_dis.data)
        # print(B_gen.shape)
        # print( B0[:,:,:B_gen.shape[-1]-B0.shape[-1]].shape)
        L_gen0 = F.softmax_cross_entropy(B_gen, B0[:,:,:B_gen.shape[-1]-B0.shape[-1]].reshape(batchsize,-1))
        L_gen1 = F.softmax_cross_entropy(F_dis, xp.zeros(batchsize, dtype=np.int32))
        gen_loss=(L_gen0.data, L_gen1.data)
        L_gen = L_gen0 + L_gen1
        L_dis = F.softmax_cross_entropy(F_dis, xp.ones(batchsize, dtype=np.int32))
        L_dis += F.softmax_cross_entropy(T_dis, xp.zeros(batchsize, dtype=np.int32))

        self.generator.cleargrads()
        L_gen.backward()
        self.gen_opt.update()

        self.discriminator.cleargrads()
        L_dis.backward()
        self.dis_opt.update()

        return (gen_loss, L_dis.data)
    
    def garagara(self):
        pass

if __name__ == '__main__':
    train = Train()
    train.training(batchsize = 3)