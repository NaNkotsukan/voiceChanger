import numpy as np
import cupy as xp
import wave
import os
import pickle
import gc
import cupy as xp

class dataset:
    def __init__(self, dataLoad=False, sampling=22050, test=None):
        self.sampling=sampling
        if dataLoad:
            self.data = []
            self.dataLen = []
            # dirs = os.listdir("voices/")
            # for d in dirs:
            #     files = os.listdir("voices/" + d)
            #     f=[]
            #     for x in files:
            #         s = self.read(f"voices/{d}/{x}")
            #         f.append(s)
            #    np res = np.hstack(f)
            #     if res.shape[0] > self.sampling*20:
            #         print(f"{res.shape[0]} {d}")
            #         res = self.encode(res)
            #         self.dataLen.append(res.shape[0])
            #         self.data.append(res)
            for x in os.listdir("v__/"):
                s = self.read(f"v__/{x}")
                if s.shape[0] > self.sampling*20:
                    print(f"{s.shape[0]} {x}")
                    d=self.encode(s)
                    self.data.append(d)
                    self.dataLen.append(d.shape[0])

            g = self.encode(self.read("test/Garagara_.wav"))
            m = self.encode(self.read("test/minase.wav"))
            s = self.encode(self.read("test/minase.wav"))
            self.testData = (g,s)

            # self.data.append(s)
            # self.data.append(m)
            # self.dataLen.append(len(s))
            # self.dataLen.append(len(m))
            # self.dataLen=tuple(self.dataLen)
            # print(len(self.data))
            # print(len(self.dataLen))
            
            with open('D:/voice/dataLen__.pickle', mode='wb') as f:
                pickle.dump(self.dataLen,f)            
            with open('dataLen___.pickle', mode='wb') as f:
                pickle.dump(self.dataLen,f)
            # with open('D:/voice/data.pickle', mode='wb') as f:
            #     pickle.dump(self.data,f)

            np.savez("data_.npz",*self.data)
        else:
            self.data = tuple(np.load("data_.npz")[y] for y in np.load("data_.npz"))
            # self.data = tuple(np.load("data.npz")[y] for y in np.load("data.npz"))
            # with open('D:/voice_/data.pickle', mode='rb') as f:
            #     self.data = pickle.load(f)
            with open('dataLen_.pickle', mode='rb') as f:
                self.dataLen = pickle.load(f)

        # g = self.encode(self.read("test/Garagara_.wav"))
        # m = self.encode(self.read("test/minase.wav"))
        # s = self.encode(self.read("test/minase.wav"))
        # self.teacher = self.teacherIndex(8, s.shape[0] - 2**16)

        g = self.encode(self.read("test/Garagara_.wav"))
        s = self.encode(self.read("test/minase.wav"))
        self.testData = (g,s)

        # self.data=list(self.data)
        # self.data.append(g)
        # self.dataLen=list(self.dataLen)
        # self.dataLen.append(len(g))

        self.dataNum = len(self.data)
        # if dataNum!=None:
        #     self.dataNum = dataNum
        #     np.argsort(np.array(self.dataLen))[::-1]


        # self.dataNum = 10
        self.testnum=10
        self.test = (xp.asarray(np.vstack([np.vstack([x[i*3501:i*3501+3501] for i in range(self.testnum)]) for x in self.data]).reshape(-1, 1, 1, 3501)), 
        np.tile(np.arange(self.dataNum).reshape(self.dataNum,1),self.testnum).flatten())

        # print(sum(self.dataLen))
        # if test:
        #     index = np.random.permutation(self.dataNum)
        #     N = round(self.dataNum*test)
        #     self.data_ = [self.data[i] for i in index[:N]]
        #     self.data = [self.data[i] for i in index[N:]]
        #     self.dataLen_ = np.array(self.dataLen)[index[:N]]
        #     self.dataLen = np.array(self.dataLen)[index[N:]]
        #     self.dataNum_ = N
        #     self.dataNum = self.dataNum-N
        #     # for x in self.dataLen_:
        #     # [[print(self.data_[y].shape, x*i) for i in range(20)] for x,y in zip(self.dataLen_//20,index[:N])]
        #     self.data_ = [np.vstack([x[y*i:y*i+4607] for i in range(1, 5)]).reshape(4, 1, -1) for x,y in zip(self.data_, self.dataLen_//5)]
        #     # print(self.data_)
        #     # self.batchSize=2
                

        # # self.dataSize = dataSize
        # self.dataSelect = dataSelect
    
    def teacherIndex(self, batchSize, dNum):
        for i in range(10000):
            sLen = np.random.permutation(dNum)
            for i in range(dNum//batchSize):
                yield sLen[i:i+batchSize]

    def reset(self, num=1_000_000, N=1):
        self.nowIndex=0
        self.index=np.hstack(tuple(np.random.permutation(self.dataNum).astype(dtype=np.int8)[:self.dataNum//N*N] for i in range(num))).reshape(self.dataNum//N*num,N).T
        return self.index.shape[1]

    def next(self,batchSize=16, dataSize = (99671,97903,99671,97903,97903,97903,97903,97903), dataSelect = [0,0,1,1,1,1,1,1]):
        index = self.index[dataSelect,self.nowIndex:self.nowIndex+batchSize]
        r = tuple(self.dataCall(i,j).reshape(batchSize, 1, j) for i,j in zip(index, dataSize))
        self.nowIndex += batchSize
        return r, index

    def dataCall(self, t, size):
        # print([self.data[i][j:j+size].shape for i,j  in zip(t, tuple(np.random.randint(self.dataLen[k]-size) for k in t))])
        return xp.asarray(np.vstack(tuple(self.data[i][j:j+size] for i,j  in zip(t, tuple(np.random.randint(40000, self.dataLen[k]-size) for k in t)))))

    # def t():
        

    # def data(self, t, size):
    #     x=np.random.randint(self.dataLen[t])
    #     self.data[x:x+size]

    # def test(self, size = 112047):
    #     return (xp.asarray(self.testData[0]).reshape(1,1,-1), xp.asarray(self.testData[0][6*self.sampling:6*self.sampling+size]).reshape(1,1,-1), xp.asarray(self.testData[1][9*self.sampling:9*self.sampling+size]).reshape(1,1,-1))
    
    # def test(self, size = 112047):
    #     # print(next(self.teacher))
    #     x = np.vstack([self.testData[1][i:i+size] for i in next(self.teacher)])
    #     # print(x)
    #     return xp.asarray(x).reshape(len(x), 1, 1, -1)

    # def test(self):
    #     [np.vstack([x[i*1024:i*1024+1024] for i in range(10)]) for x in self.data]


    def save(self, sound, name):
        self.write(self.decode(sound), name)

    def read(self, file_name):
        wave_file = wave.open(file_name,"r")
        x = wave_file.readframes(wave_file.getnframes())
        x = np.frombuffer(x, dtype= "int16")
        print(wave_file.getnchannels(), wave_file.getframerate())
        return x

    def write(self, audio, fname):
        write_wave = wave.Wave_write(f"testGen/{fname}.wav")
        write_wave.setparams(wave._wave_params(1, 2, 22050, audio.shape[0], 'NONE', 'not compressed'))
        write_wave.writeframes(audio)
        write_wave.close()

    def encode(self, x):
        y=(x/32768).astype(np.float32)
        # y=((np.sign(x)*np.log1p(np.abs(x/32768*255))*128/np.log(256))+128).astype(np.uint8)
        return y

    def decode(self, y):
        z=(y*32768).reshape(-1).astype(np.int16)
        # y=(y-128).astype(np.int8)
        # z=((np.sign(y)*255**np.abs(y/128))*128).astype(np.int16)
        return z


if __name__ == "__main__":
    data=dataset(dataLoad=True)
    input("compleate")