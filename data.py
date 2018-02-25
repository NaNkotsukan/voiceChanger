import numpy as np
import cupy as xp
import wave
import os
import pickle
import gc
import cupy as xp

class dataset:
    def __init__(self, dataLoad=False, sampling=22050):
        # # self.write(np.zeros(100000),"mazai.wav")
        self.sampling=sampling
        if dataLoad:
            dirs = os.listdir("voices/")
            self.data = []
            self.dataLen = []
            for d in dirs:
                files = os.listdir("voices/" + d)
                f=[]
                for x in files:
                    s = self.read(f"voices/{d}/{x}")
                    f.append(s)
                res = np.hstack(f)
                if res.shape[0] > self.sampling*20:
                    print(f"{res.shape[0]} {d}")
                    res = self.encode(res)
                    self.dataLen.append(res.shape[0])
                    self.data.append(res)
            for x in os.listdir("voice/"):
                s = self.read(f"voice/{x}")
                if s.shape[0] > self.sampling*20:
                    print(f"{s.shape[0]} {x}")
                    d=self.encode(s)
                    self.data.append(d)
                    self.dataLen.append(d.shape[0])

            self.dataLen=tuple(self.dataLen)
            with open('D:/voice/dataLen.pickle', mode='wb') as f:
                pickle.dump(self.dataLen,f)
            # with open('D:/voice/data.pickle', mode='wb') as f:
            #     pickle.dump(self.data,f)

            np.savez("D:/voice/data.npz",*self.data)
        else:
            self.data = tuple(np.load("D:/voice/data.npz")[y] for y in np.load("D:/voice/data.npz"))
            # with open('D:/voice/data.pickle', mode='rb') as f:
            #     self.data = pickle.load(f)
            with open('D:/voice/dataLen.pickle', mode='rb') as f:
                self.dataLen = pickle.load(f)

        self.dataNum = len(self.data)
        self.dataSize = (66550,65536,65536,65536,65536,65536,65536,65536)

        g = self.encode(self.read("test/Garagara_.wav"))
        s = self.encode(self.read("test/minase.wav"))
        self.testData = (g,s)

    def reset(self, num=1_000_000):
        self.nowIndex=0
        self.index=np.hstack(tuple(np.random.permutation(self.dataNum).astype(dtype=np.int8)[:self.dataNum//3*3] for i in range(num))).reshape(self.dataNum//3*num,3).T
        return self.index.shape[1]

    def next(self,batchSize=16):
        r = tuple(self.dataCall(i,j).reshape(batchSize, 1, j) for i,j in zip(self.index[[0,0,0,1,1,1,2,2],self.nowIndex:self.nowIndex+batchSize],self.dataSize))
        self.nowIndex+=batchSize
        return r

    def dataCall(self, t, size):
        return xp.asarray(np.vstack(tuple(self.data[i][j:j+size] for i,j  in zip(t, tuple(np.random.randint(self.dataLen[k]-size) for k in t)))))

    # def data(self, t, size):
    #     x=np.random.randint(self.dataLen[t])
    #     self.data[x:x+size]

    def test(self, size = 65536):
        return (xp.asarray(self.testData[0]).reshape(1,1,-1), xp.asarray(self.testData[0][6*self.sampling:6*self.sampling+size]).reshape(1,1,-1), xp.asarray(self.testData[1][9*self.sampling:9*self.sampling+size]).reshape(1,1,-1))

    def save(self, sound, name):
        # AudioSegment(sound.tobytes(),sample_width=1,frame_rate=44100,channels=1).export(f"testGen/denxChan{name}.mp3", format="mp3")
        self.write(self.decode(sound), name)

    def read(self, file_name):
        wave_file = wave.open(file_name,"r") #Open
        x = wave_file.readframes(wave_file.getnframes()) #frameの読み込み
        x = np.frombuffer(x, dtype= "int16") #numpy.arrayに変換
        return x

    def write(self, audio, fname):
        write_wave = wave.Wave_write(f"testGen/{fname}.wav")
        write_wave.setparams(wave._wave_params(1, 2, 22050, audio.shape[0], 'NONE', 'not compressed'))
        write_wave.writeframes(audio)
        write_wave.close()

    def encode(self, x):
        y=(x/2**16).astype(np.float32)
        # y=((np.sign(x)*np.log1p(np.abs(x/32768*255))*128/np.log(256))+128).astype(np.uint8)
        return y

    def decode(self, y):
        # y=y.astype(np.int8)-128
        # z=((np.sign(y)*255**np.abs(y/128))*128).astype(np.int16)
        z=(y*2**16).astype(np.int16)
        return z


if __name__ == "__main__":
    data=dataset(dataLoad=True)
    input("compleate")