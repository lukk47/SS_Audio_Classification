import numpy as np
from matplotlib.image import imsave
import librosa
from scipy import signal
import os
from PIL import Image
import pandas as pd
""" plot spectrogram"""

def make_dir(dir):
    try:
        os.makedirs(dir)  
    except OSError:
        pass 

def plotstft(audiopath, name='tmp.png' , n_mels = 32):
    sr_set = 16000
    n_mels = 32
    wavfile,sr = librosa.core.load(audiopath,sr = sr_set)

    if len(wavfile) < 16000:
        wavfile = np.pad(wavfile, (0, max(0, 16000 - len(wavfile))), "constant")
    if n_mels ==32:
        tmp = librosa.feature.melspectrogram(wavfile, sr=sr, n_mels=n_mels)
    elif n_mels == None:
        tmp = librosa.feature.melspectrogram(wavfile, sr=sr)
    else:
        assert False, "Wrong input of dimension of output images"

    spec = librosa.power_to_db(np.abs(tmp))
    
    print('Writing ',name)
    imsave(name, spec)


def main(data_path,img_path,n_mels):
    train_img_path = os.path.join(img_path,'train/')
    test_img_path = os.path.join(img_path,'test/')


    train_list_file = os.path.join(data_path,'train.csv')
    f = pd.read_csv(train_list_file)
    classes = set(f['Class'])
    for cate in classes:
        make_dir(train_img_path+cate)
    for [file,cate] in f.values:
        dst = os.path.join(train_img_path,cate,str(file)+'.png') 
        plotstft(os.path.join(data_path,'Train',str(file)+'.wav'),dst, n_mels = n_mels)

if __name__ == "__main__":
    root_path = os.path.join('.','data', 'images', 'audio','download',)
    data_path = os.path.join(root_path, 'urban-sound-classification')
    img_path = os.path.join(root_path,"urban_img")
    main(data_path,img_path,32)