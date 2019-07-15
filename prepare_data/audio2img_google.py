
import numpy as np
from matplotlib.image import imsave
import librosa
from scipy import signal
import os
from PIL import Image
""" plot spectrogram"""


def make_dir(dir):
    try:
        os.makedirs(dir)  
    except OSError:
        pass 


def plotstft(audiopath, name='tmp.png' , n_mels = 32):
    wavfile = librosa.core.load(audiopath, sr=16000)[0]

    if len(wavfile) < 16000:
        wavfile = np.pad(wavfile, (0, max(0, 16000 - len(wavfile))), "constant")
    # number of samples between successive frames   hop
    if n_mels == 64:
        tmp = librosa.feature.melspectrogram(wavfile, sr=16000, n_mels=n_mels, hop_length=253)
    elif n_mels ==32:
        tmp = librosa.feature.melspectrogram(wavfile, sr=16000, n_mels=n_mels)
    elif n_mels == 128:
        tmp = librosa.feature.melspectrogram(wavfile, sr=16000, n_mels=n_mels, hop_length=100)
        tmp = np.array(Image.fromarray(tmp).resize((n_mels,n_mels),Image.BICUBIC))
    else:
        assert False, "Wrong input of dimension of output images"

    spec = librosa.power_to_db(np.abs(tmp))
    
    print('Writing ',name)
    imsave(name, spec)


def audio2img_file(path,dst_folder,n_mels=32):
    folder = ''
    make_dir(dst_folder)
    for file in os.listdir(path):
        if 'wav' in file:
            dst = os.path.join(dst_folder,folder,file.replace('wav','png'))
            plotstft(os.path.join(path,folder,file),dst, n_mels = n_mels)


def exclude_fun(folder):
    for exclude_name in exclude:
        if exclude_name in folder:
            return False
    return True 


def main(data_path,img_path,n_mels):
    train_img_path = os.path.join(img_path,'train/')
    test_img_path = os.path.join(img_path,'test/')
    bg_noise_img_path = os.path.join(img_path,'audio2img_bg/')
    bg_noise_wav_path = os.path.join(data_path,'_background_noise_') 
    test_list_file = os.path.join(data_path,'testing_list.txt')

    f = open(test_list_file)
    test_list = f.read().splitlines()

    for folder in os.listdir(data_path):
        if exclude_fun(folder):
            make_dir(train_img_path+folder)
            make_dir(test_img_path+folder)

    for folder in os.listdir(train_img_path):
        for wav in os.listdir(os.path.join(data_path,folder)):
            file_name = os.path.join(folder,wav)
            if file_name not in test_list:
                dst = os.path.join(train_img_path,file_name)
            else:
                dst = os.path.join(test_img_path,file_name)
            dst = dst.replace('wav','png')
            plotstft(os.path.join(data_path,file_name),dst, n_mels = n_mels)

    f.close()
    audio2img_file(bg_noise_wav_path,bg_noise_img_path,n_mels=n_mels)
    

if __name__ == "__main__":
    exclude = '., LICENSE, _background_noise_'.split(', ')
    root_path = os.path.join('.','data', 'images', 'audio','download',)
    data_path = os.path.join(root_path, 'speech_commands_v0.01')
    img_path = os.path.join(root_path,"train_30classes_img")

    main(data_path,img_path,32)