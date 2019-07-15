import os
import numpy as np
from os import listdir
import cv2

def make_dir(dir):
    try:
        os.makedirs(dir)  
    except OSError:
        pass 

def audiofromfolder(folder):

    total = np.zeros(0,dtype=[
        ('x', np.float32, input_size ),
        ('y', np.int32, ())])

    print(listdir(folder))
    list_category = listdir(folder)
    label = 0
    sample_id = 0
    for category in list_category:
        sub_folder = os.path.join(folder,category)
        num = len(listdir(sub_folder))
        print("{} samples with label '{}'".format(num,category))
        new_samples = np.zeros(num,dtype=[
        ('x', np.float32, input_size ),
        ('y', np.int32, ())])

        for i in range (0,num):
            path = os.path.join(sub_folder,listdir(sub_folder)[i])
            image = cv2.imread(path,0).astype(float)

            if np.shape(image) != input_size[0:2]:
                print("resize images to fixed dimension")                
                image = cv2.resize(image, dsize = tuple(input_size[0:2]),interpolation=cv2.INTER_CUBIC)
            image = image[:,:,np.newaxis]
            new_samples[i]['x'] = image
            new_samples[i]['y'] = label
            sample_id+=1
            
        label+=1
        total = np.concatenate((total,new_samples), axis=0)

    return total['x'],total['y']


def noisefromfolder(folder):

    new_samples = np.empty(len(listdir(folder)),dtype=object)
    for i in range(0,len(listdir(folder))):
        path = os.path.join(folder,listdir(folder)[i])
        image = cv2.imread(path,0).astype(float)
        new_samples[i]=image
    return new_samples


def do(input_size):
    img_folder = os.path.join(DATA_PATH,'download','train_30classes_img')
    save_folder = os.path.join(DATA_PATH,'paper')
    make_dir(save_folder)
    Train_path = os.path.join(img_folder,"train")
    Valid_path = os.path.join(img_folder,"test")
    bg_noise_path = os.path.join(img_folder,"audio2img_bg")
    data_label = os.path.join(save_folder, "audio30_{}.npz".format(input_size[0]))
    data_noise = data_label.replace('.npz','_bg.npy')

    
    if os.path.exists(data_label):
        var = input("sample_label file already exists, replacing with new? (y/n)")
        if 'n' in var:
            return
        else:
            print ('loading training dataset')
            train_x_orig, train_y = audiofromfolder(Train_path)
            print ('loading testing dataset')
            test_x_orig, test_y = audiofromfolder(Valid_path)
            train_x,test_x = train_x_orig,test_x_orig
            save_list = dict()
            save_list.update(train_x=train_x, train_y=train_y,test_x=test_x, test_y=test_y)
            # data_label = data_label.replace('.npz','_{}.npz'.format(key))
            np.savez(data_label,**save_list)
            print('{} npz file is saved'.format(data_label))

    else:
        print ('loading training dataset')
        train_x_orig, train_y = audiofromfolder(Train_path)
        print ('loading testing dataset')
        test_x_orig, test_y = audiofromfolder(Valid_path)
        train_x,test_x = train_x_orig,test_x_orig
        save_list = dict()
        save_list.update(train_x=train_x, train_y=train_y,test_x=test_x, test_y=test_y)
        # data_label = data_label.replace('.npz','_{}.npz'.format(key))
        np.savez(data_label,**save_list)
        print('{} npz file is saved'.format(data_label))

    if os.path.exists(data_noise):
        var = input("noise file already exists, replacing with new? (y/n)")
        if 'n' in var:
            return
    else:
        if os.path.exists(bg_noise_path):
            print ('loading background noise')
            bg_noise = noisefromfolder(bg_noise_path)
            np.save(data_noise,bg_noise)
            print('background noise is saved')
        else:
            print('no background noise is found')

def normalize_np():
    save_folder = os.path.join(DATA_PATH,'paper')
    data_label = os.path.join(save_folder, "audio30_32.npz")
    tmp = np.load(data_label)
    new = {}
    for key in tmp.keys():
        if 'x' in key:
            new[key] = normalization(tmp[key])
        else:
            new[key] = tmp[key]

    save_list = {}
    # save_list.update(train_x_u = tmp['train_x_u'], train_y_u=tmp['train_y_u'])
    save_list.update(train_x=new['train_x'], train_y=new['train_y'],test_x=new['test_x'], test_y=new['test_y'])
    import pdb;pdb.set_trace()
    new_data_label = data_label.replace('.npz','_norm.npz')
    np.savez(new_data_label,**save_list)
    print('{} npz file is saved'.format(new_data_label))

def normalization(matrix):
    shape = np.shape(matrix)
    matrix = np.reshape(matrix,(shape[0],-1))
    min_val = np.amin(matrix,axis=-1,keepdims=True)
    max_val = np.amax(matrix,axis=-1,keepdims=True)
    if sum( (min_val -max_val)==0)>1:
        print("Wrong divide")
        import pdb;pdb.set_trace()
    result = (matrix-min_val)/(max_val-min_val)
    result = np.reshape(result,shape)
    result = result*2-1
    return result


if __name__ == "__main__":
    DATA_PATH = os.path.join('data', 'images', 'audio')
    input_size = (32,32,1)
    do(input_size)
