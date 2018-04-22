
from keras.utils.np_utils import to_categorical
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import random


original_facial_expression =  {'Angry': 0,'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5,'Neutral': 6};
final_facial_expresion = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'];


def fnReconstruct(original_pixels, size=(48,48)):
        original_pixels = np.array(list(map(int, original_pixels.split())))
        return original_pixels.reshape(size)

def fnGetEmotionCount(y_private, classes, verbose=True):
    emo_classcount = {}
    for new_num, _class in enumerate(classes):
        y_private.loc[(y_private == original_facial_expression[_class])] = new_num
        class_count = sum(y_private == (new_num))
        if verbose:
            print('{}: {} with {} samples'.format(new_num, _class, class_count))
        emo_classcount[_class] = (new_num, class_count)
        return y_private.values, emo_classcount
 
    
def fnLoadData(Sample_split_fraction=0.3, usage='PrivateTest', boolCategorize=True, verbose=True,
default_classes=['Angry', 'Happy'], filepath='data/fer2013.csv'):
    
    df = pd.read_csv(filepath)
    df = df[df.Usage == usage]
    frames = []
    for _class in default_classes:
        class_df = df[df['emotion'] == original_facial_expression[_class]]
        frames.append(class_df)
    data = pd.concat(frames, axis=0)
    rows = random.sample(list(data.index), int(len(data) * Sample_split_fraction))
    data = data.loc[rows]
    print ('{} set for {}: {}'.format(usage, default_classes, data.shape))
    data['pixels'] = data.pixels.apply(lambda x: fnReconstruct(x))
    x = np.array([mat for mat in data.pixels])
    X_private = x.reshape(-1, 1, x.shape[1], x.shape[2])
    Y_private, new_dict = fnGetEmotionCount(data.emotion, default_classes, verbose)
    print (new_dict)
    if boolCategorize:
        Y_private = to_categorical(Y_private)
    return X_private, Y_private, new_dict



def fnSaveData(X_private, Y_private, fname='', folder='data/'):
    np.save(folder + 'X_private' + fname, X_private)
    np.save(folder + 'Y_private' + fname, Y_private)
    
    
if __name__ == '__main__':
    # makes the numpy arrays ready to use:
    print ('Making moves...')
    final_facial_expresion = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

    X_private, Y_private, emo_dict = fnLoadData(Sample_split_fraction=1.0,
                                            default_classes=final_facial_expresion,
                                            usage='PrivateTest',
                                            verbose=True)
    print ('Saving...')
    fnSaveData(X_private, Y_private, fname='_test')
    print (X_private.shape)
    print (Y_private.shape)
    print ('Done!')