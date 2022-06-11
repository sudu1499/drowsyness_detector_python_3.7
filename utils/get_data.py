import cv2
from glob import glob
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

def get_data(config):
    img_path=config['img_path']
    encoder=config['encoder']
    ohe=OneHotEncoder()
    x=[]
    y=[]
    for i in glob(img_path+'\\*'):
        lable=i.split('\\')[-1]
        for j in glob(i+'\\*'):
            img=cv2.imread(j)
            x.append(img)
            y.append(lable)
    x=np.array(x)
    y=np.array(y)
    y=np.reshape(y,(-1,1))

    y=ohe.fit_transform(y).toarray()
    pickle.dump(ohe,open('encoder','ab'))
    x_tr,x_test,y_tr,y_test=train_test_split(x,y,test_size=.25)
    x_train,x_val,y_train,y_val=train_test_split(x_tr,y_tr,test_size=.25)

    return x_train,x_test,x_val,y_train,y_test,y_val
