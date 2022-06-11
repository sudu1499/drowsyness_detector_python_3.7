#from utils.get_data import get_data
#from utils.detector import detect_eye
#from utils.model import get_model
from tensorflow.keras.callbacks import EarlyStopping
import yaml
from sklearn.preprocessing import OneHotEncoder
import cv2,os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.applications import xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,BatchNormalization

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
    
    
def get_model(size):
    xcp=xception.Xception(include_top=False,weights='imagenet',input_shape=(size,size,3))

    for i in xcp.layers:
        i.trainable=False

    model=Sequential()
    model.add(xcp)
    model.add(Flatten())
    model.add(Dense(100,activation='relu',kernel_initializer='HeNormal'))
    model.add(Dropout(rate=.2))
    model.add(BatchNormalization())

    model.add(Dense(100,activation='relu',kernel_initializer='HeNormal'))
    model.add(Dropout(rate=.2))
    model.add(BatchNormalization())

    model.add(Dense(2,activation='softmax',kernel_initializer='HeNormal'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
    return model
def detect_eye(img_path,size):
    det=cv2.CascadeClassifier('haarcascade_eye.xml')
    det2=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    folder=input('folder name: ')
    new_path=img_path+f'\\{folder}'
    os.makedirs(new_path,exist_ok=True)
    count=0
    c=True
    vid=cv2.VideoCapture(0)
    while c:
        _,frame=vid.read()
        frameg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ee=det2.detectMultiScale(frameg,1.3,5)

        for fx,fy,fxx,fyy in ee:
            face=frameg[fy:fy+fyy,fx:fx+fxx]
            cv2.imshow('eye detectd',frame)

            e=det.detectMultiScale(face)
            for ex,ey,xx,yy in e:
                crp=face[ey:ey+yy,ex:ex+xx]
                crp=cv2.resize(crp,(size,size))
                #cv2.imshow('eye detectd',frame)
                cv2.imwrite(new_path+f'\\{count}.jpeg',crp)
                count+=1
                if count==150:
                    c=False
                                 ##########
            #cv2.rectangle(frame,(ex,ey),(ex+xx,ey+yy),(255,0,0),3)
            #cv2.imshow(' ',crp)
        
        if cv2.waitKey(1)==ord('q'):
            break
        
    vid.release()
    cv2.destroyAllWindows()

config=yaml.safe_load(open('config.yaml'))
img_path=config['img_path']
size=config['size']
model_path=config['model_path']

#################################### for adding new eye images######################
#detect_eye(img_path,size)
####################################################################################

x_train,x_test,x_val,y_train,y_test,y_val=get_data(config)

clb=EarlyStopping(patience=4,monitor='val_accuracy',restore_best_weights=True)

model=get_model(size)
model.fit(x_train/255,y_train,validation_data=(x_val/255,y_val),callbacks=clb,epochs=20,batch_size=16)

g=model.evaluate(x_test/255,y_test)
model.save(model_path)