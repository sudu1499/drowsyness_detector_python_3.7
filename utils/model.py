from keras.applications import xception
from keras.models import Sequential
from keras.layers import Dense,Flatten

def get_model(size):
    xcp=xception.Xception(include_top=False,weights='imagenet',input_shape=(size,size,3))

    for i in xcp.layers:
        i.trainable=False

    model=Sequential()
    model.add(xcp)
    model.add(Flatten())
    model.add(Dense(100,activation='relu',kernel_initializer='HeNormal'))
    model.add(Dense(100,activation='relu',kernel_initializer='HeNormal'))
    model.add(Dense(2,activation='softmax',kernel_initializer='HeNormal'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
    return model

