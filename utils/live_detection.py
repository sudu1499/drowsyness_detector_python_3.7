import cv2
import numpy as np
import yaml
from tensorflow.keras.models import load_model
import pickle
import time

config=yaml.safe_load(open('config.yaml'))
model_path=config['model_path']
encoder=config['encoder']

model=load_model(model_path)

encoder=pickle.load(open('encoder','rb'))

det=cv2.CascadeClassifier('haarcascade_eye.xml')
det2=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vid=cv2.VideoCapture(0)
diff=0
final=0
while 1:
        init=time.time()
        diff=0
        output=[]
        while diff<=.5:
                #o=np.array([0,0])
                #o=o.reshape((-1,2))
                
                _,frame=vid.read()
                frameg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                f=det2.detectMultiScale(frameg,1.3,5)
                for fx,fy,fxx,fyy in f:
                        face=frame[fy:fy+fyy,fx:fx+fxx]

                        e=det.detectMultiScale(face)
                        for ex,ey,xx,yy in e:
                                crp=face[ey:ey+yy,ex:ex+xx]
                                crp=cv2.resize(crp,(112,112))
                                crp=np.reshape(crp,(1,112,112,3))
                                #o[0][int(np.argmax(model.predict(crp/255)))]=1 

                                output.append(np.argmax(model.predict(crp/255)))
                        
                        #print(encoder.inverse_transform(o)[0][0])
                #o[0][0]=0    
                #o[0][1]=0    
                
                final=time.time()
                diff=final-init
        #print(output)
        if output.count(0)>output.count(1):
                print('sleepy')
        else:
                print('active')
        # try:
        #         if output.count(0)/len(output)>.5:
        #                 print('sleepy')
        #         else:
        #                 print('active')
        # except ZeroDivisionError:
        #         print('active')

vid.release()

# cv2.destroyAllWindows()
