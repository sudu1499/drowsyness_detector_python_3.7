import cv2
import yaml
import os
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

    