from multiprocessing import Condition
import socket,cv2, pickle,struct,yaml,time
from tensorflow.keras.models import load_model
import numpy as np
import client2

config=yaml.safe_load(open('config.yaml'))
model_path=config['model_path']
encoder=config['encoder']

model=load_model(model_path)

encoder=pickle.load(open('encoder','rb'))

det=cv2.CascadeClassifier('haarcascade_eye.xml')
det2=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def recognize(model,encoder,frame):
	frameg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	f=det2.detectMultiScale(frameg,1.3,5)
	for fx,fy,fxx,fyy in f:
		face=frame[fy:fy+fyy,fx:fx+fxx]
		e=det.detectMultiScale(face)
		for ex,ey,xx,yy in e:
			crp=face[ey:ey+yy,ex:ex+xx]
			crp=cv2.resize(crp,(112,112))
			crp=np.reshape(crp,(1,112,112,3))
			return np.argmax(model.predict(crp/255))
  
                        


# create socket
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '192.168.1.3' # paste your server ip address here
port = 9990
client_socket.connect((host_ip,port)) # a tuple
#client_socket.setblocking(0)
data = b""
payload_size = struct.calcsize("Q")
fc=0
f_o=[]
while True:
	while len(data) < payload_size:
		packet = client_socket.recv(4*1024) # 4K
		if not packet: break
		data+=packet
	packed_msg_size = data[:payload_size]
	data = data[payload_size:]
	msg_size = struct.unpack("Q",packed_msg_size)[0]

	while len(data) < msg_size:
		data += client_socket.recv(4*1024)
	frame_data = data[:msg_size]
	data  = data[msg_size:]
	frame = pickle.loads(frame_data)
	if len(f_o)==10:
		condtion=2#any number
		if f_o.count(0)>f_o.count(1):
			print('sleepy')
			client2.send_msg('0')
			#client_socket.send(str(1).encode())
			#print('has been sent')
			#f_o=[]
		else:
			print('active')
			client2.send_msg('1')
			#client_socket.send(str(0).encode())
			#print('has been sent')
			#f_o=[]
		
		f_o=[]
	else:
		f_o.append(recognize(model,encoder,frame))
    
	# cv2.imshow("RECEIVING VIDEO",frame)
	# key = cv2.waitKey(1) & 0xFF
	# if key  == ord('q'):
	# 	break
client_socket.close()