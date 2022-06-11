import socket, cv2, pickle,struct,imutils
from threading import Thread
# Socket Create
class server1(Thread):
	def run(self):
		server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		host_name  = socket.gethostname()
		host_ip = socket.gethostbyname(host_name)
		print('HOST IP:',host_ip)
		port = 9990
		socket_address = ('192.168.1.2',port)

		# Socket Bind
		server_socket.bind(socket_address)
		#server_socket.setsockopt( socket.SOL_SOCKET,socket.SO_RCVTIMEO,1)
		# Socket Listen
		server_socket.listen(5)
		print("LISTENING AT:",socket_address)

		# Socket Accept
		while True:
			client_socket,addr = server_socket.accept()
			print('GOT CONNECTION FROM:',addr)
			if client_socket:
				vid = cv2.VideoCapture(0)
				
				while(vid.isOpened()):
					img,frame = vid.read()
					frame = imutils.resize(frame,width=320)
					a = pickle.dumps(frame)
					message = struct.pack("Q",len(a))+a
					client_socket.sendall(message)

					# try:
					# 	state=int(client_socket.recv(1024).decode())
					# 	print(state)

					# except:
					# 	pass
				# cv2.imshow('TRANSMITTING VIDEO',frame)
				# key = cv2.waitKey(1) & 0xFF
				# if key ==ord('q'):
				# 	client_socket.close()
class server2(Thread):
	def run(self):
		s=socket.socket()
		s.bind(('192.168.1.2',8880))
		s.listen(1)
		print('#####################started second server########################')
		c,a=s.accept()
		print('connected to ',c)
		while 1:
			print(c.recv(1024).decode())

s1=server1()
s2=server2()

s1.start()
s2.start()