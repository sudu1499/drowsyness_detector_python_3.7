import socket

s2=socket.socket()
s2.connect(('192.168.1.3',8880))
def send_msg(msg):
    global s2
    print("###########msg from client2##############")
    s2.sendall(msg.encode())