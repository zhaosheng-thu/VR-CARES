import socket

# 创建一个socket对象
s = socket.socket()

# 获取服务器的主机名
host = socket.gethostname()

remote_host_name = socket.getfqdn('192.168.1.7')
print('IP地址是192.168.1.7的远程主机，其主机名是:%s'%remote_host_name)
# 设置要连接的端口
port = 10309

# 连接到服务器
s.connect((host, port))
print("Connected to the server")

while True:
    data = s.recv(1024)
    if not data:
        break
    # 接收服务器发送的数据
    # print(f'\nassistant: {data.decode('utf-8')}\n')
    input_text = input("You: ")
    # 发送数据到服务器
    
    s.send(input_text.encode('utf-8'))

# 关闭连接
s.close()