import socket

# 创建 socket 对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 获取本地主机名
host = socket.gethostname()
host = '172.17.0.5'
# 192.168.1.7
print("host: ", host)
# 设置端口号
port = 10309

# 绑定端口号
server_socket.bind((host, port))

# 设置最大连接数，超过后排队
server_socket.listen(5)

while True:
    # 建立客户端连接
    client_socket, addr = server_socket.accept()

    print('连接地址：', addr)

    # 发送数据到客户端
    client_socket.send('欢迎访问服务器！'.encode('utf-8'))

    # 接收客户端数据
    data = client_socket.recv(1024)
    print('收到的数据：', data.decode('utf-8'))

    # 关闭连接
    client_socket.close()
