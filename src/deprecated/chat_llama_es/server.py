# server.py
import socket

class TCPServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            print(f"服务器已启动，监听 {self.host}:{self.port}...")
            while True:
                client_socket, addr = server_socket.accept()
                print('连接地址：', addr)
                yield client_socket

    def stop(self):
        # 服务器停止逻辑
        pass
