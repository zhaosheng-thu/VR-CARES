# main.py
from server import TCPServer
from model import ChatModel
import socket

def main():
    host = socket.gethostname()
    port = 10309
    dialogue_rounds = 5
    server = TCPServer(host, port)
    model = ChatModel('/root/szhao/model-weights/Llama-2-7b-chat-hf')
    
    # initialize the dialogue with the background prompts
    dialogue = model.init_dialogue()
    output_text = model(dialogue)
    dialogue += f'{output_text}</s>\n'
    
    for client_socket in server.start():
        
        # As the server starts, it sends the initial message to the client
        client_socket.send(output_text.encode("utf-8"))
        print("start:", dialogue)
        # Always wait for the client to send the message
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            input_text = data.decode("utf-8")
            dialogue += f'<s>Human: {input_text}</s>\n<s>Assistant:'

            # call the model
            output_text = model(dialogue)
            client_socket.send(output_text.encode("utf-8"))

            # update
            dialogue += f'{output_text}</s>\n'
            dialogue_rounds -= 1
            print("rounds left:", dialogue_rounds)
            if dialogue_rounds <= 0:
                break
            

        print("end:", dialogue)
        client_socket.close()

if __name__ == "__main__":
    main()
