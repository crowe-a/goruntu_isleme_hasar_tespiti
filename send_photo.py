import socket
import os,time

def send_with_tcp(x):
    print(x)
    #input(" : ")
    # Sunucu adresi ve portu
    HOST = 'localhost'
    PORT = 8000


    while True:
        # Sunucuya bağlan
        try:
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((HOST, PORT))
                i=0
                time.sleep(1)
                while i<=x:
                    if i==1000:
                        break
                    
                    print(i)
                    # Gönderilecek dosya adı ve yolu
                    filename = 'photo_and_txt/hasarli{}.jpg'.format(i)
                    filepath = os.path.abspath(filename)
                    

                    # Dosya boyutunu hesapla
                    filesize = os.path.getsize(filepath)

                    # Dosya boyutunu sunucuya gönder
                    client_socket.sendall(filesize.to_bytes(8, byteorder='big'))


                    # Sunucudan onay al
                    data = client_socket.recv(1024)
                    if data != b'OK':
                        raise ValueError('Sunucu onayı alınamadı')

                    # Dosya verisini sunucuya gönder
                    with open(filepath, 'rb') as f:
                        while True:
                            data = f.read(1024)
                            client_socket.sendall(data)
                            if not data:
                                break
                    print('Dosya gönderildi',i)
                    i+=1
                    
                

                    

                    # # Sunucudan cevap al
                    # data = client_socket.recv(1024)
                    # if data != b'OK':
                    #     raise ValueError('Sunucu cevabı alınamadı')
                

                    
        except:
            print("hata")
