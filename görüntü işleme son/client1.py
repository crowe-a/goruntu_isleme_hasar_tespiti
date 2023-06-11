import socket
import os
def soket_çağır():
    # Sunucu adresi ve portu
    HOST = '192.168.137.1'
    PORT = 8000

    # Gönderilecek dosya adı ve yolu
    filename = "/home/kuara/Desktop/görüntü işleme son/foto and txt/hasarlı0.jpg"
    filepath = os.path.abspath(filename)

    # Dosya boyutunu hesapla
    filesize = os.path.getsize(filepath)

    # Sunucuya bağlan
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))

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
                if not data:
                    break
                client_socket.sendall(data)

        # Sunucudan cevap al
        data = client_socket.recv(1024)
        if data != b'OK':
            raise ValueError('Sunucu cevabı alınamadı')

    print('Dosya gönderildi')
soket_çağır()