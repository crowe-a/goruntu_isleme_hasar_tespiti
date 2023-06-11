import torch
import numpy as np
import cv2
from time import time

from dronekit import Command, connect, VehicleMode, LocationGlobalRelative

from pymavlink import mavutil
import client1

connection_string="127.0.0.1:14550"
# connection_string="tcp:127.0.0.1:5763"

# iha = connect(connectinString, wait_ready=True)

#iha = connect('/dev/ttyAMA0', wait_ready=True, baud=57600)

#iha = connect(connection_string, baud=115200, wait_ready=True, timeout=60)

#iha = connect(connection_string, wait_ready=False);iha.wait_ready(True,timeout=300)

class SolarPanel:
    

    def __init__(self, capture_index, model_name,z):
        """
        hangi kamerayı kullancağımız, hangi modeli kullanacağımız ekran kartı mı yoksa işlemci mi kullanacağız
        ve bazı değişkenlere atama yapıyoruz
        """
        
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.object_saved = set() 
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.z=z
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        kameradan görüntü alıyoruz
        """
       
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Pytorch hub'dan Yolov5 modelini indiriyoruz
        ve bunu modüle geri döndürüyoruz 
        """
        if model_name:
            model = torch.hub.load('/home/kuara/Desktop/görüntü işleme son/görüntü işleme son/yolov5', 'custom', path=model_name,source='local')
       
        return model

    def score_frame(self, frame):
        """
        kameradan aldığı görüntüyü modele sokarak ondan tahmin oranı alıyoruz 
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        classlarımızı labela dönüştürüyoruz.
        """
        return self.classes[int(x)]
    def plot_boxes(self, results, frame):
        """
        aranan objenin hangi konumlar içinde olduğunu buluyoruz.
        """
        labels, cord = results
        n = len(labels) 
        # .shape için --> 0 height, 1 width, 2 number of channels
        #x_shape = genişlik w, y_shape = yükseklik h
        x_shape, y_shape = frame.shape[1], frame.shape[0]


        for i in range(n):
            row = cord[i]
            if row[4] >= 0.6:

                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                valu = self.class_to_label(labels[i])

                while "BakimGereken" == valu:
                    red = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), red, 2)
                    cv2.putText(frame, f"{valu} {row[4]:.2f}", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, red, 2)
                    cv2.putText(frame, "BakimGereken panel tespit edildi.", (440,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                    # cv2.imwrite("/home/kuara/Desktop/görüntü işleme son/foto and txt/hasarlı{}.jpg".format(self.z),frame)
                    # liste=[iha.location.global_relative_frame.lat,iha.location.global_relative_frame.lon,iha.location.global_relative_frame.alt]
                    # with open("/home/kuara/Desktop/görüntü işleme son/foto and txt/liste{}.txt".format(self.z),"w+") as file:
                    #     for item in liste:
                    #         file.write("%s " % item)
                    #     file.write("\n")
                    #     print("kayıt ediliyor")
                    #     print('Alınan liste:',self.z)
                    self.z+=1
                    break
                    
                while "Saglam" == valu :
                    green = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), green, 2)
                    cv2.putText(frame, f"{valu} {row[4]:.2f}", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, green, 2)
                    cv2.putText(frame, "Saglam panel tespit edildi.", (540,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                    break

        return frame

    def __call__(self):
        
        """
        kameramızı açarak aranan nesnenin nerede olduğunu hangi nesne olduğunu ve % kaç olasılıkla onun olduğunu yazıyoruz.
        """
        
        cap = self.get_video_capture()
        assert cap.isOpened()
      
        while True:
              
            ret, frame = cap.read()
            if not ret:
                print("Kamera okunamadı..")
                break
            
            frame = cv2.resize(frame, (1024,780))
            
            start_time = time()
            results = self.score_frame(frame)
                    
                    
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            #print(f"her saniye frame yaz : {fps}")
             
            cv2.putText(frame, "Kontrol ediliyor...",(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(frame, f'FPS: {int(fps)}',(10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow('Solar Panel Detection', frame)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
      
        cap.release()
        
        cv2.destroyAllWindows()
        

detector = SolarPanel(capture_index=0, model_name='/home/kuara/Desktop/görüntü işleme son/görüntü işleme son/lastson.pt',z=0)
detector()

#client1.soket_çağır()