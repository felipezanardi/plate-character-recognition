from ultralytics import YOLO

#DATA = r'C:\Users\Felipe\PycharmProjects\PlateDetection\data' # Endereço (full path) do database
DATA = 'data' # Endereço do database

modeloBase = YOLO('yolo11n-cls.pt') # Carregar Modelo

results = modeloBase.train(data=DATA, epochs=20, imgsz=96) # Treinar Modelo
