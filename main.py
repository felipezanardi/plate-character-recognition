from ultralytics import YOLO

DATA = r'C:\Users\Lider\PycharmProjects\PlateDetection\data' # Endereço do database

model = YOLO('yolo11n-cls.pt') # Carregar Modelo

results = model.train(data=DATA, epochs=20, imgsz=96) # Treinar Modelo

'''
epochs: quanto maior, melhor

imgsz: tamanho da imagem processada.
    as imagens do database tem resolução 75x100,
    imgsz=64 faz com que se perca uma parte de informação da imagem,
    fazendo com que perca acurácia, porém tempo de treinamento é menor.
    Usar imgsz foi ideal para não perder informação, sem aumentar muito
    o tempo de treinamento
    (aparentemente é recomendado esse valor ser um número múltiplo de 32)
    (a partir do epoch 15/16, não apresenta grande melhora)
'''
