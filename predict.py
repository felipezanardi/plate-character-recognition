from ultralytics import YOLO
import numpy as np

modelo = YOLO(r'runs\classify\train5\weights\best.pt') # Carregar modelo próprio

resultado = modelo(r'data\train\t\3491.png') # Imagem para identificar caractere

print(resultado)                            # teste - ver dados

names_dict = resultado[0].names

probs = resultado[0].probs.data.tolist()

print(names_dict)                           # teste - ver dados
print(probs)                                # teste - ver dados

print("O caractere é: ", names_dict[np.argmax(probs)])
