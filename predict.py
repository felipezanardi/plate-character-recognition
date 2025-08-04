from ultralytics import YOLO
import numpy as np
from PIL import Image

modeloTreinado = YOLO(r'runs\classify\train5\weights\best.pt') # Carregar modelo próprio

# Imagem para identificar caractere
resultado = modeloTreinado(r'data\train\t\3491.png')    # endereço da imagem teste

names = resultado[0].names
probs = resultado[0].probs.data.tolist()

print("\nO caractere é: ", names[np.argmax(probs)]) # Output (classe com maior probabilidade)

# Gerar imagem com probabilidades do top5 (5 classe com maiores probabilidades)
for i, r in enumerate(resultado):
    im_bgr = r.plot()
    im_rgb = Image.fromarray(im_bgr[..., ::-1])
    
    r.show() # Mostrar resultado
    
    r.save(filename=f"resultado{i}.jpg") # Salvar resultado
