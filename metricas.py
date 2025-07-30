import os
import pandas as pd
import matplotlib.pyplot as plt

csv = r'runs\classify\train5\results.csv'
resultados = pd.read_csv(csv)

''' Loss '''
plt.figure()
plt.plot(resultados['epoch'], resultados['train/loss'], label='train loss')
plt.plot(resultados['epoch'], resultados['val/loss'], label='val loss', c='red')
plt.grid()
plt.title('Loss / Epochs')
plt.ylabel('loss')
plt.xlabel('epochs')

''' Acurácia '''
plt.figure()
plt.plot(resultados['epoch'], resultados['metrics/accuracy_top1'] * 100)
plt.grid()
plt.title('Acurácia / Epochs')
plt.ylabel('acurácia (%)')
plt.xlabel('epochs')

plt.show()
