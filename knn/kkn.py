import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Datos originales (peso, estatura)
x1 = np.array([[49], [1.43]])
x2 = np.array([[51], [1.55]])
x3 = np.array([[57], [1.58]])
x4 = np.array([[47], [1.55]])
x5 = np.array([[54], [1.60]])
x6 = np.array([[56], [1.58]])
x7 = np.array([[59], [1.64]])
x8 = np.array([[53], [1.61]])
x9 = np.array([[58], [1.63]])
x10 = np.array([[52], [1.60]])
x11 = np.array([[75], [1.73]]) # adultos
x12 = np.array([[80], [1.75]])
x13 = np.array([[75], [1.69]])
x14 = np.array([[65], [1.71]])
x15 = np.array([[75], [1.79]])
x16 = np.array([[77], [1.76]])
x17 = np.array([[65], [1.71]])
x18 = np.array([[70], [1.70]])
x19 = np.array([[78], [1.81]])
x20 = np.array([[70], [1.67]])

c0 = np.zeros(10)
c1 = np.ones(10)

X = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20), axis=1)
C = np.concatenate((c0, c1), axis=0)
C = np.uint8(C)

# Nuevos datos a clasificar
y1 = np.array([[62], [1.73]])
y2 = np.array([[70], [1.70]])
y3 = np.array([[53], [1.68]])
Y = np.concatenate((y1, y2, y3), axis=1)

# Parámetro K
k = 3

# Función para realizar la clasificación con KNN
def knn_classify(X, C, Y, k):
    predictions = []
    for y in Y.T:  # Itera sobre cada nuevo punto
        # Calcula la distancia de y a todos los puntos en X
        distances = np.array([distance.euclidean(y, x) for x in X.T])
        # Encuentra los k vecinos más cercanos
        nearest_neighbors_indices = np.argsort(distances)[:k]
        # Predice la clase mayoritaria entre los k vecinos más cercanos
        nearest_classes = C[nearest_neighbors_indices]
        pred_class = np.argmax(np.bincount(nearest_classes))
        predictions.append(pred_class)
    return predictions

# Clasifica los nuevos puntos
predictions = knn_classify(X, C, Y, k)

# Visualiza los datos originales y los nuevos puntos clasificados
for i in range(X.shape[1]):
    if C[i] == 0:
        marker = 'v'
        color = 'red'
    else: 
        marker = 'o'
        color = 'blue'
    plt.scatter(x=X[0, i], y=X[1, i], marker=marker, c=color)

for i, y in enumerate(Y.T):
    plt.scatter(x=y[0], y=y[1], marker='s', c='black')
    plt.text(y[0]+0.1, y[1], f'Class: {predictions[i]}', fontsize=12)

plt.xlabel('Peso kg')
plt.ylabel('Estatura mtrs')
plt.title('KNN')
plt.legend(['Clase 0', 'Clase 1', 'Nuevos puntos'])
plt.show()

# Mostrar las predicciones
for i, pred in enumerate(predictions):
    print(f"Nuevo punto {i+1} clasificado como Clase {pred}")
