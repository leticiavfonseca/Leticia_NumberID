import tensorflow.keras as keras
import tensorflow as tf
import gzip
import numpy as np
import matplotlib.pyplot as plt

# Pedro Luiz Barrozo dos Santos
# IplanRio - GTIG 18/06/2021
# Exemplo de Rede Neural Simples (DNN) com finalidade de portifólio
# Esse programa constrói e treina um modelo de rconhecimento de dígitos utilizando a biblioteca TensorFlow

# Carregando as imagens e labels da pasta source
TRAIN_DATASET_IMAGES = gzip.open('src/train-images-idx3-ubyte.gz','r')
TRAIN_DATASET_LABELS = gzip.open('src/train-labels-idx1-ubyte.gz','r')

image_size = 28
num_images = 500000

TRAIN_DATASET_IMAGES.read(16)
TRAIN_DATASET_LABELS.read(8)

buf_1 = TRAIN_DATASET_IMAGES.read()
buf_2 = TRAIN_DATASET_LABELS.read()

train_img = np.frombuffer(buf_1, dtype=np.uint8).astype(np.float32) / 255
train_lbl = np.frombuffer(buf_2, dtype=np.uint8).astype(np.float32)


# Array com 60000 imagens de 28x28 pixels para treinamento e validação dispostas na forma de uma array com 784 (28x28) elementos em uma dimensão
train_img = train_img.reshape(-1, 1 ,784)

# Array com 60000 labels correspondentes às 60000 imagens utilizadas para treinamento dos dados
train_lbl = train_lbl.reshape(-1,1).astype(int)
lbl_array = np.zeros([train_lbl.size, 1, 10])
for index in range(train_lbl.size):
    lbl_array[index, 0, train_lbl[index]] = 1

# Declaração do modelo
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape = (1, 784)))
model.add(keras.layers.Dense(128, activation='relu', name='First_Layer'))
model.add(keras.layers.Dense(64, activation='relu', name='Hidden_Layer'))
model.add(keras.layers.Dense(10, activation='softmax', name='Output_Layer'))


model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='binary_crossentropy'
)
print("Treinando o modelo")
model.fit(train_img, lbl_array, batch_size=1000, epochs=40)

model.save('Models/DNN_Digit_Recognition[bs=1000_epch=40_opz=adam_loss=binary-crossentropy]')
