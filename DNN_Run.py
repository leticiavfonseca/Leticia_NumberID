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
TEST_DATASET_IMAGES = gzip.open('src/t10k-images-idx3-ubyte.gz','r')
TEST_DATASET_LABELS = gzip.open('src/t10k-labels-idx1-ubyte.gz','r')
TEST_DATASET_IMAGES.read(16)
TEST_DATASET_LABELS.read(8)

class Neural():
    def __init__(self, images, label, img_size=28):
        self._model = keras.models.load_model('Models/DNN_Digit_Recognition[bs=1000_epch=40_opz=adam_loss=binary-crossentropy]')
        self._test_img = (np.frombuffer(images.read(), dtype=np.uint8).astype(np.float32) / 255).reshape(-1, 1 ,784)
        self._test_lbl = np.frombuffer(label.read(), dtype=np.uint8).astype(np.float32)
        self._test_lbl = self._test_lbl.reshape(-1,1).astype(int)
        self._imgsz = img_size


    def precision(self):
        count = 0
        for index in range(10000):
            pred = np.argmax(self._model(self._test_img[index].reshape(-1,1,784)))
            lbl = self._test_lbl[index]
            if int(pred) == int(lbl):
                count += 1

        print('Precisão do modelo :', 100 * count/(10000))
        return None

    def plot_n_guess(self, term: int):
        result = self._model(self._test_img[int(term)].reshape(-1,1,784))
        print(f"O modelo identificou o número como {np.argmax(result)}")
        image = np.asarray(self._test_img[int(term)]).reshape(-1,28)
        plt.figure(1)
        plt.subplot(1, 2, 1)
        img = plt.imshow(image)
        plt.show()
        return None


if __name__ == "__main__":
    net = Neural(TEST_DATASET_IMAGES, TEST_DATASET_LABELS)

    net.plot_n_guess(6)
    net.precision()