from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adadelta
from keras import models,layers,optimizers
from PIL import Image
import random
import keras as K
from keras.layers import Conv2D, MaxPooling2D,Dense, Dropout, Flatten

# 验证码中的字符, 不用汉字
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#             'v', 'w', 'x', 'y', 'z']
# ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#             'V', 'W', 'X', 'Y', 'Z']


# 验证码一般都无视大小写；验证码长度4个字符
# def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    while 1:
        image = ImageCaptcha()

        captcha_text = random_captcha_text()
        captcha_text = ''.join(captcha_text)
        captcha = image.generate(captcha_text)
        # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件
        train_image = []
        train_text = []
        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)
        captcha_image = rgb2gray(captcha_image)
        captcha_text = to_one_hot(captcha_text, 4, 10)
        captcha_text = captcha_text.reshape(40)
        captcha_image = captcha_image / 255
        train_text.append(captcha_text)
        train_image.append(captcha_image)
        train_text = np.array(train_text)
        train_image = np.array(train_image)
        yield ({'conv2d_1_input': train_image}, {'dense_1': train_text})


def to_one_hot(text,dimention1, dimention2):
    results = np.zeros((dimention1, dimention2))
    for i in range(len(text)):
        results[i][int(text[i])] = 1.
    return results


batch_size = 128
epochs = 100


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = gray.reshape((60, 160, 1))
    return gray


def new_metrics_accuracy(y_true, y_pred):
    if not K.backend.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.backend.cast(y_true, y_pred.dtype)
    y_true_new = K.backend.reshape(y_true, [-1, 4, 10])
    y_pred_new = K.backend.reshape(y_pred, [-1, 4, 10])
    y_true_maxindex = K.backend.argmax(y_true_new, 2)
    y_pred_maxindex = K.backend.argmax(y_pred_new, 2)
    aeqb = tf.cast(tf.equal(y_true_maxindex, y_pred_maxindex), tf.int32)
    results = tf.equal(tf.reduce_sum(aeqb, 1), tf.reduce_sum(tf.ones_like(aeqb), 1))
    print(results)
    print(K.backend.cast(results, K.backend.floatx()))
    print(K.backend.mean(K.backend.cast(results, K.backend.floatx())))
    return K.backend.mean(K.backend.cast(results, K.backend.floatx()))


if __name__ == '__main__':
    #产生随机的10000个样本并保存至captcha.npz文件中
    #text = []
    #image = []
    #for i in range(10000):
     #   text_one,image_one = gen_captcha_text_and_image()
     #   text.append(text_one)
      #  image.append(image_one)
    #np.savez('captcha', image=image, text=text)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(60, 160, 1), padding='SAME'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='SAME'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='SAME'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='SAME'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='SAME'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(40, activation='sigmoid'))
    model.summary()
    rp = optimizers.rmsprop(lr=0.01)
    model.compile(optimizer=rp, loss='binary_crossentropy',  metrics=[new_metrics_accuracy])
    print(model.summary())
    model.fit_generator(generator=gen_captcha_text_and_image(), steps_per_epoch=128, epochs=5000, verbose=1)
    model.save('D:/USTCSSE/PyCharm/pythonProject/model1.h5')
    score = model.evaluate(test_image, test_text, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])