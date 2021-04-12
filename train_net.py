import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import matplotlib.pyplot as plt
import os
xsize = 2
model_input = tf.keras.layers.Input(shape=(xsize,))
net = tf.keras.layers.Dense(3*xsize, activation='relu')(model_input)
net = tf.keras.layers.Dense(3*xsize, activation='relu')(net)
# net = tf.keras.layers.Dense(xsize, activation='relu')(net)
net = tf.keras.layers.Dense(1, activation='relu')(net)
model = tf.keras.models.Model(inputs=model_input, outputs=net)
model.compile(optimizer='adam',
              loss='msle',
              metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])


# 训练集和测试集生成
if __name__ == '__main__':
    num_trainsamples = 200000
    train_data = np.zeros((num_trainsamples, 3))
    for i in range(num_trainsamples):
        train_data[i][0] = random.uniform(0.0001, 0.3)
        train_data[i][1] = random.uniform(0.0001, 0.3)
        train_data[i][2] = 4/(train_data[i][0]*3)+1/train_data[i][1]
    X = train_data[:, 0:2]
    y = train_data[:, 2].reshape(num_trainsamples, 1)
    num_testsamples = 2000
    test_data = np.zeros((num_testsamples, 4))
    for i in range(num_testsamples):
        test_data[i][0] = random.uniform(0.1, 0.3)
        test_data[i][1] = random.uniform(0.1, 1.1)
        test_data[i][2] = 4/(test_data[i][0]*3)+1 / \
            test_data[i][1]  # true value in green
        
    logdir = './callbacks'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir,
                                     "mybestmodel.h5")

    callbacks = [
        keras.callbacks.TensorBoard(logdir),
        keras.callbacks.ModelCheckpoint(output_model_file,
                                        save_best_only=True),
    ]
    # model.load_weights(output_model_file)
    history = model.fit(X, y, epochs=10, shuffle=True, validation_data=(
        test_data[:, 0:2], test_data[:, 2].reshape(num_testsamples, 1)), callbacks=callbacks)
    # history = model.fit(X, y, epochs=10, validation_data=(
    #     test_data[:, 0:2], test_data[:, 2].reshape(num_testsamples, 1)), callbacks=callbacks)
    print('tag:done!')
    predictions = model.predict(test_data[:, 0:2])
    #test data的第三列才是预测的
    for i in range(num_testsamples):
        test_data[i][3] = predictions[i]

    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    # 将数据点分成三部分画，在颜色上有区分度
    ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c='g')
    ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 3], c='r')
    plt.legend(['real','predict'])
    plt.show()
