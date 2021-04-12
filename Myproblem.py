# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import math
from tensorflow import keras
import tensorflow as tf
# MODEL_URL = '/home/xutengfei/jiegouyouhua/callbacks/mybestmodel.h5'
MODEL_URL = 'mybestmodel.h5'
MODEL_URL2 = '/home/xutengfei/jiegouyouhua/callbacks_15yueshu/mybestmodel15_yueshu.h5'
# xsize = 2
# model_input = tf.keras.layers.Input(shape=(xsize,))
# net = tf.keras.layers.Dense(4*xsize, activation='relu')(model_input)
# net = tf.keras.layers.Dense(2*xsize, activation='relu')(net)
# net = tf.keras.layers.Dense(1, activation='relu')(net)
# model = tf.keras.models.Model(inputs=model_input, outputs=net)

# model.compile(optimizer='adam',
#               loss='msle',
#               metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])

# model.load_weights(output_model_file)

# 这种写法更简洁
model = keras.models.load_model(MODEL_URL)
model_yueshu = keras.models.load_model(MODEL_URL2)

class MyProblem(ea.Problem):  # 继承Problem父类

    def __init__(self):

        name = 'MyProblem'  # 初始化name(函数名称,可以随意设置)
        M = 1  # 初始化M(目标维数)
        maxormins = [1]  # 初始化目标最小最大化标记列表,1:min;-1:max
        Dim = 2  # 初始化Dim(决策变量维数)
        varTypes = [0] * Dim  # 初始化决策变量类型,0:连续;1:离散
        lb = [0.1, 0.1]  # 决策变量下界
        ub = [0.5, 0.5]  # 决策变量上界
        lbin = [0, 0]  # 决策变量下边界
        ubin = [1, 1]  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,
                            ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数,pop为传入的种群对象

        Vars = pop.Phen  # 得到决策变量矩阵
        x1 = Vars[:, [0]]  # 取出第一列得到所有个体的x1组成的列向量
        x2 = Vars[:, [1]]  # 取出第二列得到所有个体的x2组成的列向量
        # # 计算目标函数值,赋值给pop种群对象的ObjV属性
        pop.ObjV = model.predict(Vars)
        # pop.ObjV = 4/(3*x1)+1/(x2)
        # 采用可行性法则处理约束,生成种群个体违反约束程度矩阵
        pop.CV = np.hstack([model_yueshu(Vars)-1])  # 第三个约束
        # pop.CV = np.hstack([4*x1/math.sqrt(3)+math.sqrt(3)*x2-1])  # 第三个约束        






