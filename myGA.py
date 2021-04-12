import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt

# 可以算出最小值和真实最小值的差别
DNA_SIZE = 24
POP_SIZE = 200
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 10
X_BOUND = [0.01, 0.5]
Y_BOUND = [0.01, 0.5]
MODEL_URL = 'mybestmodel.h5'

# 绘制直方图一类的DM；“树形结构”；或许根本不需要什么算法？


def F(x, y):
    return 4/(3*x)+1/y


def F_net(x, y, model_url=MODEL_URL):
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)
    x_net = np.concatenate((x, y), axis=1)
    model = keras.models.load_model(model_url)
    return np.squeeze(model.predict(x_net))


def plot_3d(ax, model_url=MODEL_URL):

    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)

    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_zlim(-20, 20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()


def get_fitness(pop, model_url=MODEL_URL):
    x, y = translateDNA(pop)
    pred = -F_net(x, y)  # the smaller F(x,y), the bigger fitness
    # 适应度约束法
    con = (4*x/sqrt(3)+sqrt(3)*y)
    pred[con > 1] = -10000

    return (pred - np.min(pred)) + 1e-3


def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:, 1::2]  # 奇数列表示X
    y_pop = pop[:, ::2]  # 偶数列表示y

    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    x = x_pop.dot(2**np.arange(DNA_SIZE)
                  [::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    y = y_pop.dot(2**np.arange(DNA_SIZE)
                  [::-1])/float(2**DNA_SIZE-1)*(Y_BOUND[1]-Y_BOUND[0])+Y_BOUND[0]
    return x, y


def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(
                low=0, high=DNA_SIZE*2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


def mutation(child, MUTATION_RATE=0.3):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness)/(fitness.sum()))
    return pop[idx]


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
    predict = F(x[max_fitness_index], y[max_fitness_index])
    real = F(sqrt(3)/7, sqrt(3)/7)
    print(real, predict, abs(real-predict)/real)


if __name__ == "__main__":
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    plot_3d(ax)

    # matrix (POP_SIZE, DNA_SIZE)
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*2))
    for _ in range(N_GENERATIONS):  # 迭代N代
        x, y = translateDNA(pop)
        if 'sca' in locals():
            sca.remove()
        sca = ax.scatter(x, y, F_net(x, y), c='black', marker='o')
        plt.show()
        plt.pause(0.05)
        pop_son = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))

        # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix

        # 筛选获得符合约束条件的坐标点
        # pop ->popson ;popson+pop->pop

        # 采用这种方法约束（不满足直接删去）会消除大量的点，不好
        # x_con, y_con = translateDNA(pop)
        # con = (4*x_con/sqrt(3)+sqrt(3)*y_con)
        # pop = pop[con <= 1]

        # x_son_con, y_son_con = translateDNA(pop_son)
        # con_son = (4*x_son_con/sqrt(3)+sqrt(3)*y_son_con)
        # pop_son = pop_son[con_son <= 1]

        # pop = np.concatenate((pop, pop_son), axis=0)
        # fitness = get_fitness(pop)
        # # argsort默认从小到大排序,因此取-fitness
        # pop = pop[np.argsort(-fitness), :]

        # if pop.shape[0] > POP_SIZE:
        #     pop = pop[:POP_SIZE]
        #     fitness = fitness[:POP_SIZE]

        pop = np.concatenate((pop, pop_son), axis=0)
        fitness = get_fitness(pop)
        # argsort默认从小到大排序,因此取-fitness
        pop = pop[np.argsort(-fitness), :]
        pop = pop[:POP_SIZE]
        fitness = fitness[:POP_SIZE]
        pop = select(pop, fitness)  # 选择生成新的种群

    print_info(pop)
    plt.ioff()
    plot_3d(ax)
