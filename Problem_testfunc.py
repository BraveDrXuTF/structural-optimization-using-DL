# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import math
from test_func import Ackley_cal, Rastrigin_cal, Sphere_cal


class MyProblem_Ackley(ea.Problem):  # 继承Problem父类

    def __init__(self):

        name = 'MyProblem'  # 初始化name(函数名称,可以随意设置)
        M = 1  # 初始化M(目标维数)
        maxormins = [1]  # 初始化目标最小最大化标记列表,1:min;-1:max
        Dim = 2  # 初始化Dim(决策变量维数)
        varTypes = [0] * Dim  # 初始化决策变量类型,0:连续;1:离散
        # lb = [-10, -10]  # 决策变量下界
        # ub = [10, 10]  # 决策变量上界
        lb = [-1, -1]  # 决策变量下界
        ub = [1, 1]  # 决策变量上界        
        lbin = [1, 1]  # 决策变量下边界
        ubin = [1, 1]  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,
                            ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数,pop为传入的种群对象

        Vars = pop.Phen  # 得到决策变量矩阵
        x1 = Vars[:, [0]]  # 取出第一列得到所有个体的x1组成的列向量
        x2 = Vars[:, [1]]  # 取出第二列得到所有个体的x2组成的列向量
        # 计算目标函数值,赋值给pop种群对象的ObjV属性
        pop.ObjV = Ackley_cal(x1, x2)
        # 采用可行性法则处理约束,生成种群个体违反约束程度矩阵
        pop.CV = None # 第三个约束 通过Ackley-1无解，看来通过这种方法无法代替DM

class MyProblem_Rastrigin(ea.Problem):  # 继承Problem父类

    def __init__(self):

        name = 'MyProblem'  # 初始化name(函数名称,可以随意设置)
        M = 1  # 初始化M(目标维数)
        maxormins = [1]  # 初始化目标最小最大化标记列表,1:min;-1:max
        Dim = 2  # 初始化Dim(决策变量维数)
        varTypes = [0] * Dim  # 初始化决策变量类型,0:连续;1:离散
        lb = [-10, -10]  # 决策变量下界
        ub = [10, 10]  # 决策变量上界
        # lb = [-1, -1]  # 决策变量下界
        # ub = [1, 1]  # 决策变量上界        
        lbin = [1, 1]  # 决策变量下边界
        ubin = [1, 1]  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,
                            ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数,pop为传入的种群对象

        Vars = pop.Phen  # 得到决策变量矩阵
        x1 = Vars[:, [0]]  # 取出第一列得到所有个体的x1组成的列向量
        x2 = Vars[:, [1]]  # 取出第二列得到所有个体的x2组成的列向量
        # 计算目标函数值,赋值给pop种群对象的ObjV属性
        pop.ObjV = Rastrigin_cal(x1, x2)
        # 采用可行性法则处理约束,生成种群个体违反约束程度矩阵
        pop.CV = None # 第三个约束 通过Ackley-1无解，看来通过这种方法无法代替DM


class MyProblem_Sphere(ea.Problem):  # 继承Problem父类

    def __init__(self):

        name = 'MyProblem'  # 初始化name(函数名称,可以随意设置)
        M = 1  # 初始化M(目标维数)
        maxormins = [1]  # 初始化目标最小最大化标记列表,1:min;-1:max
        Dim = 2  # 初始化Dim(决策变量维数)
        varTypes = [0] * Dim  # 初始化决策变量类型,0:连续;1:离散
        # lb = [-10, -10]  # 决策变量下界
        # ub = [10, 10]  # 决策变量上界
        lb = [-1, -1]  # 决策变量下界
        ub = [1, 1]  # 决策变量上界        
        lbin = [1, 1]  # 决策变量下边界
        ubin = [1, 1]  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,
                            ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数,pop为传入的种群对象

        Vars = pop.Phen  # 得到决策变量矩阵
        x1 = Vars[:, [0]]  # 取出第一列得到所有个体的x1组成的列向量
        x2 = Vars[:, [1]]  # 取出第二列得到所有个体的x2组成的列向量
        # 计算目标函数值,赋值给pop种群对象的ObjV属性
        pop.ObjV = Sphere_cal(x1, x2)
        # 采用可行性法则处理约束,生成种群个体违反约束程度矩阵
        pop.CV = None # 第三个约束 通过Ackley-1无解，看来通过这种方法无法代替DM