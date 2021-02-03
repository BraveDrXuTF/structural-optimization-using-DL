# -*- coding: utf-8 -*-
"""main.py"""
import numpy as np
import geatpy as ea  # import geatpy
from Myproblem import MyProblem  # 导入自定义问题接口
"""============================实例化问题对象========================"""
problem = MyProblem()  # 实例化问题对象
"""==============================种群设置==========================="""
Encoding = 'RI'  # 编码方式
NIND = 50  # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
                  problem.borders)  # 创建区域描述器
population = ea.Population(Encoding, Field, NIND)
# 实例化种群对象(此时种群还没被真正初始化, 仅仅是生成一个种群对象)
"""===========================算法参数设置=========================="""
myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
# 实例化一个算法模板对象
myAlgorithm.MAXGEN = 100  # 最大遗传代数
myAlgorithm.mutOper.F = 0.5  # 设置差分进化的变异缩放因子
myAlgorithm.recOper.XOVR = 0.5  # 设置交叉概率
myAlgorithm.drawing = 2  # 设置绘图方式
"""=====================调用算法模板进行种群进化====================="""
myAlgorithm.run()
