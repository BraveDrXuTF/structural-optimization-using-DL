# -*- coding: utf-8 -*-
"""main.py"""
import numpy as np
import geatpy as ea  # import geatpy
from Problem_testfunc import MyProblem_Ackley, MyProblem_Rastrigin, MyProblem_Sphere  # 导入自定义问题接口
"""=======================实例化问题对象==========================="""
# problem = MyProblem_Ackley()  # 实例化问题对象
# problem = MyProblem_Rastrigin()  # 实例化问题对象
problem = MyProblem_Sphere()
"""=========================种群设置=============================="""
Encoding = 'RI'  # 编码方式
NIND = 25  # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
                  problem.borders)  # 创建区域描述器
population = ea.Population(Encoding, Field, NIND)
"""=========================算法参数设置============================"""
myAlgorithm = ea.moea_NSGA2_templet(problem, population)
myAlgorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
myAlgorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
myAlgorithm.MAXGEN = 10  # 最大进化代数
myAlgorithm.logTras = 1
myAlgorithm.verbose = False  # 设置是否打印输出日志信息
myAlgorithm.drawing = 1
"""==========================调用算法模板进行种群进化==============
调用run执行算法模板,得到帕累托最优解集NDSet以及最后一代种群。
NDSet是一个种群类Population的对象。
NDSet.ObjV为最优解个体的目标函数值;NDSet.Phen为对应的决策变量值。
详见Population.py中关于种群类的定义。
"""
[NDSet, population] = myAlgorithm.run()
NDSet.save()  # 把非支配种群的信息保存到文件中
"""===========================输出结果========================"""
print('用时:%s 秒' % myAlgorithm.passTime)
print('非支配个体数:%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解!')
if myAlgorithm.log is not None and NDSet.sizes != 0:

    print('GD', myAlgorithm.log['gen'][-1])
    print('IGD', myAlgorithm.log['eval'][-1])
    print('HV', myAlgorithm.log['hv'][-1])
    print('Spacing', myAlgorithm.log['spacing'][-1])
# """======================进化过程指标追踪分析=================="""
# metricName = [['gen'], ['hv']]
# Metrics = np.array([myAlgorithm.log[metricName[i][0]]
#                     for i in range(len(metricName))]).T
# # 绘制指标追踪分析图
# ea.trcplot(Metrics, labels=metricName, titles=metricName)
