# -*- coding: utf-8 -*-
"""main.py"""
import geatpy as ea # import geatpy
from Myproblem import MyProblem

"""============================实例化问题对象========================"""
problem = MyProblem() #
# 实例化问题对象
"""==============================种群设置==========================="""
Encoding = 'RI' # 编码方式
NIND = 500 # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
problem.borders) # 创建区域描述器
population = ea.Population(Encoding, Field, NIND) #
"""===========================算法参数设置=========================="""
myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population) #
myAlgorithm.MAXGEN = 50 # 最大进化代数
myAlgorithm.mutOper.F = 0.9 # 差分进化中的参数F
myAlgorithm.recOper.XOVR = 0.7 # 设置交叉概率
myAlgorithm.logTras = 1 #
myAlgorithm.verbose = True # 设置是否打印输出日志信息
myAlgorithm.drawing = 1 #
"""==========================调用算法模板进行种群进化==============="""
[BestIndi, population] = myAlgorithm.run()
BestIndi.save() # 把最优个体的信息保存到文件中
"""=================================输出结果======================="""
print('评价次数:%s' % myAlgorithm.evalsNum)
print('时间已过 %s 秒' % myAlgorithm.passTime)
if BestIndi.sizes != 0:
    print('最优的目标函数值为:%s' % BestIndi.ObjV[0][0])
    print('最优的控制变量值为:')
    for i in range(BestIndi.Phen.shape[1]):
        print(BestIndi.Phen[0, i])
else:
    print('没找到可行解。')







# 旧版 geatpy
# # 输出结果
# best_gen = np.argmax(obj_trace[:, 1])  # 记录最优种群是在哪一代
# best_ObjV = obj_trace[best_gen, 1]
# print('最优的目标函数值为:%s' % (best_ObjV))
# print('最优的决策变量值为:')
# for i in range(var_trace.shape[1]):
#     print(var_trace[best_gen, i])
# print('有效进化代数:%s' % (obj_trace.shape[0]))
# print('最优的一代是第 %s 代' % (best_gen + 1))
# print('评价次数:%s' % (myAlgorithm.evalsNum))
# print('时间已过 %s 秒' % (myAlgorithm.passTime))
