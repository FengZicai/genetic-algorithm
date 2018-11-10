import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt


def function_GA():
    best_fitness = []
    gen = []
    bestx = []
    best_value = []
    # population = []

    population = np.zeros((NP, leng))
    for i in range(NP):

        population[i, :] = [np.random.uniform(a, b) for j in range(leng)]
    for k in range(NG):
        fit = fitness(population)
        # Tournament Selection
        parent = []
        for i in range(int(NP/2)):
            list_t = rd.sample(range(0, len(population)), 2)
            if fit[list_t[0]] > fit[list_t[1]]:
                parent.append(population[list_t[0], :])
            else:
                parent.append(population[list_t[1], :])
        parent = np.array(parent)

        offspring_c = []
        x1 = np.zeros(leng)
        x2 = np.zeros(leng)
        #crossover
        for i in range(len(parent)):
            list_c = rd.sample(range(0, len(parent)), 2)
            if np.random.random() < pc:
                uj = np.random.random()
                if uj <= 0.5:
                    rj = (2 * uj) ** (1 / (nc + 1))
                else:
                    rj = (1 / (2 * (1 - uj))) ** (1 / (nc + 1))
                for j in range(leng):
                    x1[j] = 0.5 * ((1 + rj) * parent[list_c[0], j] + (1 - rj) * parent[list_c[1], j])
                    x2[j] = 0.5 * ((1 - rj) * parent[list_c[0], j] + (1 + rj) * parent[list_c[1], j])
            else:
                x1 = parent[list_c[0], :]
                x2 = parent[list_c[1], :]
            offspring_c.append(x1)
            offspring_c.append(x2)
        offspring_c = np.array(check_value(offspring_c))

        offspring_m = offspring_c
        # mutation
        for i in range(len(offspring_m)):
            if np.random.random() < pm:
                for j in range(leng):
                    if np.random.random() < pmm:
                        vk = offspring_m[i, j]
                        uk = b
                        lk = a
                        sita1 = (vk - lk) / (uk - lk)
                        sita2 = (uk - vk) / (uk - lk)
                        u = np.random.random()
                        if u <= 0.5:
                            sita = (2 * u + (1 - 2 * u) * ((1 - sita1) ** (nm + 1))) ** (1 / (nm + 1)) - 1
                        else:
                            sita = 1 - (2 * (1 - u) + 2 * (u - 0.5) * ((1 - sita2) ** (nm + 1))) ** (1 / (nm + 1))
                        vk = vk + sita
                        offspring_m[i, j] = vk
        offspring_m = np.array(check_value(offspring_m))

        population = np.r_[parent, offspring_m]
        fit = fitness(population)
        result = [[fit[j], j] for j in range(len(population))]
        result.sort(reverse=True)
        result = result[: NP]
        population = np.array([population[result[k][1]] for k in range(len(result))])
        #计算每代里最好的函数值保存下来
        value = test_function(population)
        best_value.append(value[0])
        best_fitness.append(result[0][0])
        gen.append(k)
        bestx = population[0, :]
    return bestx, best_value, gen


def test_function(x):
    y = 0
    F = 0
    # for xi in x:  # F9
    #     F += xi**2

    # for xi in x: #F10
    #     F += (np.vectorize(math.floor)(xi + 0.5))**2

    # for i in range(len(x)): #F11
    #     for j in range(1, i+1):
    #         y += x[j]
    #     F += y**2

    # for xi in x:  #F12
    #     F += abs(xi)
    #     y *= abs(xi)
    # F = F + y

    for xi in x:
        F += xi**2 - 10 * np.vectorize(math.cos)(2*math.pi*xi)
    F = F + len(x) * 10

    return F


def fitness(x): #x仍然以二维矩阵分表种群数量和n元问题
    fitness = []
    for i in x:
        fitness.append(-test_function(i))
    return fitness


def check_value(x):
    for i in x:
        for j in range(len(i)):
            if i[j] < a:
                i[j] = a
            if i[j] > b:
                i[j] = b
    return x


if __name__ == '__main__':
    a = -5.12  # 自变量下界
    b = 5.12   # 自变量上界
    NP = 400  # 种群个体数:
    NG = 2000  # 最大进化代数
    pc = 0.9   # 杂交概率
    pm = 0.7  # 自变量概率
    pmm = 0.1
    # 目标函数取最大值时的自变量值：xm
    # 目标函数的最大值: fv
    nc = 2
    nm = 5
    leng = 20 #实数编码的变量元数
    be_val = []
    for i in range(20):  # 运行20次代码
        bestx, be, g = function_GA()
        be_val.append(be[-1])
        print(bestx)
        plt.plot(g, be, linewidth=1)
    sum = np.array(be_val).sum()
    mean = sum / len(be_val)
    var = np.var(np.array(be_val))
    print(be_val)
    print(mean, var)
    plt.title('F13 function')
    plt.xlabel('generations')
    plt.ylabel('F13 functional value')
    plt.show()

