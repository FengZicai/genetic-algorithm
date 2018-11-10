# 把每个循环里面最好的value值保留下来，画出曲线
import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt


def function_GA(a, b, NP, NG, pc, pm, eps):
    dict = {'0.0': 1.0, '1.0': 0.0}
    best_value = []
    best_fitness = []
    gen = []
    le1 = math.ceil(math.log(((b - a)/eps+1), 2))
    le2 = math.ceil(math.log(((b - a)/eps + 1), 2))
    leng = le1 + le2

    parent = np.zeros((NP, leng))
    offspring = np.zeros((NP, leng))
    for i in range(NP):
        parent[i, :] = initial(leng)
    dec = decode(parent, le1, le2)
    _, fit = fitness(dec)
    for k in range(NG):
        # 轮盘赌选择
        sumfit = np.sum(fit)
        px = fit/sumfit
        ppx = []
        q = 0
        for i in range(NP):
            q = q + px[i]
            ppx.append(q)

        for i in range(NP):
            sita = rd.random()
            for n in range(NP):
                if sita <= ppx[n]:
                    select_father = n
                    break
            select_mother = math.floor(rd.random() * (NP - 1) + 1)
            cut_position = math.floor(rd.random() * (leng - 2) + 1)
            r1 = rd.random()

            # crossover
            if r1 <= pc:
                offspring[i, 0:cut_position] = parent[select_father, 0:cut_position]
                offspring[i, cut_position:leng] = parent[select_mother, cut_position:leng]
                r2 = rd.random()
                # mutation
                if r2 <= pm:
                    mutation_position = math.floor(rd.random() * (leng - 1) + 1)
                    offspring[i, mutation_position] = dict[str(offspring[i, mutation_position])]
            else:
                offspring[i, :] = parent[select_father, :]
        # 子代和父代合在一起进行选择
        population = np.r_[parent, offspring]
        population_decode = decode(population, le1, le2)
        _, fit = fitness(population_decode)
        result = [[fit[j], j] for j in range(len(population))]
        result.sort(reverse=True)
        result = result[: NP]
        parent = np.array([population[result[k][1]] for k in range(len(result))])
        # 计算每代里最好的函数值保存下来
        value_decode = decode(parent, le1, le2)
        value, _ = fitness(value_decode)
        best_value.append(value[0])
        best_fitness.append(result[0][0])
        gen.append(k)

    bestx = decode(parent, le1, le2)[0, :]
    return bestx, best_value, gen


def initial(length):
    result = np.zeros(length)
    for i in range(length):
        r = rd.random()
        result[i] = round(r)
    return result


def fitness(x):
    # y = 100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (x[:, 0] - 1) ** 2  # F2
    y = (1 + ((x[:, 0] + x[:, 1] + 1) ** 2) * (19 - 14 * x[:, 0] + 3 * x[:, 0] ** 2 - 14 * x[:, 1] + 6 * x[:, 0] * x[:, 1] + 3 * x[:, 1] ** 2)) * (30 + ((2 * x[:, 0] - 3 * x[:, 1]) ** 2) * (18 - 32 * x[:, 0] + 12 * x[:, 0] ** 2 + 48 * x[:, 1] - 36 * x[:, 0] * x[:, 1] + 27 * x[:, 1]**2))  # F3
    # y = (x[:, 0]**2 + x[:, 1]-11)**2 + (x[:, 0] + x[:, 1]**2-7)**2  # F4
    # y = 4*x[:, 0]**2 -2.1*x[:, 0]**4 +(1/3)*x[:, 0]**6 + x[:, 0]* x[:, 1] -4*x[:, 1]**2 +4*x[:, 1]**4  # F5
    # y = x[:, 0]**2 + x[:, 1]**2 # F1
    f = - y + 400000
    return y, f


def decode(x, *length):
    k = np.array(x[:, 0:length[0]])
    base = np.logspace(0, length[0]-1, length[0], base=2)[::-1]
    w = np.dot(k, base)
    w = np.transpose(a + w*(b-a)/(2 ** length[0] - 1))
    m = np.array(x[:, length[0]:length[0]+length[1]])
    base = np.logspace(0, length[1] - 1, length[1], base=2)[::-1]
    v = np.dot(m, base)
    v = np.transpose(a + v * (b - a) / (2 ** length[0] - 1))
    x_k = np.c_[w, v]
    return x_k


if __name__ == '__main__':

    a = -2  # 自变量下界: a
    b = 2  # 自变量上界: b
    NP = 500    # 种群个体数: NP
    NG = 40    # 最大进化代数: NG
    pc = 0.9    # 杂交概率: pc
    pm = 0.04   # 自变量概率: pm
    eps = 0.000001   # 自变量离散精度: eps
    be_val = []
    for i in range(20):   # 运行20次代码
        bex, be, g = function_GA(a, b, NP, NG, pc, pm, eps)
        be_val.append(be[-1])
        print(bex)
        plt.plot(g, be, linewidth=1)

    sum = np.array(be_val).sum()
    mean = sum/len(be_val)
    var = np.var(np.array(be_val))
    print(be_val)
    print(mean, var)
    plt.title('F3 function')
    plt.xlabel('generations')
    plt.ylabel('F3 functional value')
    plt.show()

