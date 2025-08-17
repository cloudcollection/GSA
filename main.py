import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据导入
predictstockdata = pd.read_excel("库存量.xlsx", header=None).values
predictdaysaledataS1 = pd.read_excel("未来90天销量预测结果.xlsx", header=None).values
cangkuxinxi = pd.read_excel("仓库数据.xlsx", header=None).values
dataguanlianduS3 = pd.read_excel("商品关联度.xlsx", header=None).values

# 数据预处理（使用原始数据）
total_stock = np.mean(predictstockdata[:, 1:4].astype(float), axis=1)  # 平均库存
sales_raw = np.mean(predictdaysaledataS1.astype(float), axis=1)  # 平均预测销量
capacity_raw = cangkuxinxi[:, 0].astype(float)  # 仓容
production_raw = cangkuxinxi[:, 1].astype(float)  # 产能
rental_raw = cangkuxinxi[:, 2].astype(float)  # 仓租
association_matrix = dataguanlianduS3.astype(float)

num_products = 350
num_warehouses = 140


# 核心函数定义
def decode_allocation(encoded_allocation):
    """将编码向量转换为分配矩阵"""
    allocation_matrix = np.zeros((num_products, num_warehouses))
    for i in range(num_products):
        warehouse_idx = int(round(encoded_allocation[i])) - 1
        if 0 <= warehouse_idx < num_warehouses:
            allocation_matrix[i, warehouse_idx] = 1
    return allocation_matrix


def compute_objective(allocation_vector):
    """计算适应度函数"""
    allocation_matrix = decode_allocation(allocation_vector)

    # 关联度得分
    association_score = np.sum((allocation_matrix.T @ association_matrix) * allocation_matrix.T)

    # 仓容计算
    warehouse_stock = allocation_matrix.T @ total_stock
    capacity_overflow = np.sum(np.maximum(warehouse_stock - capacity_raw, 0))
    capacity_util = np.sum(np.minimum(warehouse_stock, capacity_raw)) / np.sum(capacity_raw)

    # 产能计算
    warehouse_sales = allocation_matrix.T @ sales_raw
    production_overflow = np.sum(np.maximum(warehouse_sales - production_raw, 0))
    production_util = np.sum(np.minimum(warehouse_sales, production_raw)) / np.sum(production_raw)

    # 仓租成本
    used_warehouses = np.any(allocation_matrix, axis=0)
    rental_cost = np.sum(rental_raw[used_warehouses])

    # 惩罚系数
    PENALTY_WEIGHT = 50000

    # 目标函数权重
    w_association = 0.1
    w_capacity = 15000
    w_production = 15000
    w_rental = 1

    obj = (w_association * association_score +
           w_capacity * capacity_util +
           w_production * production_util -
           w_rental * rental_cost -
           PENALTY_WEIGHT * (capacity_overflow + production_overflow))

    return obj


# 遗传算法实现
class GeneticOptimizer:
    def __init__(self):
        self.pop_size = 350
        self.mutation_rate = 0.25
        self.elitism_rate = 0.15
        self.tournament_size = 7
        self.max_generations = 1500
        self.patience = 30

    def initialize_population(self):
        return np.random.randint(1, num_warehouses + 1,
                                 size=(self.pop_size, num_products))

    def tournament_selection(self, pop, fitness):
        indices = np.random.choice(len(pop), self.tournament_size, replace=False)
        return pop[indices[np.argmax(fitness[indices])]]

    def uniform_crossover(self, p1, p2):
        mask = np.random.rand(num_products) < 0.5
        return np.where(mask, p1, p2)

    def mutation(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation_point = np.random.randint(num_products)
            individual[mutation_point] = np.random.randint(1, num_warehouses + 1)
        return individual

    def evolve(self, pop, fitness):
        # 精英保留
        elite_size = int(self.elitism_rate * self.pop_size)
        elite_indices = np.argsort(fitness)[-elite_size:]
        new_pop = [pop[i] for i in elite_indices]

        # 生成后代
        while len(new_pop) < self.pop_size:
            p1 = self.tournament_selection(pop, fitness)
            p2 = self.tournament_selection(pop, fitness)
            child = self.uniform_crossover(p1, p2)
            child = self.mutation(child)
            new_pop.append(child)

        return np.array(new_pop)

    def optimize(self):
        start_time = time.time()
        population = self.initialize_population()
        best_solution = None
        best_fitness = -np.inf
        no_improve = 0
        history = []

        with tqdm(total=self.max_generations, desc="优化进度") as pbar:
            for gen in range(self.max_generations):
                # 评估种群
                fitness = np.array([compute_objective(ind) for ind in population])

                # 更新最佳解
                current_best = np.max(fitness)
                if current_best > best_fitness:
                    best_fitness = current_best
                    best_solution = population[np.argmax(fitness)]
                    no_improve = 0
                else:
                    no_improve += 1

                # 记录历史
                history.append(current_best)

                # 早停机制
                if no_improve >= self.patience:
                    print(f"\n早停在第{gen}代触发")
                    break

                # 进化种群
                population = self.evolve(population, fitness)
                pbar.update(1)

        print(f"\n优化完成，耗时 {time.time() - start_time:.2f}秒")
        return best_solution, best_fitness, history


# 执行优化
optimizer = GeneticOptimizer()
best_solution, best_fitness, history = optimizer.optimize()

# 结果分析
allocation_matrix = decode_allocation(best_solution)

# 计算实际指标
warehouse_stock = allocation_matrix.T @ total_stock
capacity_utilization = np.sum(np.minimum(warehouse_stock, capacity_raw)) / np.sum(capacity_raw)
warehouse_sales = allocation_matrix.T @ sales_raw
production_utilization = np.sum(np.minimum(warehouse_sales, production_raw)) / np.sum(production_raw)
used_warehouses = np.any(allocation_matrix, axis=0)
rental_cost = np.sum(rental_raw[used_warehouses])
association_score = np.sum((allocation_matrix.T @ association_matrix) * allocation_matrix.T)

print("\n最终结果:")
print(f"仓容利用率: {capacity_utilization * 100:.2f}%")
print(f"产能利用率: {production_utilization * 100:.2f}%")
print(f"仓租成本: {rental_cost:.2f}")
print(f"关联度得分: {association_score:.2f}")

# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history)
plt.title("适应度进化曲线")
plt.xlabel("迭代次数")
plt.ylabel("目标函数值")

plt.subplot(1, 2, 2)
plt.bar(range(num_warehouses), np.sum(allocation_matrix, axis=0))
plt.title("仓库使用情况")
plt.xlabel("仓库编号")
plt.ylabel("商品数量")
plt.tight_layout()
plt.show()

# 保存结果
pd.DataFrame({
    "商品编号": range(num_products),
    "分配仓库": best_solution.astype(int)
}).to_excel("最优分配方案.xlsx", index=False)
print("结果已保存到 最优分配方案.xlsx")