import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 数据预处理 =====================
def load_and_preprocess():
    """加载并预处理所有输入数据"""
    predictstockdata = pd.read_excel("库存量.xlsx", header=None).values
    predictdaysaledataS1 = pd.read_excel("未来90天销量预测结果.xlsx", header=None).values
    cangkuxinxi = pd.read_excel("仓库数据.xlsx", header=None).values
    dataguanlianduS3 = pd.read_excel("商品关联度.xlsx", header=None).values

    # 数据预处理
    total_stock = np.mean(predictstockdata[:, 1:4].astype(float), axis=1)
    sales_raw = np.mean(predictdaysaledataS1.astype(float), axis=1)

    # 数据归一化处理
    def scale_data(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 10000

    return {
        'total_stock': scale_data(total_stock),
        'sales_raw': scale_data(sales_raw),
        'capacity_raw': scale_data(cangkuxinxi[:, 0].astype(float)),
        'production_raw': scale_data(cangkuxinxi[:, 1].astype(float)),
        'rental_raw': cangkuxinxi[:, 2].astype(float),
        'association_matrix': dataguanlianduS3.astype(float),
        'num_products': 350,
        'num_warehouses': 140
    }


# ===================== 核心算法 =====================
class AdvancedGeneticOptimizer:
    def __init__(self, params):
        self.pop_size = 300
        self.mutation_rate = 0.35
        self.elitism_rate = 0.1
        self.tournament_size = 5
        self.max_generations = 2000
        self.patience = 50
        self.params = params
        self.best_solution = None
        self.history = []

    def decode_allocation(self, encoded_allocation):
        allocation_matrix = np.zeros((self.params['num_products'],
                                      self.params['num_warehouses']))
        for i in range(self.params['num_products']):
            warehouse_idx = int(encoded_allocation[i]) - 1
            if 0 <= warehouse_idx < self.params['num_warehouses']:
                allocation_matrix[i, warehouse_idx] = 1
        return allocation_matrix

    def compute_objective(self, individual):
        allocation_matrix = self.decode_allocation(individual)

        # 关联度计算
        association_score = np.sum(
            (allocation_matrix.T @ self.params['association_matrix']) * allocation_matrix.T
        )

        # 仓容计算
        warehouse_stock = allocation_matrix.T @ self.params['total_stock']
        capacity_overflow = np.sum(np.maximum(warehouse_stock - self.params['capacity_raw'], 0))
        capacity_util = np.mean(np.minimum(warehouse_stock / self.params['capacity_raw'], 1.0))

        # 产能计算
        warehouse_sales = allocation_matrix.T @ self.params['sales_raw']
        production_overflow = np.sum(np.maximum(warehouse_sales - self.params['production_raw'], 0))
        production_util = np.mean(np.minimum(warehouse_sales / self.params['production_raw'], 1.0))

        # 成本计算
        used_warehouses = np.any(allocation_matrix, axis=0)
        rental_cost = np.sum(self.params['rental_raw'][used_warehouses])

        # 动态惩罚机制
        PENALTY_BASE = 10000
        capacity_penalty = PENALTY_BASE * (1 + capacity_overflow / np.sum(self.params['capacity_raw']))
        production_penalty = PENALTY_BASE * (1 + production_overflow / np.sum(self.params['production_raw']))

        # 目标函数权重
        weights = {
            'association': 0.5,
            'capacity': 15000,
            'production': 15000,
            'rental': 1.0,
            'warehouse': 1000
        }

        objective = (
                weights['association'] * association_score +
                weights['capacity'] * capacity_util +
                weights['production'] * production_util -
                weights['rental'] * rental_cost +
                weights['warehouse'] * np.sum(used_warehouses) -
                (capacity_penalty + production_penalty)
        )
        return objective

    def initialize_population(self):
        """修复后的智能初始化方法"""
        base_pop = np.random.randint(1, self.params['num_warehouses'] + 1,
                                     size=(self.pop_size, self.params['num_products']))

        # 改进的启发式初始化
        sorted_wh = np.argsort(-self.params['capacity_raw'])  # 按容量降序
        cycle_wh = np.resize(sorted_wh, self.params['num_products'])  # 循环填充

        for i in range(10):
            base_pop[i] = cycle_wh % self.params['num_warehouses'] + 1

        return base_pop

    def evolve_population(self, pop, fitness):
        elite_size = int(self.elitism_rate * self.pop_size)
        elite_indices = np.argsort(fitness)[-elite_size:]
        new_pop = [pop[i] for i in elite_indices]

        current_mutation_rate = self.mutation_rate * (1 + len(self.history) / self.max_generations)

        while len(new_pop) < self.pop_size:
            parents = [
                pop[np.random.choice(np.argsort(fitness)[-self.pop_size // 2:])]
                for _ in range(2)
            ]
            mask = np.random.rand(self.params['num_products']) < 0.5
            child = np.where(mask, parents[0], parents[1])

            if np.random.rand() < current_mutation_rate:
                mut_points = np.random.choice(
                    self.params['num_products'],
                    size=int(0.1 * self.params['num_products']),
                    replace=False)
                child[mut_points] = np.random.randint(
                    1, self.params['num_warehouses'] + 1,
                    size=len(mut_points))

            new_pop.append(child)

        return np.array(new_pop)[:self.pop_size]

    def optimize(self):
        start_time = time.time()
        population = self.initialize_population()
        best_fitness = -np.inf
        no_improve = 0

        with tqdm(total=self.max_generations, desc="优化进度") as pbar:
            for gen in range(self.max_generations):
                fitness = np.array([self.compute_objective(ind) for ind in population])

                current_max = np.max(fitness)
                if current_max > best_fitness:
                    best_fitness = current_max
                    self.best_solution = population[np.argmax(fitness)]
                    no_improve = 0
                else:
                    no_improve += 1

                self.history.append(current_max)

                if no_improve >= self.patience:
                    print(f"\n早停触发于第{gen}代")
                    break

                population = self.evolve_population(population, fitness)
                pbar.update(1)

        print(f"\n总耗时: {time.time() - start_time:.2f}秒")
        return self.best_solution


# ===================== 结果分析 =====================
def analyze_results(solution, params):
    allocation_matrix = np.zeros((params['num_products'], params['num_warehouses']))
    for i in range(params['num_products']):
        warehouse_idx = int(solution[i]) - 1
        if 0 <= warehouse_idx < params['num_warehouses']:
            allocation_matrix[i, warehouse_idx] = 1

    warehouse_stock = allocation_matrix.T @ params['total_stock']
    warehouse_sales = allocation_matrix.T @ params['sales_raw']
    used_warehouses = np.any(allocation_matrix, axis=0)

    print("\n=== 最终优化结果 ===")
    print(f"仓容利用率: {np.mean(warehouse_stock / params['capacity_raw']):.2%}")
    print(f"产能利用率: {np.mean(warehouse_sales / params['production_raw']):.2%}")
    print(f"使用仓库数: {np.sum(used_warehouses)}/{params['num_warehouses']}")
    print(f"仓租成本: {np.sum(params['rental_raw'][used_warehouses]):,.2f}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(self.history)
    plt.title("目标函数收敛曲线")

    plt.subplot(1, 3, 2)
    plt.scatter(params['capacity_raw'], warehouse_stock, alpha=0.6)
    plt.plot([0, max(params['capacity_raw'])], [0, max(params['capacity_raw'])], 'r--')
    plt.title("仓容利用分布")

    plt.subplot(1, 3, 3)
    plt.hist(np.sum(allocation_matrix, axis=0), bins=20)
    plt.title("仓库商品分布")

    plt.tight_layout()
    plt.show()

    pd.DataFrame({
        '商品编号': range(params['num_products']),
        '分配仓库': solution.astype(int)
    }).to_excel("最优分配方案_final.xlsx", index=False)


if __name__ == "__main__":
    params = load_and_preprocess()
    optimizer = AdvancedGeneticOptimizer(params)
    best_solution = optimizer.optimize()
    analyze_results(best_solution, params)