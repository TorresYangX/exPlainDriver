import numpy as np
from scipy.optimize import minimize

# 定义逻辑规则的数学表达式
rules = [
    lambda stop_sign, solid_red_light, no_left_turn_sign, stop, make_left_turn: 1 - stop_sign * (1 - stop),
    lambda stop_sign, solid_red_light, no_left_turn_sign, stop, make_left_turn: 1 - solid_red_light * (1 - stop),
    lambda stop_sign, solid_red_light, no_left_turn_sign, stop, make_left_turn: 1 - no_left_turn_sign * make_left_turn
]

# 初始化权重
weights = np.random.rand(len(rules))

# 训练数据
data = [
    [1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1]
]

def calculate_potential(weights, stop_sign, solid_red_light, no_left_turn_sign, stop, make_left_turn):
    potential = 0
    for w, rule in zip(weights, rules):
        potential += w * rule(stop_sign, solid_red_light, no_left_turn_sign, stop, make_left_turn)
    return potential

def calculate_probability(weights, stop_sign, solid_red_light, no_left_turn_sign):
    potentials = []
    for stop, make_left_turn in [(1, 0), (0, 1)]:  # 只考虑互斥的合法组合
        potential = calculate_potential(weights, stop_sign, solid_red_light, no_left_turn_sign, stop, make_left_turn)
        potentials.append(np.exp(potential))
    total_potential = sum(potentials)
    probabilities = [p / total_potential for p in potentials]
    return probabilities

def log_likelihood(weights, data):
    ll = 0
    for sample in data:
        stop_sign, solid_red_light, no_left_turn_sign, stop, make_left_turn = sample
        probabilities = calculate_probability(weights, stop_sign, solid_red_light, no_left_turn_sign)
        if stop == 1 and make_left_turn == 0:
            index = 0
        elif stop == 0 and make_left_turn == 1:
            index = 1
        else:
            continue  # 忽略无效数据
        ll += np.log(probabilities[index])
    return -ll  # 我们使用负对数似然，因为大多数优化函数是进行最小化

# 优化权重
result = minimize(log_likelihood, weights, args=(data,), method='L-BFGS-B')
optimal_weights = result.x

# 推理
def infer(optimal_weights, stop_sign, solid_red_light, no_left_turn_sign):
    probabilities = calculate_probability(optimal_weights, stop_sign, solid_red_light, no_left_turn_sign)
    if probabilities[0] > probabilities[1]:
        print("Action: Stop")
    else:
        print("Action: MakeLeftTurn")

# 示例推理
infer(optimal_weights, 1, 0, 0)
infer(optimal_weights, 0, 1, 0)
infer(optimal_weights, 0, 0, 1)
