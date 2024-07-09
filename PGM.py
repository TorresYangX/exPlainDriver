import numpy as np
from scipy.optimize import minimize
import pickle

# 定义逻辑规则的数学表达式
rules = [
    lambda *args: 1 - args[0] * (1 - args[18]),  # SolidRedLight → Stop
    lambda *args: 1 - args[1] * (1 - args[19]),  # SolidGreenLight → Accelerate
    lambda *args: 1 - args[2] * args[16] * (1 - args[18]),  # SolidYellowLight ∧ IntersectionAhead → Stop
    lambda *args: 1 - args[3] * args[16] * (1 - args[18]),  # YellowLeftArrowLight ∧ IntersectionAhead → Stop
    lambda *args: 1 - args[20] * (1 - args[4]),  # MakeLeftTurn → GreenLeftArrowLight
    lambda *args: 1 - args[5] * args[20],  # RedLeftArrowLight → ¬MakeLeftTurn
    lambda *args: 1 - args[6] * (1 - args[21]),  # MergingTrafficSign → Decelerate
    lambda *args: 1 - args[7] * (1 - args[18]),  # WrongWaySign → Stop
    lambda *args: 1 - args[8] * (1 - args[23]),  # KeepRightSign → ChangeToRightLane
    lambda *args: 1 - args[9] * args[20],  # NoLeftTurnSign → ¬MakeLeftTurn
    lambda *args: 1 - args[10] * args[22],  # NoRightTurnSign → ¬MakeRightTurn
    lambda *args: 1 - args[11] * (1 - args[21]),  # PedestrianCrossingSign → Decelerate
    lambda *args: 1 - args[12] * (1 - args[18]),  # StopSign → Stop
    lambda *args: 1 - args[13] * (1 - args[24])  # ThruTrafficMergeLeftSign → ChangeToLeftLane
]

def calculate_potential(weights, *features):
    potential = 0
    for w, rule in zip(weights, rules):
        result = rule(*features)
        if not isinstance(result, (int, float)):
            raise ValueError(f"Rule result is not a number: {result}")
        potential += w * result
    return potential

def calculate_probability(weights, features):
    potentials = []
    actions = [
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # Keep
        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),  # Accelerate
        (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),  # Decelerate
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),  # Stop
        (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),  # MakeLeftTurn
        (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),  # MakeRightTurn
        (0, 0, 0, 0, 0, 0, 1, 0, 0, 0),  # Merge
        (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),  # ChangeToLeftLane
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),  # ChangeToRightLane
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),  # PullOver
    ]
    for action in actions:
        combined_features = list(features) + list(action)
        combined_features = [float(f) for f in combined_features]  # 确保所有特征都是数值类型
        potential = calculate_potential(weights, *combined_features)
        potentials.append(np.exp(potential))
    total_potential = sum(potentials)
    probabilities = [p / total_potential for p in potentials]
    return probabilities

def log_likelihood(weights, data):
    ll = 0
    for sample in data:
        features = sample[:-10]
        actions = sample[-10:]
        probabilities = calculate_probability(weights, features)
        for i, action in enumerate(actions):
            if action == 1:
                ll += np.log(probabilities[i])
    return -ll  

def train(train_path, weight_path):
    with open(train_path, 'rb') as file:
        data = pickle.load(file)

    weights = np.random.rand(len(rules))
    
    result = minimize(log_likelihood, weights, args=(data,), method='L-BFGS-B')
    optimal_weights = result.x

    with open(weight_path, 'wb') as file:
        pickle.dump(optimal_weights, file)
    
    print(f"Optimal weights saved to {weight_path}")
    return optimal_weights

def load_weights(weight_path):
    with open(weight_path, 'rb') as file:
        weights = pickle.load(file)
    return weights

def infer(optimal_weights_path, features):
    optimal_weights = load_weights(optimal_weights_path)
    features = [float(f) for f in features]  # 确保所有特征都是数值类型
    probabilities = calculate_probability(optimal_weights, features)
    action_index = np.argmax(probabilities)
    actions = ["Keep", "Accelerate", "Decelerate", "Stop", "MakeLeftTurn", "MakeRightTurn", "Merge", "ChangeToLeftLane", "ChangeToRightLane", "PullOver"]
    print(f"Action: {actions[action_index]}")