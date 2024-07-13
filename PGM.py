import numpy as np
from scipy.optimize import minimize
import pickle

action_num = 17
feature_num = 18

rules = [
    lambda *args: 1 - args[3] * (1 - args[19]),  # SolidRedLight → Stop
    lambda *args: 1 - args[1] * (1 - args[18]),  # SolidGreenLight → Accelerate
    lambda *args: 1 - args[3] * (1 - (args[20] * args[24])),  # SolidYellowLight ∧ IntersectionAhead → Stop
    lambda *args: 1 - args[3] * (1 - (args[21] * args[24])),  # YellowLeftArrowLight ∧ IntersectionAhead → Stop
    lambda *args: 1 - args[22] * (1 - args[5]),  # MakeLeftTurn → GreenLeftArrowLight
    lambda *args: 1 - (1 - args[5]) * (1 - args[23]),  # RedLeftArrowLight → ¬MakeLeftTurn
    lambda *args: 1 - args[2] * (1 - args[25]),  # MergingTrafficSign → Decelerate
    lambda *args: 1 - args[3] * (1 - args[26]),  # WrongWaySign → Stop
    lambda *args: 1 - args[13] * (1 - args[27]),  # KeepRightSign → ChangeToRightLane
    lambda *args: 1 - (1 - args[5]) * (1 - args[29]),  # NoLeftTurnSign → ¬MakeLeftTurn
    lambda *args: 1 - (1 - args[6]) * (1 - args[30]),  # NoRightTurnSign → ¬MakeRightTurn
    lambda *args: 1 - args[2] * (1 - args[31]),  # PedestrianCrossingSign → Decelerate
    lambda *args: 1 - args[3] * (1 - args[32]),  # StopSign → Stop
    lambda *args: 1 - args[12] * (1 - args[33])  # ThruTrafficMergeLeftSign → ChangeToLeftLane
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
        (1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),# Keep
        (0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),# Accelerate
        (0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0),# Decelerate
        (0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0),# Stop
        (0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0),# Reverse
        (0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0),# MakeLeftTurn
        (0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0),# MakeRightTurn
        (0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0),# MakeUTurn
        (0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0),# Merge
        (0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0),# LeftPass
        (0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0),# RightPass
        (0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0),# Yield
        (0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0),# ChangeToLeftLane
        (0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0),# ChangeToRightLane
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0),# ChangeToCenterLeftTurnLane
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0),# Park
        (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1)# PullOver
    ]
    for action in actions:
        combined_features = list(action)+list(features)
        combined_features = [float(f) for f in combined_features]  # 确保所有特征都是数值类型
        potential = calculate_potential(weights, *combined_features)
        potentials.append(np.exp(potential))
    total_potential = sum(potentials)
    probabilities = [p / total_potential for p in potentials]
    return probabilities

def log_likelihood(weights, data):
    ll = 0
    for sample in data:
        actions = sample[:action_num]
        features = sample[action_num:]
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
    features = [float(f) for f in features]
    probabilities = calculate_probability(optimal_weights, features)
    action_index = np.argmax(probabilities)
    actions = ["Keep", "Accelerate", "Decelerate", "Stop", "Reverse",
        "MakeLeftTurn", "MakeRightTurn", "MakeUTurn", "Merge",
        "LeftPass", "RightPass", "Yield", "ChangeToLeftLane",
        "ChangeToRightLane", "ChangeToCenterLeftTurnLane",
        "Park", "PullOver"]
    print(f"Action: {actions[action_index]}")