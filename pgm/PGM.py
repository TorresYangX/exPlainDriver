import numpy as np
import torch.nn.functional as F
import torch

action_num = 17
condition_num = 13
action_indices = list(range(action_num))

# Keep, Accelerate, Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, Merge, LeftPass, RightPass,(10) 
# Yield, ChangeToLeftLane, ChangeToRightLane, ChangeToCenterLeftTurnLane, Park, PullOver(16)
# SolidRedLight, SolidYellowLight, YellowLeftArrowLight,(19)
# RedLeftArrowLight, MergingTrafficSign, WrongWaySign,(22)
# NoLeftTurnSign, NoRightTurnSign, PedestrianCrossingSign, StopSign, RedYieldSign, DoNotPassSign, SlowSign(29)

KEEP = 0
ACCELERATE = 1
DECELAERATE = 2
STOP = 3
REVERSE = 4
MAKE_LEFT_TURN = 5
MAKE_RIGHT_TURN = 6
MAKE_U_TURN = 7
MERGE = 8
LEFT_PASS = 9
RIGHT_PASS = 10
YIELD = 11
CHANGE_TO_LEFT_LANE = 12
CHANGE_TO_RIGHT_LANE = 13
CHANGE_TO_CENTER_LEFT_TURN_LANE = 14
PARK = 15
PULL_OVER = 16
SOLID_RED_LIGHT = 17
SOLID_YELLOW_LIGHT = 18
YELLOW_LEFT_ARROW_LIGHT = 19
RED_LEFT_ARROW_LIGHT = 20
MERGING_TRAFFIC_SIGN = 21
WRONG_WAY_SIGN = 22
NO_LEFT_TURN_SIGN = 23
NO_RIGHT_TURN_SIGN = 24
PEDESTRIAN_CROSSING_SIGN = 25
STOP_SIGN = 26
RED_YIELD_SIGN = 27
DO_NOT_PASS_SIGN = 28
SLOW_SIGN = 29


def compute_interface_weights(p_i):
    return np.log(p_i / (1 - p_i))


def compute_satisfaction(data, formulas):
    satisfaction_counts = np.zeros((len(data), len(formulas)))
    for i, instance in enumerate(data):
        for j, formula in enumerate(formulas):
            satisfaction_counts[i, j] = formula(instance)
    return satisfaction_counts


def log_sum_exp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


# def compute_log_likelihood(satisfaction_counts, weights, interface_weights, regularization):
#     weighted_sum = satisfaction_counts @ weights + interface_weights
#     log_likelihood = np.sum(weighted_sum) - log_sum_exp(weighted_sum)
#     log_likelihood -= 0.5 * regularization * np.sum(weights ** 2)  # L2 regularize
#     return log_likelihood


def compute_log_likelihood(satisfaction_counts, weights, regularization):
    weighted_sum = satisfaction_counts @ weights
    log_likelihood = np.sum(weighted_sum) - log_sum_exp(weighted_sum)
    log_likelihood -= 0.5 * regularization * np.sum(weights ** 2)  # L2 regularize
    return log_likelihood


# def update_weights(weights, satisfaction_counts, interface_weights, learning_rate, regularization):
#     weighted_sum = satisfaction_counts @ weights + interface_weights
#     expected_satisfaction = np.exp(weighted_sum - log_sum_exp(weighted_sum))
#     gradient = np.sum(satisfaction_counts, axis=0) - np.sum(expected_satisfaction[:, None] * satisfaction_counts, axis=0)
#     gradient -= regularization * weights
#     weights += learning_rate * gradient
#     return weights

def update_weights(weights, satisfaction_counts, learning_rate, regularization):
    weighted_sum = satisfaction_counts @ weights
    expected_satisfaction = np.exp(weighted_sum - log_sum_exp(weighted_sum))
    gradient = np.sum(satisfaction_counts, axis=0) - np.sum(expected_satisfaction[:, None] * satisfaction_counts, axis=0)
    gradient -= regularization * weights
    weights += learning_rate * gradient
    return weights


def generate_possible_instances(condition_input):
    possible_instances = []
    for action in action_indices:
        instance = np.zeros(action_num + condition_num)
        instance[action_num:] = condition_input 
        instance[action] = 1 
        possible_instances.append(instance)
    return np.array(possible_instances)


def compute_accuracy(true_labels, predictions):
    return np.mean(true_labels == predictions)


class PGM:
    
    def __init__(self, weight_path=None, learning_rate=0.01, max_iter=100, tol=1e-6, regularization=0.01, temperature=0.3):
        
        self.formulas = [
            
            lambda args: 1 - args[SOLID_RED_LIGHT] + args[SOLID_RED_LIGHT] * ((args[DECELAERATE] + args[STOP] - args[DECELAERATE] * args[STOP]) * (1 - args[ACCELERATE])),  # SolidRedLight → Decelerate ∨ Stop ∧ ¬Accelerate,
            lambda args: 1 - args[SOLID_YELLOW_LIGHT] + args[SOLID_YELLOW_LIGHT] * ((args[STOP] + args[DECELAERATE] - args[STOP] * args[DECELAERATE]) * (1 - args[ACCELERATE])),  # SolidYellowLight → Stop ∨ Decelerate ∧ ¬Accelerate
            lambda args: 1 - args[YELLOW_LEFT_ARROW_LIGHT] + args[YELLOW_LEFT_ARROW_LIGHT] * (args[STOP] + args[DECELAERATE] - args[STOP] * args[DECELAERATE]),  # YellowLeftArrowLight → Stop ∨ Decelerate
            lambda args: 1 - args[RED_LEFT_ARROW_LIGHT] + args[RED_LEFT_ARROW_LIGHT] * (1 - (args[MAKE_LEFT_TURN] + args[MAKE_U_TURN] - args[MAKE_LEFT_TURN] * args[MAKE_U_TURN])),  # RedLeftArrowLight → ¬(MakeLeftTurn ∨ MakeUTurn)
            lambda args: 1 - args[MERGING_TRAFFIC_SIGN] + args[MERGING_TRAFFIC_SIGN] * args[DECELAERATE],  # MergingTrafficSign → Decelerate
            lambda args: 1 - args[WRONG_WAY_SIGN] + args[WRONG_WAY_SIGN] * (args[STOP] + args[MAKE_U_TURN] + args[REVERSE] - args[STOP] * args[MAKE_U_TURN] - args[STOP] * args[REVERSE] - args[MAKE_U_TURN] * args[REVERSE] + args[STOP] * args[MAKE_U_TURN] * args[REVERSE]),  # WrongWaySign → Stop ∨ Reverse ∨ MakeUTurn
            lambda args: 1 - args[NO_LEFT_TURN_SIGN] + args[NO_LEFT_TURN_SIGN] * (1 - args[MAKE_LEFT_TURN]),  # NoLeftTurnSign → ¬MakeLeftTurn
            lambda args: 1 - args[NO_RIGHT_TURN_SIGN] + args[NO_RIGHT_TURN_SIGN] * (1 - args[MAKE_RIGHT_TURN]),  # NoRightTurnSign → ¬MakeRightTurn
            lambda args: 1 - args[PEDESTRIAN_CROSSING_SIGN] + args[PEDESTRIAN_CROSSING_SIGN] * (
                            (args[DECELAERATE] + args[STOP] + args[KEEP] - args[DECELAERATE] * args[STOP] -
                            args[DECELAERATE] * args[KEEP] - args[STOP] * args[KEEP] +
                            args[DECELAERATE] * args[STOP] * args[KEEP]) * (1 - args[ACCELERATE])
                            ),  # PedestrianCrossingSign → Decelerate ∨ Stop ∨ Keep ∧ ¬Accelerate
            lambda args: 1 - args[STOP_SIGN] + args[STOP_SIGN] * ((args[STOP] + args[DECELAERATE] - args[STOP] * args[DECELAERATE]) * (1 - args[ACCELERATE])),  # StopSign → Decelerate ∨ Stop ∧ ¬Accelerate
            lambda args: 1 - args[RED_YIELD_SIGN] + args[RED_YIELD_SIGN] * args[DECELAERATE], # RedYieldSign → Decelerate
            lambda args: 1 - args[DO_NOT_PASS_SIGN] + args[DO_NOT_PASS_SIGN] * (1 - (args[LEFT_PASS] + args[RIGHT_PASS] - args[LEFT_PASS] * args[RIGHT_PASS])), # DoNotPassSign → ¬(LeftPass ∨ RightPass)
            lambda args: 1 - args[SLOW_SIGN] + args[SLOW_SIGN] * args[DECELAERATE] # SlowSign → Decelerate
        ]
        
        if weight_path:
            self.weights = np.load(weight_path)
        else:
            self.weights = np.ones(len(self.formulas))
        
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        self.temperature = temperature
        

    def train_mln(self, data, saving_path, validation_data=None):
        
        """
        data: np.array([...])
        """
        
        weights = self.weights
        prev_log_likelihood = -np.inf
        prev_acc = -np.inf
        
        
        true_labels = np.argmax(data[:, :action_num], axis=1)
        
        # # compute interface weights
        # action_vector = data[:, :action_num]
        # action_prob = F.gumbel_softmax(torch.tensor(action_vector).float(), tau=self.temperature, hard=False).numpy()
        # max_prob = np.max(action_prob, axis=1)
        # interface_weights = np.clip(max_prob, 1e-6, 1 - 1e-6)
        
        for iteration in range(self.max_iter):
            # Compute satisfaction counts
            satisfaction_counts = compute_satisfaction(data, self.formulas)
            
            # Compute log likelihood
            # log_likelihood = compute_log_likelihood(satisfaction_counts, weights, interface_weights, self.regularization)
            log_likelihood = compute_log_likelihood(satisfaction_counts, weights, self.regularization)
            print(f"Iteration {iteration}, Log Likelihood: {log_likelihood}")
            
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            
            # Compute average probability of ground truth action
            predictions_prob = []
            for instance in data:
                condition_input = instance[action_num:]
                action_probs, _ = self.infer_action_probability(condition_input)
                predictions_prob.append(action_probs[true_labels[len(predictions_prob)]])
        
            avg_prob = np.mean(predictions_prob)
            print(f"Iteration {iteration}, Average Probability of Ground Truth Action: {avg_prob}")
            
            if np.abs(avg_prob - prev_acc) < self.tol:
                print("Validation score converged.")
                break
            
            if avg_prob > prev_acc:
                prev_acc = avg_prob
                np.save(saving_path, weights)
                print(f"Saving weights at iteration {iteration}")
            
            # if validation_data is not None:
            #     acc = self.eval(validation_data)
            #     print(f"Iteration {iteration}, Validation Score: {acc}")
            #     if np.abs(acc - prev_acc) < self.tol:
            #         print("Validation score converged.")
            #         break
            
            prev_log_likelihood = log_likelihood
            # weights = update_weights(weights, satisfaction_counts, interface_weights, self.learning_rate, self.regularization)
            weights = update_weights(weights, satisfaction_counts, self.learning_rate, self.regularization)
            self.weights = weights
            # save best weights
            # if validation_data is not None and acc > prev_acc:
            #     np.save(saving_path, weights)
            #     prev_acc = acc
            #     print(f"Saving weights at iteration {iteration}")
            
        return weights
    
    def eval(self, test_data):
        true_labels = np.argmax(test_data[:, :action_num], axis=1)
        predictions = []
        
        for instance in test_data:
            condition_input = instance[action_num:]
            _, action_index = self.infer_action_probability(condition_input)
            predictions.append(action_index)
            
        predictions = np.array(predictions)
        
        accuracy = compute_accuracy(true_labels, predictions)
        
        return accuracy

    
    
    def infer_action_probability(self, condition_input):
        
        """
        instance: np.array([...]) - conditions
        """
        
        possible_instances = generate_possible_instances(condition_input)
        satisfaction_counts = compute_satisfaction(possible_instances, self.formulas)
        
        # # generate interface weights
        # action_vector = possible_instances[:, :action_num]
        # action_prob = F.gumbel_softmax(torch.tensor(action_vector).float(), tau=self.temperature, hard=False).numpy()
        # interface_weights = np.sum(np.log(np.clip(action_prob, 1e-6, 1 - 1e-6)))
        
        # log_probs = satisfaction_counts @ self.weights + interface_weights
        log_probs = satisfaction_counts @ self.weights
        max_log_probs = np.max(log_probs)
        stabilized_log_probs = log_probs - max_log_probs
        exp_probs = np.exp(stabilized_log_probs)
        probs = exp_probs / np.sum(exp_probs)
        action_index = np.argmax(probs)
        return probs, action_index
    
    
    def validate_instance(self, instance):
        """
        instance: np.array([...]) - Input data with action and condition combinations
        """
        violations = []
        for idx, formula in enumerate(self.formulas):
            if formula(instance) != 1:
                violations.append(idx)
        return violations
    
    def compute_instance_probability(self, instance):
        condition_input = instance[action_num:]
        probs, _ = self.infer_action_probability(condition_input)
        action_index = np.argmax(instance[:action_num])
        return probs[action_index]