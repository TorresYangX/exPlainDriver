import numpy as np
import torch.nn.functional as F
import torch

action_num = 8
condition_num = 13
action_indices = list(range(action_num))


DECELAERATE = 0
STOP = 1
REVERSE = 2
MAKE_LEFT_TURN = 3
MAKE_RIGHT_TURN = 4
MAKE_U_TURN = 5
LEFT_PASS = 6
RIGHT_PASS = 7
SOLID_RED_LIGHT = 8
SOLID_YELLOW_LIGHT = 9
YELLOW_LEFT_ARROW_LIGHT = 10
RED_LEFT_ARROW_LIGHT = 11
MERGING_TRAFFIC_SIGN = 12
WRONG_WAY_SIGN = 13
NO_LEFT_TURN_SIGN = 14
NO_RIGHT_TURN_SIGN = 15
PEDESTRIAN_CROSSING_SIGN = 16
STOP_SIGN = 17
RED_YIELD_SIGN = 18
DO_NOT_PASS_SIGN = 19
SLOW_SIGN = 20


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

def compute_log_likelihood(satisfaction_counts, weights, interface_weights, regularization):
    weighted_sum = satisfaction_counts @ weights + interface_weights
    log_likelihood = np.sum(weighted_sum) - log_sum_exp(weighted_sum)
    log_likelihood -= 0.5 * regularization * np.sum(weights ** 2)  # L2 regularize
    return log_likelihood

def update_weights(weights, satisfaction_counts, interface_weights, learning_rate, regularization):
    weighted_sum = satisfaction_counts @ weights + interface_weights
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
            
            lambda args: 1 - args[SOLID_RED_LIGHT] + args[SOLID_RED_LIGHT] * args[STOP],  # SolidRedLight → Stop
            lambda args: 1 - args[SOLID_YELLOW_LIGHT] + args[SOLID_YELLOW_LIGHT] * (args[STOP] + args[DECELAERATE] - args[STOP] * args[DECELAERATE]),  # SolidYellowLight → Stop ∨ Decelerate
            lambda args: 1 - args[YELLOW_LEFT_ARROW_LIGHT] + args[YELLOW_LEFT_ARROW_LIGHT] * (args[STOP] + args[DECELAERATE] - args[STOP] * args[DECELAERATE]),  # YellowLeftArrowLight → Stop ∨ Decelerate
            lambda args: 1 - args[RED_LEFT_ARROW_LIGHT] + args[RED_LEFT_ARROW_LIGHT] * args[STOP] * (1 - (args[MAKE_LEFT_TURN] + args[MAKE_U_TURN] - args[MAKE_LEFT_TURN] * args[MAKE_U_TURN])),  # RedLeftArrowLight → Stop ∧ ¬(MakeLeftTurn ∨ MakeUTurn)
            lambda args: 1 - args[MERGING_TRAFFIC_SIGN] + args[MERGING_TRAFFIC_SIGN] * args[DECELAERATE],  # MergingTrafficSign → Decelerate
            lambda args: 1 - args[WRONG_WAY_SIGN] + args[WRONG_WAY_SIGN] * (args[STOP] + args[MAKE_U_TURN] + args[REVERSE] - args[STOP] * args[MAKE_U_TURN] - args[STOP] * args[REVERSE] - args[MAKE_U_TURN] * args[REVERSE] + args[STOP] * args[MAKE_U_TURN] * args[REVERSE]),  # WrongWaySign → Stop ∨ Reverse ∨ MakeUTurn
            lambda args: 1 - args[NO_LEFT_TURN_SIGN] + args[NO_LEFT_TURN_SIGN] * (1 - args[MAKE_LEFT_TURN]),  # NoLeftTurnSign → ¬MakeLeftTurn
            lambda args: 1 - args[NO_RIGHT_TURN_SIGN] + args[NO_RIGHT_TURN_SIGN] * (1 - args[MAKE_RIGHT_TURN]),  # NoRightTurnSign → ¬MakeRightTurn
            lambda args: 1 - args[PEDESTRIAN_CROSSING_SIGN] + args[PEDESTRIAN_CROSSING_SIGN] * (args[STOP] + args[DECELAERATE] - args[STOP] * args[DECELAERATE]),  # PedestrianCrossingSign → Decelerate ∨ Stop 
            lambda args: 1 - args[STOP_SIGN] + args[STOP_SIGN] * args[STOP],  # StopSign → Stop
            lambda args: 1 - args[RED_YIELD_SIGN] + args[RED_YIELD_SIGN] * args[DECELAERATE], # RedYieldSign → Decelerate
            lambda args: 1 - args[DO_NOT_PASS_SIGN] + args[DO_NOT_PASS_SIGN] * (1 - (args[LEFT_PASS] + args[RIGHT_PASS] - args[LEFT_PASS] * args[RIGHT_PASS])), # DoNotPassSign → ¬(LeftPass ∨ RightPass)
            lambda args: 1 - args[SLOW_SIGN] + args[SLOW_SIGN] * args[DECELAERATE] # SlowSign → Decelerate

            
            # lambda args: 1 - args[18] + args[18] * (args[1] + args[0] - args[1] * args[0]),  # SolidGreenLight → Accelerate ∨ Keep
            # lambda args: 1 - args[24] * args[18] + args[24] * args[18] * args[7],  # IntersectionAhead ∧ SolidGreenLight → MakeUTurn
            # lambda args: 1 - args[22] + args[22] * (args[5] + args[7] - args[5] * args[7]),  # GreenLeftArrowLight → MakeLeftTurn ∨ MakeUTurn
            # lambda args: 1 - args[27] + args[27] * args[13],  # KeepRightSign → ChangeToRightLane
            # lambda args: 1 - args[33] + args[33] * args[12],  # ThruTrafficMergeLeftSign → ChangeToLeftLane
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
        
        # compute interface weights
        action_vector = data[:, :action_num]
        action_prob = F.gumbel_softmax(torch.tensor(action_vector).float(), tau=self.temperature, hard=False).numpy()
        max_prob = np.max(action_prob, axis=1)
        interface_weights = np.clip(max_prob, 1e-6, 1 - 1e-6)
        
        for iteration in range(self.max_iter):
            # Compute satisfaction counts
            satisfaction_counts = compute_satisfaction(data, self.formulas)
            
            # Compute log likelihood
            log_likelihood = compute_log_likelihood(satisfaction_counts, weights, interface_weights, self.regularization)
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
            
            if validation_data is not None:
                acc = self.eval(validation_data)
                print(f"Iteration {iteration}, Validation Score: {acc}")
                if np.abs(acc - prev_acc) < self.tol:
                    print("Validation score converged.")
                    break
            
            prev_log_likelihood = log_likelihood
            weights = update_weights(weights, satisfaction_counts, interface_weights, self.learning_rate, self.regularization)
            self.weights = weights
            # save best weights
            if validation_data is not None and acc > prev_acc:
                np.save(saving_path, weights)
                prev_acc = acc
                print(f"Saving weights at iteration {iteration}")
            
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
        possible_instances = generate_possible_instances(condition_input)
        satisfaction_counts = compute_satisfaction(possible_instances, self.formulas)
        
        # generate interface weights
        action_vector = possible_instances[:, :action_num]
        action_prob = F.gumbel_softmax(torch.tensor(action_vector).float(), tau=self.temperature, hard=False).numpy()
        max_prob = np.max(action_prob, axis=1)
        interface_weights = np.clip(max_prob, 1e-6, 1 - 1e-6)
        
        
        log_probs = satisfaction_counts @ self.weights + interface_weights
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