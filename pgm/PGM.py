import numpy as np
from pgm.config import *
from scipy.special import softmax

def compute_satisfaction(data, formulas):
    satisfaction_counts = np.zeros((len(data), len(formulas)))
    for i, instance in enumerate(data):
        for j, formula in enumerate(formulas):
            satisfaction_counts[i, j] = formula(instance)
    return satisfaction_counts


def log_sum_exp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


def compute_log_likelihood(satisfaction_counts, weights, regularization):
    weighted_sum = satisfaction_counts @ weights
    log_likelihood = np.sum(weighted_sum) - log_sum_exp(weighted_sum)
    log_likelihood -= 0.5 * regularization * np.sum(weights ** 2)  # L2 regularize
    return log_likelihood


def update_weights(weights, satisfaction_counts, learning_rate, regularization):
    weighted_sum = satisfaction_counts @ weights
    expected_satisfaction = np.exp(weighted_sum - log_sum_exp(weighted_sum))
    gradient = np.sum(satisfaction_counts, axis=0) - np.sum(expected_satisfaction[:, None] * satisfaction_counts, axis=0)
    gradient -= regularization * weights
    weights += learning_rate * gradient
    return weights


def generate_possible_instances(condition_input, action_num, condition_num):
    possible_instances = []
    action_indices = list(range(action_num))
    
    for action in action_indices:
        instance = np.zeros(action_num + condition_num)
        instance[action_num:] = condition_input 
        instance[action] = 1 
        possible_instances.append(instance)
    return np.array(possible_instances)


def compute_accuracy(true_labels, predictions):
    return np.mean(true_labels == predictions)


class PGM:
    
    def __init__(self, config, weights=None, learning_rate=0.01, max_iter=100, tol=1e-6, regularization=0.01):
        self.formulas = config.formulas
        self.action_num = config.action_num
        self.condition_num = config.condition_num
        
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(len(self.formulas))
        
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        

    def train_mln(self, data, saving_path):
        """
        data: np.array([...])
        """   
        weights = self.weights
        prev_log_likelihood = -np.inf
        prev_acc = -np.inf
               
        true_labels = np.argmax(data[:, :self.action_num], axis=1)
        
        for iteration in range(self.max_iter):
            satisfaction_counts = compute_satisfaction(data, self.formulas)
            log_likelihood = compute_log_likelihood(satisfaction_counts, weights, self.regularization)
            predictions_prob = []
            for instance in data:
                condition_input = instance[self.action_num:]
                action_probs, _ = self.infer_action_probability(condition_input)
                predictions_prob.append(action_probs[true_labels[len(predictions_prob)]])
        
            avg_prob = np.mean(predictions_prob)
            print(f"Iteration {iteration}, Average Probability of Ground Truth Action: {avg_prob}, Log Likelihood: {log_likelihood}")
                    
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                np.save(saving_path, weights)
                print("log likelihood converged.")
                break
            
            if np.abs(avg_prob - prev_acc) < self.tol:
                np.save(saving_path, weights)
                print("accuracy converged.")
                break
            
            if avg_prob > prev_acc:
                prev_acc = avg_prob
                np.save(saving_path, weights)
                print(f"Saving weights at iteration {iteration}")
                  
            prev_log_likelihood = log_likelihood
            weights = update_weights(weights, satisfaction_counts, self.learning_rate, self.regularization)
            self.weights = weights
        return weights
    
    
    def eval(self, test_data):
        true_labels = np.argmax(test_data[:, :self.action_num], axis=1)
        predictions = []
        for instance in test_data:
            condition_input = instance[self.action_num:]
            _, action_index = self.infer_action_probability(condition_input)
            predictions.append(action_index)
        predictions = np.array(predictions)
        accuracy = compute_accuracy(true_labels, predictions)
        return accuracy

    
    
    def infer_action_probability(self, condition_input):
        """
        instance: np.array([...]): conditions
        """
        possible_instances = generate_possible_instances(condition_input, self.action_num, self.condition_num)
        satisfaction_counts = compute_satisfaction(possible_instances, self.formulas)
        log_probs = satisfaction_counts @ self.weights
        probs = softmax(log_probs)
        action_index = np.argmax(probs)
        return probs, action_index
    
    
    def validate_instance(self, instance):
        """
        instance: np.array([...]): Input data with action and condition combinations
        """
        violations = []
        for idx, formula in enumerate(self.formulas):
            if formula(instance) != 1:
                violations.append(idx)
        return violations
    
    
    def compute_instance_probability(self, instance):
        """
        Compute the probability of a given instance.
        instance: np.array([...]): Input data with action and condition combinations
        """
        condition_input = instance[self.action_num:]
        probs, _ = self.infer_action_probability(condition_input)
        return probs[instance[:self.action_num].argmax()]
        