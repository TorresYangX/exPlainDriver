import numpy as np

action_num = 17
condition_num = 20
action_indices = list(range(17))

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

def generate_possible_instances(condition_input):
    possible_instances = []
    for action in action_indices:
        instance = np.zeros(action_num+condition_num)
        instance[action_num:] = condition_input 
        instance[action] = 1 
        possible_instances.append(instance)
    return np.array(possible_instances)

def compute_accuracy(true_labels, predictions):
    return np.mean(true_labels == predictions)

def compute_precision(true_labels, predictions):
    true_positive = np.sum((true_labels == 1) & (predictions == 1))
    false_positive = np.sum((true_labels == 0) & (predictions == 1))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

def compute_recall(true_labels, predictions):
    true_positive = np.sum((true_labels == 1) & (predictions == 1))
    false_negative = np.sum((true_labels == 1) & (predictions == 0))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

def compute_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0



class PGM:
    
    def __init__(self, weight_path=None, learning_rate=0.01, max_iter=100, tol=1e-6, regularization=0.01):
        
        self.formulas = [
            lambda args: 1 - args[19] + args[19] * args[3],  # SolidRedLight → Stop
            lambda args: 1 - args[18] + args[18] * (args[1] + args[0] - args[1] * args[0]),  # SolidGreenLight → Accelerate ∨ Keep
            lambda args: 1 - args[24] * args[18] + args[24] * args[18] * args[7],  # IntersectionAhead ∧ SolidGreenLight → MakeUTurn
            lambda args: 1 - args[20] * args[24] + args[20] * args[24] * args[3],  # SolidYellowLight ∧ IntersectionAhead → Stop
            lambda args: 1 - args[21] * args[24] + args[21] * args[24] * args[3],  # YellowLeftArrowLight ∧ IntersectionAhead → Stop
            lambda args: 1 - args[22] + args[22] * (args[5] + args[7] - args[5] * args[7]),  # GreenLeftArrowLight → MakeLeftTurn ∨ MakeUTurn
            lambda args: 1 - args[23] + args[23] * args[3] * (1 - (args[5] + args[7] - args[5] * args[7])),  # RedLeftArrowLight → Stop ∧ ¬(MakeLeftTurn ∨ MakeUTurn)
            lambda args: 1 - args[25] + args[25] * args[2],  # MergingTrafficSign → Decelerate
            lambda args: 1 - args[26] + args[26] * (args[3] + args[4] + args[7] - args[3] * args[4] - args[3] * args[7] - args[4] * args[7] + args[3] * args[4] * args[7]),  # WrongWaySign → Stop ∨ Reverse ∨ MakeUTurn
            lambda args: 1 - args[27] + args[27] * args[13],  # KeepRightSign → ChangeToRightLane
            lambda args: 1 - args[29] + args[29] * (1 - args[5]),  # NoLeftTurnSign → ¬MakeLeftTurn
            lambda args: 1 - args[30] + args[30] * (1 - args[6]),  # NoRightTurnSign → ¬MakeRightTurn
            lambda args: 1 - args[31] + args[31] * args[2],  # PedestrianCrossingSign → Decelerate
            lambda args: 1 - args[32] + args[32] * args[3],  # StopSign → Stop
            lambda args: 1 - args[33] + args[33] * args[12],  # ThruTrafficMergeLeftSign → ChangeToLeftLane
        ]
        
        if weight_path:
            self.weights = np.load(weight_path)
        else:
            self.weights = np.ones(len(self.formulas))
        
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization


    def train_mln(self, data, validation_data=None):
        
        """
        data: np.array([...])
        """
        
        weights = self.weights
        prev_log_likelihood = -np.inf
        prev_acc = -np.inf
        true_labels = np.argmax(data[:, :17], axis=1)
        
        for iteration in range(self.max_iter):
            satisfaction_counts = compute_satisfaction(data, self.formulas)
            log_likelihood = compute_log_likelihood(satisfaction_counts, weights, self.regularization)
            print(f"Iteration {iteration}, Log Likelihood: {log_likelihood}")
            
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            
            predictions_prob = []
            for instance in data:
                condition_input = instance[17:]
                action_probs = self.infer_action_probability(condition_input)
                predictions_prob.append(action_probs[true_labels[len(predictions_prob)]])
        
            avg_prob = np.mean(predictions_prob)
            print(f"Iteration {iteration}, Average Probability of Ground Truth Action: {avg_prob}")
            
            if validation_data is not None:
                acc,_,_,_ = self.eval(validation_data)
                print(f"Iteration {iteration}, Validation Score: {acc}")
                if np.abs(acc - prev_acc) < self.tol:
                    print("Validation score converged.")
                    break
            
            prev_log_likelihood = log_likelihood
            weights = update_weights(weights, satisfaction_counts, self.learning_rate, self.regularization)
            self.weights = weights
            
        return weights
    
    def validate_model(self, validation_data):
        satisfaction_counts = compute_satisfaction(validation_data, self.formulas)
        log_likelihood = compute_log_likelihood(satisfaction_counts, self.weights)
        return log_likelihood
    
    def infer_action_probability(self, condition_input):
        possible_instances = generate_possible_instances(condition_input)
        satisfaction_counts = compute_satisfaction(possible_instances, self.formulas)
        log_probs = satisfaction_counts @ self.weights
        probs = np.exp(log_probs)
        probs /= np.sum(probs)
        return probs
    
    def eval(self, test_data):
        true_labels = np.argmax(test_data[:, :action_num], axis=1)
        predictions = []
        
        for instance in test_data:
            condition_input = instance[action_num:]
            action_probs = self.infer_action_probability(condition_input)
            predicted_action = np.argmax(action_probs)
            predictions.append(predicted_action)
            
        predictions = np.array(predictions)
        
        accuracy = compute_accuracy(true_labels, predictions)
        precision_list = []
        recall_list = []
        f1_list = []
        
        for label in range(17):  
            precision = compute_precision(true_labels == label, predictions == label)
            recall = compute_recall(true_labels == label, predictions == label)
            f1_score = compute_f1_score(precision, recall)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1_score)
        
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        f1_score = np.mean(f1_list)
        
        return accuracy, precision, recall, f1_score