import numpy as np

class BDDX:
  def __init__(self):
    self.predicate = {
      'KEEP': 0,
      'ACCELERATE': 1,
      'DECELERATE': 2,
      'STOP': 3,
      'REVERSE': 4,
      'MAKE_LEFT_TURN': 5,
      'MAKE_RIGHT_TURN': 6,
      'MAKE_U_TURN': 7,
      'MERGE': 8,
      'LEFT_PASS': 9,
      'RIGHT_PASS': 10,
      'YIELD': 11,
      'CHANGE_TO_LEFT_LANE': 12,
      'CHANGE_TO_RIGHT_LANE': 13,
      'PARK': 14,
      'PULL_OVER': 15,
      
      'SOLID_RED_LIGHT': 16,
      'SOLID_YELLOW_LIGHT': 17,
      'YELLOW_LEFT_ARROW_LIGHT': 18,
      'RED_LEFT_ARROW_LIGHT': 19,
      'MERGING_TRAFFIC_SIGN': 20,
      'NO_LEFT_TURN_SIGN': 21,
      'NO_RIGHT_TURN_SIGN': 22,
      'PEDESTRIAN_CROSSING_SIGN': 23,
      'STOP_SIGN': 24,
      'RED_YIELD_SIGN': 25,
      'SLOW_SIGN': 26,
      'SOLID_GREEN_LIGHT': 27,
      
      'KEEP_CS': 28,
      'ACCELERATE_CS': 29,
      'DECELERATE_CS': 30,
      'STOP_CS': 31,
      'REVERSE_CS': 32,
      'STRAIGHT_CS': 33,
      'LEFT_CS': 34,
      'RIGHT_CS': 35,
      
      'KEEP_LLM': 36,
      'ACCELERATE_LLM': 37,
      'DECELERATE_LLM': 38,
      'STOP_LLM': 39,
      'REVERSE_LLM': 40,
      'MAKE_LEFT_TURN_LLM': 41,
      'MAKE_RIGHT_TURN_LLM': 42,
      'MAKE_U_TURN_LLM': 43,
      'MERGE_LLM': 44,
      'LEFT_PASS_LLM': 45,
      'RIGHT_PASS_LLM': 46,
      'YIELD_LLM': 47,
      'CHANGE_TO_LEFT_LANE_LLM': 48,
      'CHANGE_TO_RIGHT_LANE_LLM': 49,
      'PARK_LLM': 50,
      'PULL_OVER_LLM': 51
    }
      
    self.action_num = 16
    self.condition_num = 36
      
    self.formulas = [            
      lambda args: 1 - args[self.predicate["SOLID_RED_LIGHT"]] + args[self.predicate["SOLID_RED_LIGHT"]] * \
                      ((1 - args[self.predicate["ACCELERATE"]]) * \
                       (1 - args[self.predicate["LEFT_PASS"]]) * \
                       (1 - args[self.predicate["YIELD"]])), # SolidRedLight → ¬Accelerate ∧ ¬LeftPass ∧ ¬Yield
      
      lambda args: 1 - args[self.predicate["SOLID_YELLOW_LIGHT"]] + args[self.predicate["SOLID_YELLOW_LIGHT"]] * \
                      ((args[self.predicate["MAKE_LEFT_TURN"]] + \
                        args[self.predicate["MAKE_RIGHT_TURN"]] + \
                        args[self.predicate["KEEP"]] + \
                        args[self.predicate["STOP"]] + \
                        args[self.predicate["DECELERATE"]] - \
                        args[self.predicate["MAKE_LEFT_TURN"]] * args[self.predicate["MAKE_RIGHT_TURN"]] - \
                        args[self.predicate["MAKE_LEFT_TURN"]] * args[self.predicate["KEEP"]] - \
                        args[self.predicate["MAKE_LEFT_TURN"]] * args[self.predicate["STOP"]] - \
                        args[self.predicate["MAKE_LEFT_TURN"]] * args[self.predicate["DECELERATE"]] - \
                        args[self.predicate["MAKE_RIGHT_TURN"]] * args[self.predicate["KEEP"]] - \
                        args[self.predicate["MAKE_RIGHT_TURN"]] * args[self.predicate["STOP"]] - \
                        args[self.predicate["MAKE_RIGHT_TURN"]] * args[self.predicate["DECELERATE"]] - \
                        args[self.predicate["KEEP"]] * args[self.predicate["STOP"]] - \
                        args[self.predicate["KEEP"]] * args[self.predicate["DECELERATE"]] - \
                        args[self.predicate["STOP"]] * args[self.predicate["DECELERATE"]] + \
                        args[self.predicate["MAKE_LEFT_TURN"]] * args[self.predicate["MAKE_RIGHT_TURN"]] * \
                        args[self.predicate["KEEP"]] * args[self.predicate["STOP"]] * \
                        args[self.predicate["DECELERATE"]]) * \
                        (1 - args[self.predicate["ACCELERATE"]])), #SolidYellowLight → MakeLeftTurn ∨ MakeRightTurn∨ Keep ∨ Stop ∨ Decelerate ∧ ¬Accelerate

      lambda args: 1 - args[self.predicate["YELLOW_LEFT_ARROW_LIGHT"]] + args[self.predicate["YELLOW_LEFT_ARROW_LIGHT"]] * \
                      (args[self.predicate["STOP"]] + args[self.predicate["DECELERATE"]] - \
                      args[self.predicate["STOP"]] * args[self.predicate["DECELERATE"]]),  # YellowLeftArrowLight → Stop ∨ Decelerate

      lambda args: 1 - args[self.predicate["RED_LEFT_ARROW_LIGHT"]] + args[self.predicate["RED_LEFT_ARROW_LIGHT"]] * \
                      (1 - (args[self.predicate["MAKE_LEFT_TURN"]] + args[self.predicate["MAKE_U_TURN"]] - \
                      args[self.predicate["MAKE_LEFT_TURN"]] * args[self.predicate["MAKE_U_TURN"]])),  # RedLeftArrowLight → ¬(MakeLeftTurn ∨ MakeUTurn)

      lambda args: 1 - args[self.predicate["MERGING_TRAFFIC_SIGN"]] + args[self.predicate["MERGING_TRAFFIC_SIGN"]] * \
                      args[self.predicate["DECELERATE"]],  # MergingTrafficSign → Decelerate

      lambda args: 1 - args[self.predicate["NO_LEFT_TURN_SIGN"]] + args[self.predicate["NO_LEFT_TURN_SIGN"]] * \
                      (1 - args[self.predicate["MAKE_LEFT_TURN"]]),  # NoLeftTurnSign → ¬MakeLeftTurn

      lambda args: 1 - args[self.predicate["NO_RIGHT_TURN_SIGN"]] + args[self.predicate["NO_RIGHT_TURN_SIGN"]] * \
                      (1 - args[self.predicate["MAKE_RIGHT_TURN"]]),  # NoRightTurnSign → ¬MakeRightTurn

      lambda args: 1 - args[self.predicate["RED_YIELD_SIGN"]] + args[self.predicate["RED_YIELD_SIGN"]] * \
                      args[self.predicate["DECELERATE"]],  # RedYieldSign → Decelerate

      lambda args: 1 - args[self.predicate["SLOW_SIGN"]] + args[self.predicate["SLOW_SIGN"]] * \
                      (1 - args[self.predicate["ACCELERATE"]]),  # SlowSign → ¬Accelerate
                      
      lambda args: 1 - args[self.predicate["STOP_SIGN"]] + args[self.predicate["STOP_SIGN"]] * \
                      (1 - args[self.predicate["PULL_OVER"]]),  # StopSign → ¬Pull_Over
                      
      # lambda args: 1 - args[self.predicate["KEEP_CS"]] + args[self.predicate["KEEP_CS"]] * \
      #                 args[self.predicate["KEEP"]],  # KEEP_CS → Keep

      # lambda args: 1 - args[self.predicate["ACCELERATE_CS"]] + args[self.predicate["ACCELERATE_CS"]] * \
      #                 args[self.predicate["ACCELERATE"]],  # ACCELERATE_CS → Accelerate
                      
      # lambda args: 1 - args[self.predicate["DECELERATE_CS"]] + args[self.predicate["DECELERATE_CS"]] * \
      #                 args[self.predicate["DECELERATE"]],  # DECELERATE_CS → DECELERATE
                      
      # lambda args: 1 - args[self.predicate["STOP_CS"]] + args[self.predicate["STOP_CS"]] * \
      #                 args[self.predicate["STOP"]],  # STOP_CS → Stop
      
      lambda args: 1 - args[self.predicate["KEEP_CS"]] + args[self.predicate["KEEP_CS"]] * \
                      (args[self.predicate["KEEP"]] + args[self.predicate["ACCELERATE"]]),  # KEEP_CS → KEEP ∨ ACCELERATE
                      
      lambda args: 1 - args[self.predicate["ACCELERATE_CS"]] + args[self.predicate["ACCELERATE_CS"]] * \
                      (args[self.predicate["KEEP"]] + args[self.predicate["ACCELERATE"]]),  # ACCELERATE_CS → KEEP ∨ ACCELERATE

      lambda args: 1 - args[self.predicate["DECELERATE_CS"]] + args[self.predicate["DECELERATE_CS"]] * \
                      (args[self.predicate["DECELERATE"]] + args[self.predicate["STOP"]]),  # DECELERATE_CS → DECELERATE ∨ STOP
      
      lambda args: 1 - args[self.predicate["STOP_CS"]] + args[self.predicate["STOP_CS"]] * \
                      (args[self.predicate["DECELERATE"]] + args[self.predicate["STOP"]]),  # STOP_CS → DECELERATE ∨ STOP
                      
      lambda args: 1 - args[self.predicate["REVERSE_CS"]] + args[self.predicate["REVERSE_CS"]] * \
                      args[self.predicate["REVERSE"]],  # REVERSE_CS → REVERSE
      
      lambda args: 1 - args[self.predicate["LEFT_CS"]] + args[self.predicate["LEFT_CS"]] * \
                      (args[self.predicate["MAKE_LEFT_TURN"]] + args[self.predicate["CHANGE_TO_LEFT_LANE"]] - \
                       args[self.predicate["MAKE_LEFT_TURN"]] * args[self.predicate["CHANGE_TO_LEFT_LANE"]]),  # LEFT_CS → MakeLeftTurn ∨ ChangeToLeftLane
      
      lambda args: 1 - args[self.predicate["RIGHT_CS"]] + args[self.predicate["RIGHT_CS"]] * \
                      (args[self.predicate["MAKE_RIGHT_TURN"]] + args[self.predicate["CHANGE_TO_RIGHT_LANE"]] - \
                       args[self.predicate["MAKE_RIGHT_TURN"]] * args[self.predicate["CHANGE_TO_RIGHT_LANE"]]),  # RIGHT_CS → MakeRightTurn ∨ ChangeToRightLane   
           
      lambda args: 1 - (args[self.predicate["LEFT_CS"]] * args[self.predicate["CHANGE_TO_RIGHT_LANE_LLM"]]) + \
              (args[self.predicate["LEFT_CS"]] * args[self.predicate["CHANGE_TO_RIGHT_LANE_LLM"]] * args[self.predicate["CHANGE_TO_LEFT_LANE"]]), # LEFT_CS ∧ ChangeToRightLane_llm → ChangeToLeftLane   
                
      lambda args: 1 - (args[self.predicate["RIGHT_CS"]] * args[self.predicate["CHANGE_TO_LEFT_LANE_LLM"]]) + \
              (args[self.predicate["RIGHT_CS"]] * args[self.predicate["CHANGE_TO_LEFT_LANE_LLM"]] * args[self.predicate["CHANGE_TO_RIGHT_LANE"]]),  # RIGHT_CS ∧ ChangeToLeftLane_llm → ChangeToRightLane   
       
      # KEEP_LLM → KEEP
      # ACCELERATE_LLM → ACCELERATE
      # DECELERATE_LLM → DECELERATE
      # STOP_LLM → STOP
      # REVERSE_LLM → REVERSE
      # MAKE_LEFT_TURN_LLM → MAKE_LEFT_TURN
      # MAKE_RIGHT_TURN_LLM → MAKE_RIGHT_TURN
      # MAKE_U_TURN_LLM → MAKE_U_TURN
      # MERGE_LLM → MERGE
      # LEFT_PASS_LLM → LEFT_PASS
      # RIGHT_PASS_LLM → RIGHT_PASS
      # YIELD_LLM → YIELD
      # CHANGE_TO_LEFT_LANE_LLM → CHANGE_TO_LEFT_LANE
      # CHANGE_TO_RIGHT_LANE_LLM → CHANGE_TO_RIGHT_LANE
      # PARK_LLM → PARK
      # PULL_OVER_LLM → PULL_OVER
      
      lambda args: 1 - args[self.predicate["KEEP_LLM"]] + args[self.predicate["KEEP_LLM"]] * \
                      args[self.predicate["KEEP"]],  # KEEP_LLM → KEEP
      
      lambda args: 1 - args[self.predicate["ACCELERATE_LLM"]] + args[self.predicate["ACCELERATE_LLM"]] * \
                      args[self.predicate["ACCELERATE"]],  # ACCELERATE_LLM → ACCELERATE
                      
      lambda args: 1 - args[self.predicate["DECELERATE_LLM"]] + args[self.predicate["DECELERATE_LLM"]] * \
                      args[self.predicate["DECELERATE"]],  # DECELERATE_LLM → DECELERATE  
                      
      lambda args: 1 - args[self.predicate["STOP_LLM"]] + args[self.predicate["STOP_LLM"]] * \
                      args[self.predicate["STOP"]],  # STOP_LLM → STOP  
                      
      lambda args: 1 - args[self.predicate["REVERSE_LLM"]] + args[self.predicate["REVERSE_LLM"]] * \
                      args[self.predicate["REVERSE"]],  # REVERSE_LLM → REVERSE
                      
      lambda args: 1 - args[self.predicate["MAKE_LEFT_TURN_LLM"]] + args[self.predicate["MAKE_LEFT_TURN_LLM"]] * \
                      args[self.predicate["MAKE_LEFT_TURN"]],  # MAKE_LEFT_TURN_LLM → MAKE_LEFT_TURN
                      
      lambda args: 1 - args[self.predicate["MAKE_RIGHT_TURN_LLM"]] + args[self.predicate["MAKE_RIGHT_TURN_LLM"]] * \
                      args[self.predicate["MAKE_RIGHT_TURN"]],  # MAKE_RIGHT_TURN_LLM → MAKE_RIGHT_TURN
                      
      lambda args: 1 - args[self.predicate["MAKE_U_TURN_LLM"]] + args[self.predicate["MAKE_U_TURN_LLM"]] * \
                      args[self.predicate["MAKE_U_TURN"]],  # MAKE_U_TURN_LLM → MAKE_U_TURN
                      
      lambda args: 1 - args[self.predicate["MERGE_LLM"]] + args[self.predicate["MERGE_LLM"]] * \
                      args[self.predicate["MERGE"]],  # MERGE_LLM → MERGE
                      
      lambda args: 1 - args[self.predicate["LEFT_PASS_LLM"]] + args[self.predicate["LEFT_PASS_LLM"]] * \
                      args[self.predicate["LEFT_PASS"]],  # LEFT_PASS_LLM → LEFT_PASS
      
      lambda args: 1 - args[self.predicate["RIGHT_PASS_LLM"]] + args[self.predicate["RIGHT_PASS_LLM"]] * \
                      args[self.predicate["RIGHT_PASS"]],  # RIGHT_PASS_LLM → RIGHT_PASS
                      
      lambda args: 1 - args[self.predicate["YIELD_LLM"]] + args[self.predicate["YIELD_LLM"]] * \
                      args[self.predicate["YIELD"]],  # YIELD_LLM → YIELD
                      
      lambda args: 1 - args[self.predicate["CHANGE_TO_LEFT_LANE_LLM"]] + args[self.predicate["CHANGE_TO_LEFT_LANE_LLM"]] * \
                      args[self.predicate["CHANGE_TO_LEFT_LANE"]],  # CHANGE_TO_LEFT_LANE_LLM → CHANGE_TO_LEFT_LANE
                      
      lambda args: 1 - args[self.predicate["CHANGE_TO_RIGHT_LANE_LLM"]] + args[self.predicate["CHANGE_TO_RIGHT_LANE_LLM"]] * \
                      args[self.predicate["CHANGE_TO_RIGHT_LANE"]],  # CHANGE_TO_RIGHT_LANE_LLM → CHANGE_TO_RIGHT_LANE
                      
      lambda args: 1 - args[self.predicate["PARK_LLM"]] + args[self.predicate["PARK_LLM"]] * \
                      args[self.predicate["PARK"]],  # PARK_LLM → PARK
                      
      lambda args: 1 - args[self.predicate["PULL_OVER_LLM"]] + args[self.predicate["PULL_OVER_LLM"]] * \
                      args[self.predicate["PULL_OVER"]]  # PULL_OVER_LLM → PULL_OVER            
    ]
  
  def generate_compliant_data(self):
    data = []
    for rule in self.formulas:
        args = np.zeros(self.action_num+self.condition_num)
        if "MAKE_LEFT_TURN" in rule.__code__.co_consts:
            args[self.predicate["MAKE_LEFT_TURN"]] = 1
        elif "MAKE_RIGHT_TURN" in rule.__code__.co_consts:
            args[self.predicate["MAKE_RIGHT_TURN"]] = 1
        else:
            args[np.random.choice(list(range(16)))] = 1
        condition_indices = [i for name, i in self.predicate.items() if name in rule.__code__.co_consts and i >= 16 and i < 28]
        print(condition_indices)
        if condition_indices:
          for index in condition_indices:
              args[index] = 1
        args[np.random.choice(list(range(28, 33)))] = 1
        args[np.random.choice(list(range(34, 36)))] = 1
        
        if rule(args):
            data.append(list(args))
    return data
  
  
  def generate_noncompliant_data(self):
    data = []
    while len(data) < len(self.formulas):
        args = np.zeros(self.action_num+self.condition_num)
        args[np.random.choice(list(range(16)))] = 1
        selected_conditions = np.random.choice(list(range(16, 28)), np.random.randint(0, 3), replace=False)
        for index in selected_conditions:
            args[index] = 1
        args[np.random.choice(list(range(28, 33)))] = 1
        args[np.random.choice(list(range(34, 36)))] = 1
        if not any(rule(args) for rule in self.formulas):
            data.append(args)
    return data
  
  
  def generate_dataset(self, total_samples, real_data, positive_rate=0.8):
    num_compliant = int(total_samples * positive_rate)  # 80% synthetic positive samples
    num_noncompliant = total_samples - num_compliant  # 20% synthetic negative samples

    # Generate synthetic compliant data
    compliant_data = []
    while len(compliant_data) < num_compliant:
        compliant_data.extend(self.generate_compliant_data())
        if len(compliant_data) > num_compliant:
            compliant_data = compliant_data[:num_compliant]

    # Generate synthetic noncompliant data
    noncompliant_data = []
    while len(noncompliant_data) < num_noncompliant:
        noncompliant_data.extend(self.generate_noncompliant_data())
        if len(noncompliant_data) > num_noncompliant:
            noncompliant_data = noncompliant_data[:num_noncompliant]

    # Perturb real_data to create real noncompliant data
    perturbed_real_data = []
    for sample in real_data:
        if np.random.rand() < 1-positive_rate:
            perturbed_sample = sample.copy()
            action_indices = list(range(16))  # Indices corresponding to action predicates
            current_action_index = np.argmax(perturbed_sample[action_indices])  # Find current action
            perturbed_sample[current_action_index] = 0  # Set the current action to 0
            remaining_actions = action_indices.copy()
            remaining_actions.remove(current_action_index)  # Remove the current action index from the list
            new_action_index = np.random.choice(remaining_actions)  # Select a new action index randomly
            perturbed_sample[new_action_index] = 1  # Set the new action
            perturbed_real_data.append(perturbed_sample)

    # Combine synthetic data with real data
    pos_dataset = compliant_data + real_data 
    neg_dataset = noncompliant_data + perturbed_real_data
    np.random.shuffle(pos_dataset)  # Shuffle the combined dataset
    np.random.shuffle(neg_dataset)
    return pos_dataset, neg_dataset
  

def balance_dataset(dataset, action_num, target_ratio=0.1):
    """
    Balance the dataset by ensuring each action appears with a target ratio.

    Parameters:
    - dataset: List of numpy arrays representing the dataset where each instance has actions and conditions.
    - action_num: Total number of action predicates.
    - target_ratio: Target ratio for each action in the dataset (default is 0.1).

    Returns:
    - Balanced dataset as a list of numpy arrays.
    """

    # Convert dataset to numpy array for easier processing
    dataset = np.array(dataset)

    # Count the occurrences of each action
    action_counts = np.zeros(action_num)
    for instance in dataset:
        action_indices = list(range(action_num))
        action_counts += instance[action_indices]

    # Calculate target number of samples for each action
    total_samples = len(dataset)
    target_counts = total_samples * target_ratio

    # Create a dictionary to store indices for each action
    action_indices_dict = {i: [] for i in range(action_num)}
    for index, instance in enumerate(dataset):
        action_indices = list(range(action_num))
        for action_index in action_indices:
            if instance[action_index] == 1:
                action_indices_dict[action_index].append(index)

    # Create balanced dataset
    balanced_dataset = []
    for action_index in range(action_num):
        indices = action_indices_dict[action_index]
        if len(indices) < target_counts:
            # Need to oversample
            additional_indices = np.random.choice(indices, int(target_counts - len(indices)), replace=True)
            balanced_dataset.extend(dataset[indices + additional_indices])
        else:
            # Need to undersample
            sampled_indices = np.random.choice(indices, int(target_counts), replace=False)
            balanced_dataset.extend(dataset[sampled_indices])

    # Shuffle the balanced dataset
    balanced_dataset = np.array(balanced_dataset)
    np.random.shuffle(balanced_dataset)

    return balanced_dataset.tolist()
  
  
class DriveLM:
  def __init__(self):
    self.predicate = {
      'NORMAL': 0,
      'FAST': 1,
      'SLOW': 2,
      'STOP': 3,
      'LEFT': 4,
      'RIGHT': 5,
      'STRAIGHT': 6,
      
      'SOLID_RED_LIGHT': 7,
      'SOLID_YELLOW_LIGHT': 8,
      'YELLOW_LEFT_ARROW_LIGHT': 9,
      'RED_LEFT_ARROW_LIGHT': 10,
      'MERGING_TRAFFIC_SIGN': 11,
      'NO_LEFT_TURN_SIGN': 12,
      'NO_RIGHT_TURN_SIGN': 13,
      'PEDESTRIAN_CROSSING_SIGN': 14,
      'STOP_SIGN': 15,
      'RED_YIELD_SIGN': 16,
      'SLOW_SIGN': 17,
      'SOLID_GREEN_LIGHT': 18,
          
      'DOUBLE_DASHED_WHITE_LINE_LEFT': 19,
      'DOUBLE_DASHED_WHITE_LINE_RIGHT': 20,
      'SINGLE_SOLID_WHITE_LINE_LEFT': 21,
      'SINGLE_SOLID_WHITE_LINE_RIGHT': 22,
      'DOUBLE_SOLID_WHITE_LINE_LEFT': 23,
      'DOUBLE_SOLID_WHITE_LINE_RIGHT': 24,
      'SINGLE_ZIGZAG_WHITE_LINE_LEFT': 25,
      'SINGLE_ZIGZAG_WHITE_LINE_RIGHT': 26,
      'SINGLE_SOLID_YELLOW_LINE_LEFT': 27,
      'SINGLE_SOLID_YELLOW_LINE_RIGHT': 28,
      
      'NORMAL_CS': 29,
      'FAST_CS': 30,
      'SLOW_CS': 31,
      'STOP_CS': 32,
      'LEFT_CS': 33,
      'RIGHT_CS': 34,
      'STRAIGHT_CS': 35,
      
      'NORMAL_LLM': 36,
      'FAST_LLM': 37,
      'SLOW_LLM': 38,
      'STOP_LLM': 39,
      'LEFT_LLM': 40,
      'RIGHT_LLM': 41,
      'STRAIGHT_LLM': 42
    }
    
    self.velocity_action_num = 4
    self.direction_action_num = 3
    self.action_num = 7
    self.condition_num = 36
      
    self.formulas = [      
      
      lambda args: 1 - args[self.predicate["SOLID_RED_LIGHT"]] + args[self.predicate["FAST"]] * \
                      (1 - args[self.predicate["FAST"]]), # SolidRedLight → ¬Fast
                      
      lambda args: 1 - args[self.predicate["SOLID_YELLOW_LIGHT"]] + args[self.predicate["FAST"]] * \
                      (1 - args[self.predicate["FAST"]]), # SolidYellowLight → ¬Fast
                      
      lambda args: 1 - args[self.predicate["YELLOW_LEFT_ARROW_LIGHT"]] + args[self.predicate["YELLOW_LEFT_ARROW_LIGHT"]] * \
                      (args[self.predicate["STOP"]] + args[self.predicate["SLOW"]] - \
                      args[self.predicate["STOP"]] * args[self.predicate["SLOW"]]),  # YellowLeftArrowLight → Stop ∨ Slow

      lambda args: 1 - args[self.predicate["RED_LEFT_ARROW_LIGHT"]] + args[self.predicate["RED_LEFT_ARROW_LIGHT"]] * \
                      (1 - args[self.predicate["LEFT"]]),  # RedLeftArrowLight → ¬Left
      
      lambda args: 1 - args[self.predicate["MERGING_TRAFFIC_SIGN"]] + args[self.predicate["MERGING_TRAFFIC_SIGN"]] * \
                     (1- args[self.predicate["FAST"]]),  # MergingTrafficSign → ¬Fast

      lambda args: 1 - args[self.predicate["NO_LEFT_TURN_SIGN"]] + args[self.predicate["NO_LEFT_TURN_SIGN"]] * \
                      (1 - args[self.predicate["LEFT"]]),  # NoLeftTurnSign → ¬Left

      lambda args: 1 - args[self.predicate["NO_RIGHT_TURN_SIGN"]] + args[self.predicate["NO_RIGHT_TURN_SIGN"]] * \
                      (1 - args[self.predicate["RIGHT"]]),  # NoRightTurnSign → ¬Right
      
      lambda args: 1 - args[self.predicate["RED_YIELD_SIGN"]] + args[self.predicate["RED_YIELD_SIGN"]] * \
                      (1-args[self.predicate["FAST"]]),  # RedYieldSign → ¬Fast  

      lambda args: 1 - args[self.predicate["SLOW_SIGN"]] + args[self.predicate["SLOW_SIGN"]] * \
                      (1 - args[self.predicate["FAST"]]),  # SlowSign → ¬Fast  
                      
      lambda args: 1 - args[self.predicate["SINGLE_SOLID_WHITE_LINE_LEFT"]] + args[self.predicate["SINGLE_SOLID_WHITE_LINE_LEFT"]] * \
                      (1 - args[self.predicate["LEFT"]]),  # SingleSolidWhiteLineLeft → ¬Left
                      
      lambda args: 1 - args[self.predicate["SINGLE_SOLID_WHITE_LINE_RIGHT"]] + args[self.predicate["SINGLE_SOLID_WHITE_LINE_RIGHT"]] * \
                      (1 - args[self.predicate["RIGHT"]]),  # SingleSolidWhiteLineRight → ¬Right
                      
      lambda args: 1 - args[self.predicate["DOUBLE_SOLID_WHITE_LINE_LEFT"]] + args[self.predicate["DOUBLE_SOLID_WHITE_LINE_LEFT"]] * \
                      (1 - args[self.predicate["LEFT"]]),  # DOUBLE_SOLID_WHITE_LINE_LEFT → ¬Left
                      
      lambda args: 1 - args[self.predicate["DOUBLE_SOLID_WHITE_LINE_RIGHT"]] + args[self.predicate["DOUBLE_SOLID_WHITE_LINE_RIGHT"]] * \
                      (1 - args[self.predicate["RIGHT"]]),  # DOUBLE_SOLID_WHITE_LINE_RIGHT → ¬Right

      lambda args: 1 - args[self.predicate["SINGLE_ZIGZAG_WHITE_LINE_LEFT"]] + args[self.predicate["SINGLE_ZIGZAG_WHITE_LINE_LEFT"]] * \
                      (1 - args[self.predicate["STOP"]]),  # SingleZigzagWhiteLineLeft → ¬Stop

      lambda args: 1 - args[self.predicate["SINGLE_ZIGZAG_WHITE_LINE_RIGHT"]] + args[self.predicate["SINGLE_ZIGZAG_WHITE_LINE_RIGHT"]] * \
                      (1 - args[self.predicate["STOP"]]),  # SingleZigzagWhiteLineRight → ¬Stop 
                      
      lambda args: 1 - args[self.predicate["NORMAL_CS"]] + args[self.predicate["NORMAL_CS"]] * \
                      args[self.predicate["NORMAL"]],  # NORMAL_CS → NORMAL
                      
      lambda args: 1 - args[self.predicate["FAST_CS"]] + args[self.predicate["FAST_CS"]] * \
                      args[self.predicate["FAST"]],  # FAST_CS → FAST
                      
      lambda args: 1 - args[self.predicate["SLOW_CS"]] + args[self.predicate["SLOW_CS"]] * \
                      args[self.predicate["SLOW"]],  # SLOW_CS → SLOW
                      
      lambda args: 1 - args[self.predicate["STOP_CS"]] + args[self.predicate["STOP_CS"]] * \
                      args[self.predicate["STOP"]],  # STOP_CS → STOP 
                      
      lambda args: 1 - args[self.predicate["LEFT_CS"]] + args[self.predicate["LEFT_CS"]] * \
                      args[self.predicate["LEFT"]],  # LEFT_CS → LEFT
                      
      lambda args: 1 - args[self.predicate["RIGHT_CS"]] + args[self.predicate["RIGHT_CS"]] * \
                      args[self.predicate["RIGHT"]],  # RIGHT_CS → RIGHT
                      
      lambda args: 1 - args[self.predicate["STRAIGHT_CS"]] + args[self.predicate["STRAIGHT_CS"]] * \
                      args[self.predicate["STRAIGHT"]],  # STRAIGHT_CS → STRAIGHT
                      
      lambda args: 1 - args[self.predicate["NORMAL_LLM"]] + args[self.predicate["NORMAL_LLM"]] * \
                      args[self.predicate["NORMAL"]],  # NORMAL_LLM → NORMAL
                      
      lambda args: 1 - args[self.predicate["FAST_LLM"]] + args[self.predicate["FAST_LLM"]] * \
                      args[self.predicate["FAST"]],  # FAST_LLM → FAST
                      
      lambda args: 1 - args[self.predicate["SLOW_LLM"]] + args[self.predicate["SLOW_LLM"]] * \
                      args[self.predicate["SLOW"]],  # SLOW_LLM → SLOW
                      
      lambda args: 1 - args[self.predicate["STOP_LLM"]] + args[self.predicate["STOP_LLM"]] * \
                      args[self.predicate["STOP"]],  # STOP_LLM → STOP
                      
      lambda args: 1 - args[self.predicate["LEFT_LLM"]] + args[self.predicate["LEFT_LLM"]] * \
                      args[self.predicate["LEFT"]],  # LEFT_LLM → LEFT
                      
      lambda args: 1 - args[self.predicate["RIGHT_LLM"]] + args[self.predicate["RIGHT_LLM"]] * \
                      args[self.predicate["RIGHT"]],  # RIGHT_LLM → RIGHT
                      
      lambda args: 1 - args[self.predicate["STRAIGHT_LLM"]] + args[self.predicate["STRAIGHT_LLM"]] * \
                      args[self.predicate["STRAIGHT"]]  # STRAIGHT_LLM → STRAIGHT 
    ] 
                      
      
      
  

    
            
