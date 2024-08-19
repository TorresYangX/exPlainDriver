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
    }
      
    self.action_num = 16
    self.condition_num = 20
      
    self.formulas = [      
      lambda args: args[self.predicate["KEEP"]], # KEEP
      lambda args: args[self.predicate["ACCELERATE"]], # ACCELERATE
      lambda args: args[self.predicate["DECELERATE"]], # DECELERATE
      lambda args: args[self.predicate["STOP"]], # STOP
      lambda args: args[self.predicate["REVERSE"]], # REVERSE
      lambda args: args[self.predicate["MAKE_LEFT_TURN"]], # MAKE_LEFT_TURN
      lambda args: args[self.predicate["MAKE_RIGHT_TURN"]], # MAKE_RIGHT_TURN
      lambda args: args[self.predicate["MAKE_U_TURN"]], # MAKE_U_TURN
      lambda args: args[self.predicate["MERGE"]], # MERGE
      lambda args: args[self.predicate["LEFT_PASS"]], # LEFT_PASS
      lambda args: args[self.predicate["RIGHT_PASS"]], # RIGHT
      lambda args: args[self.predicate["YIELD"]], # YIELD
      lambda args: args[self.predicate["CHANGE_TO_LEFT_LANE"]], # CHANGE_TO_LEFT_LANE
      lambda args: args[self.predicate["CHANGE_TO_RIGHT_LANE"]], # CHANGE_TO_RIGHT_LANE
      lambda args: args[self.predicate["PARK"]], # PARK
      lambda args: args[self.predicate["PULL_OVER"]], # PULL_OVER
      # lambda args: 1 - args[self.predicate["SOLID_RED_LIGHT"]] + args[self.predicate["SOLID_RED_LIGHT"]] * \
      #                 ((args[self.predicate["KEEP"]] + args[self.predicate["DECELERATE"]] + args[self.predicate["STOP"]] - \
      #                   args[self.predicate["KEEP"]] * args[self.predicate["DECELERATE"]] - \
      #                   args[self.predicate["KEEP"]] * args[self.predicate["STOP"]] - \
      #                   args[self.predicate["DECELERATE"]] * args[self.predicate["STOP"]] + \
      #                   args[self.predicate["KEEP"]] * args[self.predicate["DECELERATE"]] * args[self.predicate["STOP"]]) * \
      #                   (1 - args[self.predicate["ACCELERATE"]])), # SolidRedLight → Keep ∨ Decelerate ∨ Stop ∧ ¬Accelerate,
      
      lambda args: 1 - args[self.predicate["SOLID_RED_LIGHT"]] + args[self.predicate["SOLID_RED_LIGHT"]] * \
                      ((1 - args[self.predicate["ACCELERATE"]]) * \
                       (1 - args[self.predicate["PULL_OVER"]]) * \
                       (1 - args[self.predicate["PARK"]])), # SolidRedLight → ¬Accelerate ∧ ¬PullOver ∧ ¬Park

      
      # lambda args: 1 - args[self.predicate["SOLID_YELLOW_LIGHT"]] + args[self.predicate["SOLID_YELLOW_LIGHT"]] * \
      #                 ((args[self.predicate["KEEP"]] + args[self.predicate["STOP"]] + args[self.predicate["DECELERATE"]] - \
      #                   args[self.predicate["KEEP"]] * args[self.predicate["STOP"]] - \
      #                   args[self.predicate["KEEP"]] * args[self.predicate["DECELERATE"]] - \
      #                   args[self.predicate["STOP"]] * args[self.predicate["DECELERATE"]] + \
      #                   args[self.predicate["KEEP"]] * args[self.predicate["STOP"]] * args[self.predicate["DECELERATE"]]) * \
      #                   (1 - args[self.predicate["ACCELERATE"]])), # SolidYellowLight → Keep ∨ Stop ∨ Decelerate ∧ ¬Accelerate
      
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

      # lambda args: 1 - args[self.predicate["PEDESTRIAN_CROSSING_SIGN"]] + args[self.predicate["PEDESTRIAN_CROSSING_SIGN"]] * \
      #                 ((args[self.predicate["DECELERATE"]] + args[self.predicate["STOP"]] + args[self.predicate["KEEP"]] - \
      #                 args[self.predicate["DECELERATE"]] * args[self.predicate["STOP"]] - \
      #                 args[self.predicate["DECELERATE"]] * args[self.predicate["KEEP"]] - \
      #                 args[self.predicate["STOP"]] * args[self.predicate["KEEP"]] + \
      #                 args[self.predicate["DECELERATE"]] * args[self.predicate["STOP"]] * args[self.predicate["KEEP"]]) * \
      #                 (1 - args[self.predicate["ACCELERATE"]])),  # PedestrianCrossingSign → Decelerate ∨ Stop ∨ Keep ∧ ¬Accelerate
      
      # lambda args: 1 - args[self.predicate["PEDESTRIAN_CROSSING_SIGN"]] + args[self.predicate["PEDESTRIAN_CROSSING_SIGN"]] * \
      #                 (1 - args[self.predicate["ACCELERATE"]]), # PedestrianCrossingSign → ¬Accelerate

      # lambda args: 1 - args[self.predicate["STOP_SIGN"]] + args[self.predicate["STOP_SIGN"]] * \
      #                 ((args[self.predicate["KEEP"]] + args[self.predicate["STOP"]] + args[self.predicate["DECELERATE"]] - \
      #                   args[self.predicate["KEEP"]] * args[self.predicate["STOP"]] - \
      #                   args[self.predicate["KEEP"]] * args[self.predicate["DECELERATE"]] - \
      #                   args[self.predicate["STOP"]] * args[self.predicate["DECELERATE"]] + \
      #                   args[self.predicate["KEEP"]] * args[self.predicate["STOP"]] * args[self.predicate["DECELERATE"]]) * \
      #                   (1 - args[self.predicate["ACCELERATE"]])), # StopSign → Keep ∨ Decelerate ∨ Stop ∧ ¬Accelerate
      
      # lambda args: 1 - args[self.predicate["STOP_SIGN"]] + args[self.predicate["STOP_SIGN"]] * \
      #                 (1 - args[self.predicate["ACCELERATE"]]), # StopSign → ¬Accelerate

      lambda args: 1 - args[self.predicate["RED_YIELD_SIGN"]] + args[self.predicate["RED_YIELD_SIGN"]] * \
                      args[self.predicate["DECELERATE"]],  # RedYieldSign → Decelerate

      lambda args: 1 - args[self.predicate["SLOW_SIGN"]] + args[self.predicate["SLOW_SIGN"]] * \
                      (1 - args[self.predicate["ACCELERATE"]]),  # SlowSign → ¬Accelerate
                      
      # lambda args: 1 - args[self.predicate["SOLID_GREEN_LIGHT"]] + args[self.predicate["SOLID_GREEN_LIGHT"]] * \
      #                 (args[self.predicate["KEEP"]] + args[self.predicate["ACCELERATE"]] - \
      #                  args[self.predicate["KEEP"]] * args[self.predicate["ACCELERATE"]]), # SolidGreenLight → Keep ∨ Accelerate

      lambda args: 1 - args[self.predicate["KEEP_CS"]] + args[self.predicate["KEEP_CS"]] * \
                      args[self.predicate["KEEP"]],  # KEEP_CS → Keep

      lambda args: 1 - args[self.predicate["ACCELERATE_CS"]] + args[self.predicate["ACCELERATE_CS"]] * \
                      args[self.predicate["ACCELERATE"]],  # ACCELERATE_CS → Accelerate
                      
      lambda args: 1 - args[self.predicate["DECELERATE_CS"]] + args[self.predicate["DECELERATE_CS"]] * \
                      args[self.predicate["DECELERATE"]],  # DECELERATE_CS → DECELERATE
                      
      lambda args: 1 - args[self.predicate["STOP_CS"]] + args[self.predicate["STOP_CS"]] * \
                      args[self.predicate["STOP"]],  # STOP_CS → Stop
      
      
      # lambda args: 1 - args[self.predicate["KEEP_CS"]] + args[self.predicate["KEEP_CS"]] * \
      #                 (args[self.predicate["KEEP"]] + args[self.predicate["ACCELERATE"]] - \
      #                  args[self.predicate["KEEP"]] * args[self.predicate["ACCELERATE"]]),# KEEP_CS → Keep ∨ Accelerate
                      
      # lambda args: 1 - args[self.predicate["ACCELERATE_CS"]] + args[self.predicate["ACCELERATE_CS"]] * \
      #                 (args[self.predicate["KEEP"]] + args[self.predicate["ACCELERATE"]] - \
      #                  args[self.predicate["KEEP"]] * args[self.predicate["ACCELERATE"]]),# ACCELERATE_CS → Keep ∨ Accelerate
      
      # lambda args: 1 - args[self.predicate["DECELERATE_CS"]] + args[self.predicate["DECELERATE_CS"]] * \
      #                 (args[self.predicate["DECELERATE"]] + args[self.predicate["STOP"]] - \
      #                  args[self.predicate["DECELERATE"]] * args[self.predicate["STOP"]]),#DECELERATE_CS → DECELERATE ∨ Stop
                      
      # lambda args: 1 - args[self.predicate["STOP_CS"]] + args[self.predicate["STOP_CS"]] * \
      #                 (args[self.predicate["DECELERATE"]] + args[self.predicate["STOP"]] - \
      #                  args[self.predicate["DECELERATE"]] * args[self.predicate["STOP"]]),#STOP_CS → DECELERATE ∨ Stop
                      
      lambda args: 1 - args[self.predicate["REVERSE_CS"]] + args[self.predicate["REVERSE_CS"]] * \
                      args[self.predicate["REVERSE"]],  # REVERSE_CS → REVERSE
                      
      lambda args: 1 - args[self.predicate["LEFT_CS"]] + args[self.predicate["LEFT_CS"]] * \
                      args[self.predicate["MAKE_LEFT_TURN"]],  # LEFT_CS → MakeLeftTurn
                      
      lambda args: 1 - args[self.predicate["RIGHT_CS"]] + args[self.predicate["RIGHT_CS"]] * \
                      args[self.predicate["MAKE_RIGHT_TURN"]],  # RIGHT_CS → MakeRightTurn  

    ]
    
    self.nature_rule=[
      '1. When you encounter a solid red light, you should stop. \n2. You can turn right on a red light unless a \'NO TURN ON RED\' sign is posted;',
      'When approaching a solid yellow traffic light, you should stop if you can do so safely. If you cannot stop safely, you should cautiously cross the intersection.',
      'When you see a yellow arrow, you should prepare to stop, as the protected turning time is ending, unless you cannot stop safely OR you are already in the intersection, in which case you should cautiously complete your turn.',
      'When you see a red arrow, you should stop and not make any turns. Remain stopped until a green traffic signal light or green arrow appears.',
      'When you see a Merging Traffic Sign, you should decelarate and be prepared to allow other drivers to merge into your lane.',
      'When you encounter a NO LEFT TURN sign, you should not make a left turn.',
      'When you encounter a NO RIGHT TURN sign, you should not make a right turn.',
      'When you encounter a Pedestrian Crossing Sign, you should DECELERATE and be prepared to stop for pedestrians.',
      'When approaching a STOP sign, you should make a full stop before entering the crosswalk OR at the limit line. If there is no limit line or crosswalk, you should stop before entering the intersection. After stopping, you should check traffic in all directions before proceeding.',
      'When approaching a red YIELD sign, you should decelerate AND be ready to stop to let any vehicle, bicyclist, OR pedestrian pass before you proceed.',
      'When you see a SLOW sign, you should decelerate.',
  ]
      
  def nature_rule_mapping(self, index):
    violate_rule = []
    for i in index:
      violate_rule.append(self.nature_rule[i])
    return violate_rule
    

    
            
