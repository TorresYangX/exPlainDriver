class BDDX:
  def __init__(self):
    self.predicate = {
      'KEEP': 0,
      'ACCELERATE': 1,
      'DECELAERATE': 2,
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
      'CHANGE_TO_CENTER_LEFT_TURN_LANE': 14,
      'PARK': 15,
      'PULL_OVER': 16,
      'SOLID_RED_LIGHT': 17,
      'SOLID_YELLOW_LIGHT': 18,
      'YELLOW_LEFT_ARROW_LIGHT': 19,
      'RED_LEFT_ARROW_LIGHT': 20,
      'MERGING_TRAFFIC_SIGN': 21,
      'WRONG_WAY_SIGN': 22,
      'NO_LEFT_TURN_SIGN': 23,
      'NO_RIGHT_TURN_SIGN': 24,
      'PEDESTRIAN_CROSSING_SIGN': 25,
      'STOP_SIGN': 26,
      'RED_YIELD_SIGN': 27,
      'DO_NOT_PASS_SIGN': 28,
      'SLOW_SIGN': 29
    }
      
    self.action_num = 17
    self.condition_num = 13
      
    self.formulas = [
      lambda args: 1 - args[self.predicate["SOLID_RED_LIGHT"]] + args[self.predicate["SOLID_RED_LIGHT"]] * \
                      ((args[self.predicate["DECELAERATE"]] + args[self.predicate["STOP"]] - args[self.predicate["DECELAERATE"]] * \
                        args[self.predicate["STOP"]]) * (1 - args[self.predicate["ACCELERATE"]])),  # SolidRedLight → Decelerate ∨ Stop ∧ ¬Accelerate,
      
      lambda args: 1 - args[self.predicate["SOLID_YELLOW_LIGHT"]] + args[self.predicate["SOLID_YELLOW_LIGHT"]] * \
                      ((args[self.predicate["STOP"]] + args[self.predicate["DECELAERATE"]] - \
                      args[self.predicate["STOP"]] * args[self.predicate["DECELAERATE"]]) * \
                      (1 - args[self.predicate["ACCELERATE"]])),  # SolidYellowLight → Stop ∨ Decelerate ∧ ¬Accelerate

      lambda args: 1 - args[self.predicate["YELLOW_LEFT_ARROW_LIGHT"]] + args[self.predicate["YELLOW_LEFT_ARROW_LIGHT"]] * \
                      (args[self.predicate["STOP"]] + args[self.predicate["DECELAERATE"]] - \
                      args[self.predicate["STOP"]] * args[self.predicate["DECELAERATE"]]),  # YellowLeftArrowLight → Stop ∨ Decelerate

      lambda args: 1 - args[self.predicate["RED_LEFT_ARROW_LIGHT"]] + args[self.predicate["RED_LEFT_ARROW_LIGHT"]] * \
                      (1 - (args[self.predicate["MAKE_LEFT_TURN"]] + args[self.predicate["MAKE_U_TURN"]] - \
                      args[self.predicate["MAKE_LEFT_TURN"]] * args[self.predicate["MAKE_U_TURN"]])),  # RedLeftArrowLight → ¬(MakeLeftTurn ∨ MakeUTurn)

      lambda args: 1 - args[self.predicate["MERGING_TRAFFIC_SIGN"]] + args[self.predicate["MERGING_TRAFFIC_SIGN"]] * \
                      args[self.predicate["DECELAERATE"]],  # MergingTrafficSign → Decelerate

      lambda args: 1 - args[self.predicate["WRONG_WAY_SIGN"]] + args[self.predicate["WRONG_WAY_SIGN"]] * \
                      (args[self.predicate["STOP"]] + args[self.predicate["MAKE_U_TURN"]] + args[self.predicate["REVERSE"]] - \
                      args[self.predicate["STOP"]] * args[self.predicate["MAKE_U_TURN"]] - \
                      args[self.predicate["STOP"]] * args[self.predicate["REVERSE"]] - \
                      args[self.predicate["MAKE_U_TURN"]] * args[self.predicate["REVERSE"]] + \
                      args[self.predicate["STOP"]] * args[self.predicate["MAKE_U_TURN"]] * args[self.predicate["REVERSE"]]),  # WrongWaySign → Stop ∨ Reverse ∨ MakeUTurn

      lambda args: 1 - args[self.predicate["NO_LEFT_TURN_SIGN"]] + args[self.predicate["NO_LEFT_TURN_SIGN"]] * \
                      (1 - args[self.predicate["MAKE_LEFT_TURN"]]),  # NoLeftTurnSign → ¬MakeLeftTurn

      lambda args: 1 - args[self.predicate["NO_RIGHT_TURN_SIGN"]] + args[self.predicate["NO_RIGHT_TURN_SIGN"]] * \
                      (1 - args[self.predicate["MAKE_RIGHT_TURN"]]),  # NoRightTurnSign → ¬MakeRightTurn

      lambda args: 1 - args[self.predicate["PEDESTRIAN_CROSSING_SIGN"]] + args[self.predicate["PEDESTRIAN_CROSSING_SIGN"]] * \
                      ((args[self.predicate["DECELAERATE"]] + args[self.predicate["STOP"]] + args[self.predicate["KEEP"]] - \
                      args[self.predicate["DECELAERATE"]] * args[self.predicate["STOP"]] - \
                      args[self.predicate["DECELAERATE"]] * args[self.predicate["KEEP"]] - \
                      args[self.predicate["STOP"]] * args[self.predicate["KEEP"]] + \
                      args[self.predicate["DECELAERATE"]] * args[self.predicate["STOP"]] * args[self.predicate["KEEP"]]) * \
                      (1 - args[self.predicate["ACCELERATE"]])),  # PedestrianCrossingSign → Decelerate ∨ Stop ∨ Keep ∧ ¬Accelerate

      lambda args: 1 - args[self.predicate["STOP_SIGN"]] + args[self.predicate["STOP_SIGN"]] * \
                      ((args[self.predicate["STOP"]] + args[self.predicate["DECELAERATE"]] - \
                      args[self.predicate["STOP"]] * args[self.predicate["DECELAERATE"]]) * \
                      (1 - args[self.predicate["ACCELERATE"]])),  # StopSign → Decelerate ∨ Stop ∧ ¬Accelerate

      lambda args: 1 - args[self.predicate["RED_YIELD_SIGN"]] + args[self.predicate["RED_YIELD_SIGN"]] * \
                      args[self.predicate["DECELAERATE"]],  # RedYieldSign → Decelerate

      lambda args: 1 - args[self.predicate["DO_NOT_PASS_SIGN"]] + args[self.predicate["DO_NOT_PASS_SIGN"]] * \
                      (1 - (args[self.predicate["LEFT_PASS"]] + args[self.predicate["RIGHT_PASS"]] - \
                      args[self.predicate["LEFT_PASS"]] * args[self.predicate["RIGHT_PASS"]])),  # DoNotPassSign → ¬(LeftPass ∨ RightPass)

      lambda args: 1 - args[self.predicate["SLOW_SIGN"]] + args[self.predicate["SLOW_SIGN"]] * \
                      args[self.predicate["DECELAERATE"]]  # SlowSign → Decelerate

    ]
    
    self.nature_rule=[
      '1. When you encounter a solid red light, you should stop. \n2. You can turn right on a red light unless a \'NO TURN ON RED\' sign is posted;',
      'When approaching a solid yellow traffic light, you should stop if you can do so safely. If you cannot stop safely, you should cautiously cross the intersection.',
      'When you see a yellow arrow, you should prepare to stop, as the protected turning time is ending, unless you cannot stop safely OR you are already in the intersection, in which case you should cautiously complete your turn.',
      'When you see a red arrow, you should stop and not make any turns. Remain stopped until a green traffic signal light or green arrow appears.',
      'When you see a Merging Traffic Sign, you should decelarate and be prepared to allow other drivers to merge into your lane.',
      'When you encounter a WRONG WAY sign, you should not enter the roadway.',
      'When you encounter a NO LEFT TURN sign, you should not make a left turn.',
      'When you encounter a NO RIGHT TURN sign, you should not make a right turn.',
      'When you encounter a Pedestrian Crossing Sign, you should decelaerate and be prepared to stop for pedestrians.',
      'When approaching a STOP sign, you should make a full stop before entering the crosswalk OR at the limit line. If there is no limit line or crosswalk, you should stop before entering the intersection. After stopping, you should check traffic in all directions before proceeding.',
      'When approaching a red YIELD sign, you should decelerate AND be ready to stop to let any vehicle, bicyclist, OR pedestrian pass before you proceed.',
      'When you see a DO NOT PASS sign, you should not make left pass or right pass.',
      'When you see a SLOW sign, you should decelerate.',
  ]
      
  def nature_rule_mapping(self, index):
    violate_rule = []
    for i in index:
      violate_rule.append(self.nature_rule[i])
    return violate_rule
      
            
