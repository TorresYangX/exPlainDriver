class BDDX:
  def __init__(self):
      self.KEEP = 0
      self.ACCELERATE = 1
      self.DECELAERATE = 2
      self.STOP = 3
      self.REVERSE = 4
      self.MAKE_LEFT_TURN = 5
      self.MAKE_RIGHT_TURN = 6
      self.MAKE_U_TURN = 7
      self.MERGE = 8
      self.LEFT_PASS = 9
      self.RIGHT_PASS = 10
      self.YIELD = 11
      self.CHANGE_TO_LEFT_LANE = 12
      self.CHANGE_TO_RIGHT_LANE = 13
      self.CHANGE_TO_CENTER_LEFT_TURN_LANE = 14
      self.PARK = 15
      self.PULL_OVER = 16
      self.SOLID_RED_LIGHT = 17
      self.SOLID_YELLOW_LIGHT = 18
      self.YELLOW_LEFT_ARROW_LIGHT = 19
      self.RED_LEFT_ARROW_LIGHT = 20
      self.MERGING_TRAFFIC_SIGN = 21
      self.WRONG_WAY_SIGN = 22
      self.NO_LEFT_TURN_SIGN = 23
      self.NO_RIGHT_TURN_SIGN = 24
      self.PEDESTRIAN_CROSSING_SIGN = 25
      self.STOP_SIGN = 26
      self.RED_YIELD_SIGN = 27
      self.DO_NOT_PASS_SIGN = 28
      self.SLOW_SIGN = 29
      
      self.action_num = 17
      self.condition_num = 13
      
      self.formulas = [
          lambda args: 1 - args[self.SOLID_RED_LIGHT] + args[self.SOLID_RED_LIGHT] * 
                          ((args[self.DECELAERATE] + args[self.STOP] - args[self.DECELAERATE] * 
                            args[self.STOP]) * (1 - args[self.ACCELERATE])),  # SolidRedLight → Decelerate ∨ Stop ∧ ¬Accelerate,
          lambda args: 1 - args[self.SOLID_YELLOW_LIGHT] + args[self.SOLID_YELLOW_LIGHT] * 
                          ((args[self.STOP] + args[self.DECELAERATE] - args[self.STOP] * 
                            args[self.DECELAERATE]) * (1 - args[self.ACCELERATE])),  # SolidYellowLight → Stop ∨ Decelerate ∧ ¬Accelerate
          lambda args: 1 - args[self.YELLOW_LEFT_ARROW_LIGHT] + args[self.YELLOW_LEFT_ARROW_LIGHT] * 
                          (args[self.STOP] + args[self.DECELAERATE] - args[self.STOP] * 
                            args[self.DECELAERATE]),  # YellowLeftArrowLight → Stop ∨ Decelerate
          lambda args: 1 - args[self.RED_LEFT_ARROW_LIGHT] + args[self.RED_LEFT_ARROW_LIGHT] * 
                          (1 - (args[self.MAKE_LEFT_TURN] + args[self.MAKE_U_TURN] - args[self.MAKE_LEFT_TURN] * 
                                args[self.MAKE_U_TURN])),  # RedLeftArrowLight → ¬(MakeLeftTurn ∨ MakeUTurn)
          lambda args: 1 - args[self.MERGING_TRAFFIC_SIGN] + args[self.MERGING_TRAFFIC_SIGN] * 
                          args[self.DECELAERATE],  # MergingTrafficSign → Decelerate
          lambda args: 1 - args[self.WRONG_WAY_SIGN] + args[self.WRONG_WAY_SIGN] * 
                          (args[self.STOP] + args[self.MAKE_U_TURN] + args[self.REVERSE] - args[self.STOP] * 
                            args[self.MAKE_U_TURN] - args[self.STOP] * args[self.REVERSE] - args[self.MAKE_U_TURN] * 
                            args[self.REVERSE] + args[self.STOP] * args[self.MAKE_U_TURN] * args[self.REVERSE]),  # WrongWaySign → Stop ∨ Reverse ∨ MakeUTurn
          lambda args: 1 - args[self.NO_LEFT_TURN_SIGN] + args[self.NO_LEFT_TURN_SIGN] * 
                          (1 - args[self.MAKE_LEFT_TURN]),  # NoLeftTurnSign → ¬MakeLeftTurn
          lambda args: 1 - args[self.NO_RIGHT_TURN_SIGN] + args[self.NO_RIGHT_TURN_SIGN] * 
                          (1 - args[self.MAKE_RIGHT_TURN]),  # NoRightTurnSign → ¬MakeRightTurn
          lambda args: 1 - args[self.PEDESTRIAN_CROSSING_SIGN] + args[self.PEDESTRIAN_CROSSING_SIGN] * (
                          (args[self.DECELAERATE] + args[self.STOP] + args[self.KEEP] - args[self.DECELAERATE] * args[self.STOP] -
                          args[self.DECELAERATE] * args[self.KEEP] - args[self.STOP] * args[self.KEEP] +
                          args[self.DECELAERATE] * args[self.STOP] * args[self.KEEP]) * (1 - args[self.ACCELERATE])
                          ),  # PedestrianCrossingSign → Decelerate ∨ Stop ∨ Keep ∧ ¬Accelerate
          lambda args: 1 - args[self.STOP_SIGN] + args[self.STOP_SIGN] * 
                      ((args[self.STOP] + args[self.DECELAERATE] - args[self.STOP] * args[self.DECELAERATE]) 
                        * (1 - args[self.ACCELERATE])),  # StopSign → Decelerate ∨ Stop ∧ ¬Accelerate
          lambda args: 1 - args[self.RED_YIELD_SIGN] + args[self.RED_YIELD_SIGN] * 
                          args[self.DECELAERATE], # RedYieldSign → Decelerate
          lambda args: 1 - args[self.DO_NOT_PASS_SIGN] + args[self.DO_NOT_PASS_SIGN] * 
                      (1 - (args[self.LEFT_PASS] + args[self.RIGHT_PASS] - args[self.LEFT_PASS] * 
                            args[self.RIGHT_PASS])), # DoNotPassSign → ¬(LeftPass ∨ RightPass)
          lambda args: 1 - args[self.SLOW_SIGN] + args[self.SLOW_SIGN] * args[self.DECELAERATE] # SlowSign → Decelerate
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
      
            
