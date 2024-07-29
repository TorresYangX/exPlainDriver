Natural_Rule = [
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

def mapping_natural_rule(index):
    return Natural_Rule[index]


if __name__=="__main__":
    print(mapping_natural_rule(0))
    