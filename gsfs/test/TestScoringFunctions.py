import unittest

import numpy as np
import unittest

class TestScoringFunctions(unittest.TestCase):
        
    def test_getting_score(self):        
        parent = Node(set('A'),'B')  
        child = Node(set(['A','B']),'C')
        gs = GlobalScores()

        gs.update_score(set(['A','B']),0.5)
        gs.update_score(set(['A']),0.2)
        gs.update_score(set(['A','B','C']),0.7)
        gs.update_score(set(['A','B','C','D']),0.8)
        gs.update_score(set(['A','B','C','E']),0.3) # -----

        parent.add_score(0.5)
        parent.add_score(0.7)
        parent.add_score(0.8)

        child.add_score(0.7)
        child.add_score(0.8)
        a = math.sqrt((2*math.log(3)/2)*min(0.25,parent.get_variance() + math.sqrt(2*math.log(3)/2)))
        
        self.assertEqual(((1-1/3) * 0.75 + (1/3)*((1-1/4)*(1.8/3)+ (1/4) * (1.8/3)) + a),sf.get_score(parent, child, gs))
        