import numpy as np
import unittest

class TestNode(unittest.TestCase):
    def setUp(self):
        self._node = Node(set(), 'A')
        self._init_score = 0.5
        self._node._scores.append(self._init_score)
        self._node._scores_sum = self._init_score
        self._node.T = 1
        
    def test_adding_score(self):        
        added_score = 0.8
        self._node.add_score(added_score)
        self.assertEqual(self._node.get_score(), (added_score + self._init_score)/2)
        self.assertEqual(self._node.get_variance(), np.var([added_score, self._init_score]))
        
        
    def test_var_no_scores(self):
        node = Node(set(), 'A')
        self.assertEqual(node.get_variance(), 0)
        
    def test_score_no_scores(self):
        node = Node(set(), 'A')
        self.assertEqual(node.get_score(), 0)