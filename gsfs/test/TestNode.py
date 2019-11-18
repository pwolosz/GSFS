import unittest

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
        added_score_1 = 0.8
        added_score_2 = 0.4
        self._node.add_score(added_score_1)
        self._node.add_score(added_score_2)
        
        self.assertEqual(self._node.get_score(), (added_score_1 + added_score_2 + self._init_score)/3)
        self.assertEqual(self._node.get_variance(), np.var([added_score_1, added_score_2, self._init_score]))
        self.assertEqual(self._node.T,3)
        
    def test_var_no_scores(self):
        node = Node(set(), 'A')
        self.assertEqual(node.get_variance(), 0)
        
    def test_score_no_scores(self):
        node = Node(set(), 'A')
        self.assertEqual(node.get_score(), 0)
        
    def test_used_features(self):
        node = Node(set(), 'A')
        node.add_child(Node(set('A'),'B'))
        node.add_child(Node(set('A'),'C'))
        node.add_child(Node(set('A'),'D'))
        
        used_nodes = node.get_used_features_in_children()
        self.assertEqual(len(used_nodes), 4)
        self.assertEqual(used_nodes == set(['A','B','C','D']), True)
        
    def test_adding_nodes(self):
        node = Node(set(), 'A')
        node.add_child(Node(set('A'),'C'))
        node.add_child(Node(set('A'),'B'))
        node.add_child(Node(set('A'),'D'))
        
        self.assertEqual(node._children[0]._features == set(['A','C']), True)
        self.assertEqual(node._children[1]._features == set(['A','B']), True)
        self.assertEqual(node._children[2]._features == set(['A','D']), True)