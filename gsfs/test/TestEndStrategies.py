import unittest

class TestEndStrategies(unittest.TestCase):
    def setUp(self):
        self._end_strategies = EndStrategies('default', 5)
        
    def test_new_node(self):
        node = Node(set('B'),'A')
        self.assertEqual(self._end_strategies.are_calculations_over(node), True)
        
    def test_full_node(self):
        node = Node(set(['A','B','C','D']),'E')
        node.T = 1
        self.assertEqual(self._end_strategies.are_calculations_over(node), True)
        
    def test_middle_node(self):
        node = Node(set(['A','B']),'E')
        node.T = 1
        self.assertEqual(self._end_strategies.are_calculations_over(node), False)