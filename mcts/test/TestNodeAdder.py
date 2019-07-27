class TestNodeAdder(unittest.TestCase):
    def setUp(self):
        self._root = Node(set(),None)
        self._node_adder = NodeAdder(self._root)
        
    def test_adding(self):
        all_features = ['A','B','C','D']
        
        for feature in all_features:
            self._node_adder.add_node(self._root, feature)
        
        for i in range(len(all_features)):
            self.assertEqual(self._root._children[i]._features, set(all_features[i]))
        
    def test_adding_to_prev(self):
        all_features = ['A','B','C','D']
        for feature in all_features:
            self._node_adder.add_node(self._root, feature)
            
        self._node_adder.add_node(self._root._children[0], 'B')
        
        for i in range(len(all_features)):
            self.assertEqual(self._root._children[i]._features, set(all_features[i]))
            
        self.assertEqual(len(self._root._children[0]._children), 1)
        self.assertEqual(len(self._root._children[1]._children), 1)
        self.assertEqual(len(self._root._children[2]._children), 0)
        self.assertEqual(len(self._root._children[3]._children), 0)
        
        self.assertEqual(self._root._children[0]._children[0]._features, set(['A','B']))
        self.assertEqual(self._root._children[1]._children[0]._features, set(['A','B']))
        
    def test_adding_to_next(self):
        self._node_adder.add_node(self._root, 'A')
        self._node_adder.add_node(self._root, 'C')
        self._node_adder.add_node(self._root, 'D')
        
        self._node_adder.add_node(self._root._children[0], 'B')
        self._node_adder.add_node(self._root, 'B')
        
        self.assertEqual(len(self._root._children[0]._children), 1)
        self.assertEqual(len(self._root._children[1]._children), 0)
        self.assertEqual(len(self._root._children[2]._children), 0)
        self.assertEqual(len(self._root._children[3]._children), 1)
        
        self.assertEqual(self._root._children[0]._children[0]._features, set(['A','B']))
        self.assertEqual(self._root._children[3]._children[0]._features, set(['A','B']))