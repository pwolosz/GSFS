import unittest

class TestLRavePaths(unittest.TestCase):
    def test_path_score(self):
        lrave = LRavePaths()
        lrave.add_path_score(set('A'),0.1)
        lrave.add_path_score(set('B'),0.2)
        lrave.add_path_score(set('C'),0.3)
        lrave.add_path_score(set(['B','A']),0.4)
        lrave.add_path_score(set(['C','B']),0.5)
        lrave.add_path_score(set(['C','A','B']),0.6)
        
        self.assertEqual(lrave.get_path_score('A', set()), (0.1+0.4+0.6)/3)
        self.assertEqual(lrave.get_path_score('B', set()), (0.2+0.4+0.5+0.6)/4)
        self.assertEqual(lrave.get_path_score('C', set()), (0.3+0.5+0.6)/3)
        self.assertEqual(lrave.get_path_score('B', set(['A'])), (0.4+0.6)/2)
        self.assertEqual(lrave.get_path_score('A', set(['C'])), 0.6)
        self.assertEqual(lrave.get_path_score('A', set(['B','C'])), 0.6)
        self.assertEqual(lrave.get_path_score('B', set(['C'])), (0.5+0.6)/2)
        
        self.assertEqual(lrave.get_t_l('A', set()), 3)
        self.assertEqual(lrave.get_t_l('B', set()), 4)
        self.assertEqual(lrave.get_t_l('C', set()), 3)
        self.assertEqual(lrave.get_t_l('C', set(['B'])), 2)
        self.assertEqual(lrave.get_t_l('A', set(['C'])), 1)
        self.assertEqual(lrave.get_t_l('B', set(['C'])), 2)
        self.assertEqual(lrave.get_t_l('A', set(['B','C'])), 1)