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
        lrave.add_path_score(set(['D']),1)
        
        self.assertEqual(lrave.get_path_score(set('A')), (0.1+0.4+0.6)/3)
        self.assertEqual(lrave.get_path_score(set('B')), (0.2+0.4+0.5+0.6)/4)
        self.assertEqual(lrave.get_path_score(set('C')), (0.3+0.5+0.6)/3)
        self.assertEqual(lrave.get_path_score(set(['A', 'B'])), (0.4+0.6)/2)
        self.assertEqual(lrave.get_path_score(set(['A', 'C'])), 0.6)
        self.assertEqual(lrave.get_path_score(set(['B', 'A', 'C'])), 0.6)
        self.assertEqual(lrave.get_path_score(set(['C', 'B'])), (0.5+0.6)/2)
        self.assertEqual(lrave.get_path_score(set(['D'])), 1)
        
        self.assertEqual(lrave.get_t_l(set('A')), 3)
        self.assertEqual(lrave.get_t_l(set('B')), 4)
        self.assertEqual(lrave.get_t_l(set('C')), 3)
        self.assertEqual(lrave.get_t_l(set(['C', 'B'])), 2)
        self.assertEqual(lrave.get_t_l(set(['C', 'A'])), 1)
        self.assertEqual(lrave.get_t_l(set(['B', 'C'])), 2)
        self.assertEqual(lrave.get_t_l(set(['A', 'B','C'])), 1)
        self.assertEqual(lrave.get_t_l(set(['D'])), 1)