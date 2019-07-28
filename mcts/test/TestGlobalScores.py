import unittest

class TestGlobalScores(unittest.TestCase):
    def test_scores(self):
        global_scores = GlobalScores()
        global_scores.update_score(set('A'),0.1)
        global_scores.update_score(set('B'),0.2)
        global_scores.update_score(set('C'),0.3)
        global_scores.update_score(set(['C','A']),0.4)
        global_scores.update_score(set(['C','B']),0.5)
        global_scores.update_score(set(['B','A']),0.6)
        global_scores.update_score(set(['B','A']),0.7)
        global_scores.update_score(set(['C','A']),0.8)
        global_scores.update_score(set(['C','A','B']),0.9)
        
        self.assertEqual(global_scores.get_g_rave_score('A'),(0.1+0.4+0.6+0.7+0.8+0.9)/6)
        self.assertEqual(global_scores.get_g_rave_score('B'),(0.2+0.5+0.6+0.7+0.9)/5)
        self.assertEqual(global_scores.get_g_rave_score('C'),(0.3+0.4+0.5+0.8+0.9)/5)  
        
        self.assertEqual(global_scores.get_l_rave_score(set('A')), (0.1+0.4+0.6+0.7+0.8+0.9)/6)
        self.assertEqual(global_scores.get_l_rave_score(set('B')), (0.2+0.5+0.6+0.7+0.9)/5)
        self.assertEqual(global_scores.get_l_rave_score(set('C')), (0.3+0.4+0.5+0.8+0.9)/5)
        self.assertEqual(global_scores.get_l_rave_score(set(['A','B'])), (0.6+0.7+0.9)/3)
        self.assertEqual(global_scores.get_l_rave_score(set(['B','A'])), (0.6+0.7+0.9)/3)
        self.assertEqual(global_scores.get_l_rave_score(set(['B','C'])), (0.5+0.9)/2)
        self.assertEqual(global_scores.get_l_rave_score(set(['A','C','B'])), 0.9)
        
        self.assertEqual(global_scores.get_t_l(set('A')), 6)
        self.assertEqual(global_scores.get_t_l(set('B')), 5)
        self.assertEqual(global_scores.get_t_l(set('C')), 5)
        self.assertEqual(global_scores.get_t_l(set(['A','B'])), 3)
        self.assertEqual(global_scores.get_t_l(set(['B','A'])), 3)
        self.assertEqual(global_scores.get_t_l(set(['B','C'])), 2)
        self.assertEqual(global_scores.get_t_l(set(['A','C','B'])), 1)