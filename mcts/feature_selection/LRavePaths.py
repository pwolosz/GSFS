from mcts.feature_selection.LRavePaths import *

class GlobalScores:
    """Class used for getting global scores like g-RAVE and l-RAVE."""
    
    def __init__(self):
        self.scores = {'g_rave': {},
                       'l_rave': LRavePaths()}            
    
    def update_score(self, used_features, score):
        """
        Method for updating scores for selected path
        Parameters:
        used_features: set
            Features used in current path
        score: numeric
            Value of the score
        """
        self._update_l_rave_score(used_features, score)
        self._update_g_rave_score(used_features, score)
    
    def _update_l_rave_score(self, used_features, score):       
        self.scores['l_rave'].add_path_score(used_features, score)
        
    def _update_g_rave_score(self, used_features, score):
        for name in used_features:
            if name not in self.scores['g_rave']:
                self.scores['g_rave'][name] = {'n': 1, 'score': score}
            else:
                self.scores['g_rave'][name]['score'] += score
                self.scores['g_rave'][name]['n'] += 1
    
    def get_l_rave_score(self, used_features):
        """
        Method for getting l-RAVE score for selected feature and already used features
        Parameters
        ----------
        used_features: set
            Set of features for which the score will be returned
        """
        
        return self.scores['l_rave'].get_path_score(used_features)
        
    def get_g_rave_score(self, name):
        """
        Method for getting g-RAVE score for selected feature
        Parameters
        ----------
        name: str
            Name of feature
        """
        
        if name not in self.scores['g_rave']:
            return 0
        
        score_info = self.scores['g_rave'][name]
        return score_info['score']/score_info['n']
    
    def get_t_l(self, name, used_features):
        """
        Method for getting t_l (number of iterations in computing l-RAVE)
        Parameters
        ----------
        name: str
            Name of the feature
        used_features: set
            Set of already used features
        """
        
        return self.scores['l_rave'].get_t_l(name, used_features)