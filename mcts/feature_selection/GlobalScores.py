from mcts.feature_selection.LRavePaths import *

class GlobalScores:
    """Class used for getting global scores like g-RAVE and l-RAVE."""
    
    def __init__(self):
        self.scores = {'g_rave': {},
                       'l_rave': LRavePaths()}            
    
    def update_l_rave_score(self, score, used_features):
        """
        Method for updating l-RAVE score for selected set of features
        Parameters
        ----------
        score: numeric
            Value of the score
        used_features: set
            Set of features for which the scores will be updated
        """
        
        self.scores['l_rave'].add_path_score(used_features, score)
        
    def update_g_rave_score(self, name, score):
        """
        Method for updating g-RAVE score for selected feature.
        Parameters
        ----------
        name: str
            Name of the feature to update score
        score: numeric
            Value of the score
        """
        
        if name not in self.scores['g_rave']:
            self.scores['g_rave'][name] = {'n': 1, 'score': score}
        else:
            self.scores['g_rave'][name]['score'] += score
            self.scores['g_rave'][name]['n'] += 1
    
    def get_l_rave_score(self, name, used_features):
        """
        Method for getting l-RAVE score for selected feature and already used features
        Parameters
        ----------
        name: str
            Name of the feature
        used_features: set
            Set of already used features
        """
        
        return self.scores['l_rave'].get_path_score(name, used_features)
        
    def get_g_rave_score(self, name):
        """
        Method for getting g-RAVE score for selected feature
        Parameters
        ----------
        name: str
            Name of feature
        """
        
        score_info = self.scores['g_rave'][name]
        return score_info['score']/score_info['n']