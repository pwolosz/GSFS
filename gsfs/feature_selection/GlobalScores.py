from gsfs.feature_selection.LRavePaths import *
import pandas as pd

class GlobalScores:
    """Class containing methods for getting and updating l-RAVE and g-RAVE."""
    
    def __init__(self):
        self.scores = {'g_rave': {},
                       'l_rave': LRavePaths()}            
    
    def update_score(self, used_features, score):
        """
        Method for updating g-RAVE and l-RAVE for all features inused_features.

        -----------
        Parameters:
        used_features: set
            Set of features for which the scores will be added,
        score: float
            Value of the score that will be added.

        Returns: None
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
        Method for getting l-RAVE score for selected features.

        Parameters
        ----------
        used_features: set
            Set of features for which the score will be returned, 
            the score will be average score of all nodes containingused_featuresas subset oftheir features, 
            if selected features haven’t been visited 0 is returned.

        Returns: float
            l-RAVE score for selected features.
        """
        
        return self.scores['l_rave'].get_path_score(used_features)
        
    def get_g_rave_score(self, name):
        """
        method for getting g-RAVE score for selected feature, if feature hasn’t been visited 0 is returned.

        Parameters
        ----------
        name: str
            Name of the feature for which the score will be returned, the score will be an average score from all paths containing selected feature.

        Returns: float
            g-RAVE score for selected feature.
        """
        
        if name not in self.scores['g_rave']:
            return 0
        
        score_info = self.scores['g_rave'][name]
        return score_info['score']/score_info['n']
    
    def get_n(self, name):
        """
        Method for getting number of times feature is used in a node (is in set offeatures in node), if feature hasn’t been selected 0 is returned.

        ----------
        Parameters
        name: str
            Name of the feature for which the number of times it was usedwill be returned.

        Returns: int
            How many times the feature was used.
        """
        if name not in self.scores['g_rave']:
            return 0
        
        return self.scores['g_rave'][name]['n']
    
    def get_t_l(self, used_features):
        """
        Method for getting number of iterations in calculating l-RAVE score, 
        it’s a number of nodes which scores will be taken in calculating l-RAVE score for selected features.

        Parameters
        ----------
        used_features: set
            Features for which the score will be calculated.

        Returns: int
            Number of iterations in l-RAVE calculation.
        """
        
        return self.scores['l_rave'].get_t_l(used_features)
    
    def get_l_rave_dataframe(self):
        """
        Method for getting all l-RAVE scores in form of DataFrame,with columns features, n, scores and score.

        Returns: pandas.DataFrame
            Data frame with all l-RAVE scores.
        """

        return self.scores['l_rave'].get_scores_dataframe()
        
    def get_g_rave_dataframe(self):
        """
        Method for getting all g-RAVE scores in form of DataFrame, with columns feature, n, scores and score.

        Returns: pandas.DataFrame
            DataFrame with all g-RAVE scores.
        """

        names = []
        n = []
        scores = []
        for k,v in self.scores['g_rave'].items():
            names.append(k)
            n.append(v['n'])
            scores.append(v['score'])
            
        return pd.DataFrame({
            'feature': names,
            'n': n,
            'scores': scores,
            'score':[x/y for x, y in zip(scores, n)]
        })