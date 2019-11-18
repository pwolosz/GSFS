import pandas as pd

class LRavePaths:
    """Class for providing l-RAVE scores for algorithm."""

    def __init__(self):
        self._paths = []
        self._n_vals = []
        self._scores= []
        
    def add_path_score(self, used_features, score):
        """
        Method for adding l-RAVE score for selected features.

        Parameters
        ----------
        used_features: set
            Features for which the score will be added,
        score: float
            Added score.
        """
        
        tmp_features = used_features.copy()
        
        if tmp_features not in self._paths:
            self._paths.append(tmp_features)
            self._n_vals.append(1)
            self._scores.append(score)
        else:
            ind = self._paths.index(tmp_features)
            self._n_vals[ind] += 1
            self._scores[ind] += score
    
    def get_path_score(self, used_features):
        """
        Method for getting l-RAVE score for selected features, it will be an average score of all nodes that have used_featuresas as subset of their features.

        Parameters
        ----------
        used_features: set
            Features for which the score will be calculated.

        Returns: float
            l-RAVE score for selected features.
        """
        
        if used_features is None:
            raise Exception('used_features cannot be None')
        
        indexes = list(filter(lambda i: used_features.issubset(self._paths[i]), range(len(self._paths))))
        
        # if indexes is an empty list then that means that selected path has never been visited
        if len(indexes) == 0:
            return 0
        
        n = sum([self._n_vals[i] for i in indexes])
        score = sum([self._scores[i] for i in indexes])
        return score/n

    def get_t_l(self, used_features):
        """
        Method for getting number of iterations in calculating l-RAVE score, it’s a number of nodes which scores 
        will be taken in calculating l-RAVE score for selected features.

        Parameters
        ----------
        used_features: set
            Features for which the t_l will be calculated.

        Returns: int
            Number of iterations in l-RAVE calculation.
        """
        
        if used_features is None:
            raise Exception('used_features cannot be None')
        
        indexes = list(filter(lambda i: used_features.issubset(self._paths[i]), range(len(self._paths))))
        
        # if indexes is an empty list then that means that selected path has never been visited
        if len(indexes) == 0:
            return 0
        
        return sum([self._n_vals[i] for i in indexes])
    
    def get_scores_dataframe(self):
        """
        Method for getting all l-RAVE scores that were added.

        Returns: pandas.DataFrame
            Data frame in which every row contains fields features, n, scores, score.
        """
        return pd.DataFrame({
            'features': [','.join(s) for s in self._paths],
            'n': self._n_vals, 
            'scores': self._scores,
            'score': [x/y for x, y in zip(self._scores, self._n_vals)]
        })