class LRavePaths:
    def __init__(self):
        self._paths = []
        self._n_vals = []
        self._scores= []
        
    def add_path_score(self, used_features, score):
        """
        Method for adding score for path made of selected nodes
        Parameters
        ----------
        used_features: set
            Set of features in selected path
        score: numeric
            Score that was obtained with nodes from selected path
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
        Method for getting score for selected path and feature
        Parameters
        ----------
        used_features: set
            Set of features in selected path
        """
        
        indexes = list(filter(lambda i: used_features.issubset(self._paths[i]), range(len(self._paths))))
        
        # if indexes is an empty list then that means that selected path has never been visited
        if len(indexes) == 0:
            return 0
        
        n = sum([self._n_vals[i] for i in indexes])
        score = sum([self._scores[i] for i in indexes])
        return score/n

    def get_t_l(self, used_features):
        """
        Method for getting t_l (number of iterations in computing l-RAVE)
        Parameters
        ----------
        used_features: set
            Set of features in selected path
        """
        
        indexes = list(filter(lambda i: used_features.issubset(self._paths[i]), range(len(self._paths))))
        
        # if indexes is an empty list then that means that selected path has never been visited
        if len(indexes) == 0:
            return 0
        
        return sum([self._n_vals[i] for i in indexes])