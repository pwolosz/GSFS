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
    
    def get_path_score(self, name, used_features):
        """
        Method for getting score for selected path and feature
        Parameters
        ----------
        name: str
            Name of the feature
        used_features: set
            Set of features in selected path
        """
        
        tmp_features = used_features.copy()
        tmp_features.add(name)
        
        indexes = list(filter(lambda i: tmp_features.issubset(self._paths[i]), range(len(self._paths))))
        n = sum([self._n_vals[i] for i in indexes])
        score = sum([self._scores[i] for i in indexes])
        print(indexes)
        return score/n