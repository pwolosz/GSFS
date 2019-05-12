class GlobalScores:
    """Class used for getting global scores like g_rave."""
    
    def __init__(self):
        self.scores = {'g_rave': {}}
    
    def update_g_rave_score(self, name, score):
        """
        Method for updataing g_rave score. For more information about g_rave score please see documentation.
        Parameters
        ----------
        name: str
            Name of the feature
        score: numeric
            Value of the score
        """
        if(name not in self.scores['g_rave']):
            self.scores['g_rave'][name] = {'n': 1, 'score': score}
        else:
            n = self.scores['g_rave'][name]['n']
            t_score = (self.scores['g_rave'][name]['score'] * n + score)/(n + 1)
            self.scores['g_rave'][name] = {'n': n + 1, 'score': t_score}