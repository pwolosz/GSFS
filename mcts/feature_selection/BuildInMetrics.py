from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class BuildInMetrics:
    """Class used for getting classic metrics (like accuracy, f1, etc.) to evaluate model's performance"""
    def __init__(self):
        """Initializes the metrics dictionary"""
        
        self._metrics = {
            'acc': accuracy_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score
        }
        
    def get_metric(self, name):
        """
        Method for getting metric with selected name. If the proper name won't be provided the exception will be thrown
        Parameters
        ----------
        name: str
            name of the metric
        """

        if name in self._metrics:
            return self._metrics[name]
        else:
            raise Exception('Error initializing MCTS object, \"' + name + '\" is not supported metric, available values are: ' + ', '.join(self._metrics.keys()))