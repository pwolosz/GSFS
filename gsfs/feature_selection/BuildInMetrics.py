from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class BuildInMetrics:
    """Class containing supported scoring methods, supports accuracy, f1 and Roc AUC."""
    def __init__(self):
        """Initializes the metrics dictionary"""
        
        self._metrics = {
            'acc': accuracy_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score
        }
        
    def get_metric(self, name):
        """
        method for getting metric with selected name, supported values areacc(accuracy), f1, roc_auc, if the provided name is not supported then exception is thrown.
        Parameters
        ----------
        name: str
            Name of the metric to be returned.

        Returns: sklearn metric
            One of the supported metrics.
        """

        if name in self._metrics:
            return self._metrics[name]
        else:
            raise Exception('Error initializing GSFS object, \"' + name + '\" is not supported metric, available values are: ' + ', '.join(self._metrics.keys()))