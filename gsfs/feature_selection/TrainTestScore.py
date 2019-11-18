from sklearn.model_selection import train_test_split

class TrainTestScore:
    """Class containing static method for performing scoring of a model using train-test split."""
    
    @staticmethod
    def train_test_score(metric, metric_name, model, data, labels, test_size):
        """
        Method for scoring a model using train-test split.
        
        Parameters
        ----------
        metric: sklearn metric from BuildInMetrics
            One of the supported metrics (supported metrics are in BuildInMetrics module),
        metric_name: str
            Name of used metric,
        model: sklearn model
            Model for which the train-test score will be calculated,
        data: pandas.DataFrame
            Input dataset used in train-test split,
        labels: pandas.Series
            Labels of input dataset
        test_size: float
            Fraction of the input dataset that will be used as a test dataset.
            
        Returns: float 
            Train-test split score.
        """

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = test_size, random_state = 123)
        model.fit(X_train, y_train)
        
        if metric_name in ['acc','f1']:
            predicted = model.predict(X_test)   
        else:
            predicted = model.predict_proba(X_test)[:,1]
                
        return metric(y_test, predicted)