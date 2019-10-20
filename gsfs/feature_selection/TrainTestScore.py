from sklearn.model_selection import train_test_split

class TrainTestScore:
    """Class containing static method for performing scoring of a model using train-test split."""
    
    @staticmethod
    def train_test_score(metric, metric_name, model, data, labels, test_size):
        """
        Method for scoring a model using train-test split.
        
        Parameters
        ----------
        metric: method metric
            Metric that will be calculated, one of BuildInMetrics, see documentation for more info
        metric_name: str
            Name of the metric that will be used
        model: scikit-learn model
            Model that the performance will be calculated for
        data: pd.DataFrame
            Data that will be used for train-test split
        labels: pandas.Series
            Labels that will be used for train-test split
        test_size: float
            Fraction of the input dataset that will be used as a test
            
        Returns: float, train-test split score
        """
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state = 123)
        model.fit(X_train, y_train)
        
        if metric_name in ['acc','f1']:
            predicted = model.predict(X_test)   
        else:
            predicted = model.predict_proba(X_test)[:,1]
                
        return metric(y_test, predicted)