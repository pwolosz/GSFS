from sklearn.model_selection import StratifiedKFold

class CV:
    """Class containing static method for performing cross-validation"""
    
    @staticmethod
    def cv(metric, metric_name, model, data, labels, cv):
        """
        Static method that performs cross-validation for selected dataset and model.   
        It uses StratifiedKFold fromsklearn.model_selection to make the "cv" number of splits, 
        every split will have same ratio of rows with positive class to rows with negative class 
        as the input dataset.
        
        Parameters
        ----------
        metric: sklearn metric from BuildInMetrics
            One of the supported metrics (supported metrics are in BuildInMetrics module),
        metric_name: str
            Name of used metric,
        model: sklearn model
            Model for which the cross-validation score will be calculated,
        data: pandas.DataFrame
            Input dataset used in cross-validation,
        labels: pandas.Series
            Labels of input dataset,
        cv: int
            Number of folds in cross-validation.
            
        Returns: float
            Cross-validation score for selected metric.
        """

        kfold = StratifiedKFold(n_splits=cv, random_state=123, shuffle=True)
        score = 0
        
        if metric_name in ['acc','f1']:
            for train, test in kfold.split(data, labels):
                model.fit(data.loc[train,:], labels[train])
                predicted = model.predict(data.loc[test,:])
                score += metric(labels[test], predicted)
        else:
            for train, test in kfold.split(data, labels):
                model.fit(data.loc[train,:], labels[train])
                predicted = model.predict_proba(data.loc[test,:])[:,1]
                score += metric(labels[test], predicted)
                
        return score/cv