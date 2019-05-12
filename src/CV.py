class CV:
    """Class containing method for cross-validation"""
    
    @staticmethod
    def cv(metric, metric_name, model, data, labels, cv):
        """
        Method for performing stratified cross-validation (sklearn.model_selection.StratifiedKFold)
        
        Parameters
        ----------
        metric: method metric
            Metric that will be calculated, it can be one of BuildInMetrics or own one, see documentation for more info
        metric_name: str
            Name of the metric that will be used
        model: scikit-learn model
            Model that the performance will be calculated for
        data: pd.DataFrame
            Data that will be used for cross-validation
        labels: pandas.Series
            Labels that will be used for cross-validation
        cv: int
            Number of folds
            
        Returns: float, cross-validation score
        """
        
        kfold = StratifiedKFold(n_splits=params['cv'], random_state=None, shuffle=True)
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
                
        return score/params['cv']