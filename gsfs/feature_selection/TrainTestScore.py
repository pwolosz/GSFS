from sklearn.model_selection import train_test_split

class TrainTestScore:
    
    @staticmethod
    def train_test_score(metric, metric_name, model, data, labels, test_size):
        
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)
        model.fit(X_train, y_train)
        
        if metric_name in ['acc','f1']:
            predicted = model.predict(X_test)   
        else:
            predicted = model.predict_proba(X_test)[:,1]
                
        return metric(y_test, predicted)