class EvaluationUtils:
    """Class containing methods for evaluation of MCTS model"""
    
    @staticmethod
    def eval(model, files_info, data_path, out_path, out_file_name, eval_time, metric_name, cv = 4, params = None):
        """
        Method for calculating scores for selected model and MCTS used for this model. 
        Calculated metrics will be accuracy, AUC.
        
        Parameters
        ----------
        model: scikit-learn model, with fit, predict and predict_proba methods
            Model used to MCTS
        files_info: dict
            Dictionary containing info about files to use with evaluation, every entry must have keys:
                - name - name of the file
                - target_class - name of the column with the predicted variable
                - pos_class - value of positive class, if 'numeric' then max value will be taken as positive class indicator
        data_path: str
            Path to files that will be used
        out_path: str
            Path where results will be saved
        out_file_name: str
            Name of the file that will saved as results
        eval_time: numeric
            Time in seconds that every MCTS use will last
        metric_name: str
            Name of the metric that will be used as metric in MCTS
        cv: int
            Number of folds in cross-validation
        params: dict (default: None)
            Dictionary with parameters of MCTS, if None then default ones from DefaultSettings will be used 
        
        Returns: pandas.DataFrame with results, models without MCTS will be named 'model_<metric name>',
                 the ones with MCTS will be named 'MCTS_<metric_name>'.
        """
        
        scores = pd.DataFrame(columns=['name','model_roc_auc_cv','model_acc_cv',
                                       'model_roc_test','model_acc_test',
                                       'MCTS_roc_auc','MCTS_acc','n_iter'])
        
        for file_info in files_info:
            print('Using ' + file_info['name'])
            
            mcts = MCTS(model, calculactions_done_conditions = {'type': 'time', 'max_val': eval_time}, metric=metric_name)
            data = pd.read_csv(data_path + file_info['name'])
            labels = mcts._preprocess_labels(data.loc[:,file_info['target_class']], file_info['pos_class'])
            data = data.drop(columns = [file_info['target_class']])
            data = mcts.one_hot_encode(data).fillna(0)
            
            scores = scores.append(EvaluationUtils.eval_mcts(model, data, labels, file_info['name'], 
                                                                 eval_time, metric_name, cv = 4, params = None), 
                                       ignore_index = True)
            
        scores.to_csv(out_path + out_file_name + '.csv')
        
        return scores
    
    @staticmethod
    def openml_eval(model, tasks_info, out_path, out_file_name, eval_time, metric_name, cv = 4, params = None):
        """
        Method for calculating scores using data from openml.org for selected model and MCTS used for this model. 
        Calculated metrics will be accuracy, AUC.
        
        Parameters
        ----------
        model: scikit-learn model, with fit, predict and predict_proba methods
            Model used to MCTS
        tasks_info: pandas.DataFrame
            DataFrame containing info about tasks to use with evaluation, every entry must have keys:
                - name - name of the file
                - task_id - id of the task from openml.org
        out_path: str
            Path where results will be saved
        out_file_name: str
            Name of the file that will saved as results
        eval_time: numeric
            Time in seconds that every MCTS use will last
        metric_name: str
            Name of the metric that will be used as metric in MCTS
        cv: int
            Number of folds in cross-validation
        params: dict (default: None)
            Dictionary with parameters of MCTS, if None then default ones from DefaultSettings will be used 
        
        Returns: pandas.DataFrame with results, models without MCTS will be named 'model_<metric name>',
                 the ones with MCTS will be named 'MCTS_<metric_name>'.
        """
        
        scores = pd.DataFrame(columns=['name','model_roc_auc_cv','model_acc_cv',
                                       'model_roc_test','model_acc_test',
                                       'MCTS_roc_auc','MCTS_acc','n_iter'])
        
        for index, task in tasks_info.iterrows():
            print('Using ' + task['name'] + ' (' + str(index + 1) + '/' + str(tasks_info.shape[0]) + ')')
            is_data_loaded = False
            
            try:
                openml_task = openml.tasks.get_task(task['task_id'])
                data, labels = openml_task.get_X_and_y()
                data = pd.DataFrame(data).fillna(0)
                labels = pd.Series(labels)
                is_data_loaded = True
            except:
                print(task['name'] + ' - error')
            
            if is_data_loaded:
                scores = scores.append(EvaluationUtils.eval_mcts(model, data, labels, task['name'], 
                                                                 eval_time, metric_name, cv = 4, params = None), 
                                       ignore_index = True)

            
                
        scores.to_csv(out_path + out_file_name + '.csv')
        
        return scores
    
    @staticmethod
    def eval_mcts(model, data, labels, name, eval_time, metric_name, cv = 4, params = None):
        
        mcts = MCTS(model, calculactions_done_conditions = {'type': 'time', 'max_val': eval_time}, metric=metric_name)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.25, random_state = 123)
        mcts.fit(X_train, y_train, preprocess = False)
        predicted = mcts.predict(X_test)
        predicted_proba = mcts.predict_proba(X_test)[:,1]
        model.fit(X_train, y_train)
        model_roc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        model_acc_test = accuracy_score(model.predict(X_test), y_test)
        
        return pd.Series({
            'name': name,
            'model_roc_auc_cv': CV.cv(roc_auc_score, 'roc_auc', model, data, labels, cv),
            'model_acc_cv': CV.cv(accuracy_score, 'acc', model, data, labels, cv),
            'MCTS_roc_auc': roc_auc_score(y_test, predicted_proba),
            'MCTS_acc': accuracy_score(y_test, predicted),
            'n_iter': mcts.get_number_of_iterations(),
            'MCTS_best_score': mcts._best_score,
            'model_roc_test': model_roc_test,
            'model_acc_test': model_acc_test
        })