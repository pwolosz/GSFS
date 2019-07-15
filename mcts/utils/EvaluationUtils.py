import os  
import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mcts.feature_selection.DrawTree import draw_tree
import pandas as pd
import numpy as np
import openml

class EvaluationUtils:
    """Class containing methods for evaluation of MCTS model"""
    
    @staticmethod
    def eval(model, files_info, data_path, out_path, out_file_name, eval_time, metric_name, cv = 4, mcts_cv = 5, params = None):
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
        mcts_cv: int
            Number of folds in cross-validation (in mcts algorithm)
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
    def openml_eval(model, tasks_info, out_path, out_file_name, 
                    metric_name, cv, mcts_cv, 
                    calculations_done_condition, calculations_budget, scoring_function,
                    multiarm_strategy, end_strategy, models = None, mcts_params = None, name = None):
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
        mcts_cv: int
            Number of folds in cross-validation in mcts 
        params: dict (default: None)
            Dictionary with parameters of MCTS, if None then default ones from DefaultSettings will be used 
        
        Returns: pandas.DataFrame with results, models without MCTS will be named 'model_<metric name>',
                 the ones with MCTS will be named 'MCTS_<metric_name>'.
        """
        
        scores = pd.DataFrame(columns=['name', 'model_roc_auc_cv', 'model_acc_cv',
            'model_acc_cv','MCTS_roc_auc','MCTS_acc','MCTS_f1','MCTS_f1_var',
            'MCTS_acc_var','MCTS_roc_var','n_iter','MCTS_best_score'])
        
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
                scores = scores.append(EvaluationUtils.eval_mcts(model, data, labels, name, out_path, metric_name, cv, mcts_cv, 
                  calculations_done_condition, calculations_budget, scoring_function,
                  multiarm_strategy, end_strategy, models, mcts_params), 
                                       ignore_index = True)
     
        scores.to_csv(out_path + '/' + out_file_name + '.csv')
        
        return scores
    
    @staticmethod
    def eval_mcts(model, data, labels, name, out_path, metric_name, cv, mcts_cv, 
                  calculations_done_condition, calculations_budget, scoring_function,
                  multiarm_strategy, end_strategy, models = None, mcts_params = None):

        kfold = StratifiedKFold(n_splits=cv, random_state=None, shuffle=True)
        f1_scores = []
        acc_scores = []
        auc_scores = []
        longest_paths = []
        ids = []
        
        out_path += '/' + name + str(datetime.datetime.now().timestamp())
        if not os.path.exists(out_path):
            try:  
                os.mkdir(out_path)
            except OSError:  
                print ("Creation of the directory %s failed" % path)
        
        
        for train, test in kfold.split(data, labels):
            mcts = MCTS(model, calculations_done_condition = calculations_done_condition, 
                    calculations_budget = calculations_budget,
                    metric = metric_name, params = mcts_params, end_strategy = end_strategy,
                    multiarm_strategy = multiarm_strategy, scoring_function = scoring_function)
            
            mcts_id = id(mcts)
            
            ids.append(mcts_id)
        
            mcts.fit(data.loc[train,:], labels[train], preprocess = False)
            predicted = mcts.predict(data.loc[test,:])
            predicted_proba = mcts.predict_proba(data.loc[test,:])[:,1]

            f1_scores.append(f1_score(labels[test], predicted))
            auc_scores.append(roc_auc_score(labels[test], predicted_proba))
            acc_scores.append(accuracy_score(predicted, labels[test]))
            longest_paths.append(mcts._longest_tree_branch)
            
            mcts.save_stats_to_file(out_path + '/' + str(mcts_id) + '.txt')
            pd.DataFrame({'id': ids, 'roc': auc_scores, 'acc': acc_scores,
                         'f1': f1_scores, 'longest_paths': longest_paths})
            draw_tree(mcts._root, view = False, view_nodes_info = False, file_name = out_path + '/' + str(mcts_id))
            if models is not None:
                model_scores = {}
                for key, value in models.items():
                    if key not in model_scores:
                        model_scores[key] = {'roc': [], 'acc': [], 'f1': []}
                    value.fit(data.loc[test,:])
                    predicted = mcts.predict(data.loc[test,:])
                    predicted_proba = mcts.predict_proba(data.loc[test,:])[:,1]
                    model_scores[key]['roc'].append(roc_auc_score(labels[test], predicted_proba))
                    model_scores[key]['f1'].append(f1_score(labels[test], predicted))
                    model_scores[key]['acc'].append(accuracy_score(predicted, labels[test]))

                model_scores.to_csv(out_path + '/' + 'part_scores.csv')
        
        return pd.Series({
            'name': name,
            'model_roc_auc_cv': CV.cv(roc_auc_score, 'roc_auc', model, data, labels, cv),
            'model_acc_cv': CV.cv(accuracy_score, 'acc', model, data, labels, cv),
            'model_acc_cv': CV.cv(f1_score, 'f1', model, data, labels, cv),
            'MCTS_roc_auc': np.mean(auc_scores),
            'MCTS_acc': np.mean(acc_scores),
            'MCTS_f1': np.mean(f1_scores),
            'MCTS_f1_var': np.var(acc_scores),
            'MCTS_acc_var': np.var(acc_scores),
            'MCTS_roc_var': np.var(acc_scores),
            'n_iter': mcts.get_number_of_iterations(),
            'MCTS_best_score': mcts._best_score
        })