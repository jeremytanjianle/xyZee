import pandas as pd
import numpy as np
import os
import pickle
from sklearn.base import clone
import warnings
from sklearn.metrics import f1_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold
from sklearn.utils import check_array, check_X_y

# make scorer
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
mse_scorer = make_scorer(mean_squared_error)
acc_scorer = make_scorer(accuracy_score)

def error(mem):
    """
    Essentially, this error function calculates the deviation of the latest marginal contributions 
    from the past 100 rolling mean of marginal contributions

    Used as a stopping criteria
    """
    if len(mem) < 100:
        return 1.0
    all_vals = (np.cumsum(mem, 0)/np.reshape(np.arange(1, len(mem)+1), (-1,1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)


class DShap(object):
    
    def __init__(self, X, y, X_value, y_value, model, sources=None, 
                 scoring=mse_scorer, # whats the default in sklearn?
                 verbose=False,
                 seed=100):
        """
        Args:
            X: Data covariates
            y: Data labels
            X_value: Held-out covariates to run the value function on
            y_value: Held-out labels to run the value function on
            model: The sklearn compatible learning algorithm
            sources: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value.
            directory: Directory to save results and figures.
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations.
            **kwargs: Arguments of the model
        """
            
        np.random.seed(seed)
        
        self.scoring = scoring
        self.verbose = verbose
        self._initialize_instance(X, y, X_value, y_value, sources)
        
        is_regression = (np.mean(self.y//1==self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        self.model = model
        self.random_score = self.init_score(scoring)
            
    def _initialize_instance(self, X, y, X_value, y_value, sources=None):
        """Loads or creates data."""
        
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}

        
        # save the data to object attributes
        self.X_value, self.y_value =  X_value, y_value,
        self.X, self.y, self.sources = X, y, sources
        
        # save the mem_tmc
        self.mem_tmc = np.zeros((0, self.X.shape[0]))
        idxs_shape = (0, self.X.shape[0] if self.sources is None else len(self.sources.keys()))
        self.idxs_tmc, self.idxs_g = [np.zeros(idxs_shape).astype(int) for _ in range(2)]
                
            
            
    def init_score(self, scoring):
        """ 
        Give the value of an initial untrained model. 
            (1) randomize a train set 
            (2) score on test set
        
        """
        shuffled_train_set_idx = np.random.permutation(np.arange(len(self.X)))
        self.model.fit(self.X[shuffled_train_set_idx], self.y)
        init_score = scoring(self.model, self.X_value, self.y_value)
        return init_score
        
        # return a mean score of what could be randomly drawn
        return np.mean([scoring(self.y_value, np.random.permutation(self.y_value)) for _ in range(1000)]) 
        
        
        if metric == 'r2':
            return 0.5
        if metric == 'accuracy':
            return np.max(np.bincount(self.y_value).astype(float)/len(self.y_value))
        if metric == 'f1':
            return np.mean([f1_score(
                self.y_value, np.random.permutation(self.y_value)) for _ in range(1000)])
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            self.model.fit(self.X, np.random.permutation(self.y))
            random_scores.append(self.value(self.model, self.scoring))
        return np.mean(random_scores)
        
    def value(self, model, scoring=None, X=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data different from test set.
            y: Labels, if valuation is performed on a data different from test set.
            """
        if scoring is None:
            scoring = self.scoring
        if X is None:
            X = self. X_value
        if y is None:
            y = self.y_value
        return scoring(model, X, y)
        
        
        
        
        
    def run(self, save_every=100, err=0.1, tolerance=0.01):
        """Calculates data sources(points) values.
        
        Args:
            save_every: save marginal contrivbutions every n iterations.
            err: stopping criteria for each of TMC-Shapley or G-Shapley algorithm.
            tolerance: Truncation tolerance. If None, the instance computes its own.
        """
        tmc_run= True
        while tmc_run:
            if error(self.mem_tmc) < err:
                tmc_run = False
            else:
                self._tmc_shap(save_every, tolerance=tolerance, sources=self.sources)
                self.vals_tmc = np.mean(self.mem_tmc, 0)
        
        
    def _tmc_shap(self, iterations, tolerance=None, sources=None):
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance. (ratio with respect to average performance.)
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score
        except:
            self._tol_mean_score()        
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                if self.verbose: print('{} out of {} TMC_Shapley iterations.'.format(iteration + 1, iterations))
            marginals, idxs = self.one_iteration(tolerance=tolerance, sources=sources)

            self.latest_marginals = marginals
            self.latest_idxs = idxs
            
            self.mem_tmc = np.concatenate([self.mem_tmc, np.reshape(marginals, (1,-1))])
            self.idxs_tmc = np.concatenate([self.idxs_tmc, np.reshape(idxs, (1,-1))])
        
    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.restart_model()
        for _ in range(1):
            self.model.fit(self.X, self.y)
            for _ in range(100):
                bag_idxs = np.random.choice(len(self.y_value), len(self.y_value))
                scores.append(self.value(self.model, scoring=self.scoring,
                                         X=self.X_value[bag_idxs], y=self.y_value[bag_idxs]))
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)
        
    def one_iteration(self, tolerance, sources=None):
        """Runs one iteration of TMC-Shapley algorithm."""
        
        # enforce sources as a existing dictionary
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
            
        # iterate through a permuted index
        idxs, marginal_contribs = np.random.permutation(len(sources.keys())), np.zeros(len(self.X))
        new_score = self.random_score
        X_batch, y_batch = np.zeros((0,) +  tuple(self.X.shape[1:])), np.zeros(0).astype(int)
        truncation_counter = 0
        for n, idx in enumerate(idxs):
            old_score = new_score
            X_batch = np.concatenate([X_batch, self.X[sources[idx]]])
            y_batch = np.concatenate([y_batch, self.y[sources[idx]]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.is_regression or len(set(y_batch)) == len(set(self.y_value)):
                    self.restart_model()
                    self.model.fit(X_batch, y_batch)
                    new_score = self.value(self.model, scoring=self.scoring)       
            
            # save marginal contributions by sources
            # "This marginal contribution of the data source is one monte-carlo sample of its Data Shapley value"
            # The indexes are permuted but the marginal_contribs are ordered
            marginal_contribs[sources[idx]] = (new_score - old_score) / len(sources[idx])
            
            # if, for 5 consecutive times, the additional new score does not change the meanscore much
            # then truncate the iteration
            if np.abs(new_score - self.mean_score) <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs
    
    def restart_model(self):
        
        try:
            self.model = clone(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)


    def convergence_plots(self):
        plt.rcParams['figure.figsize'] = 15,15
        for i, idx in enumerate(np.arange(min(25, self.mem_tmc.shape[-1]))):
            plt.subplot(5,5,i+1)
            plt.plot(np.cumsum(self.mem_tmc[:, idx])/np.arange(1, len(self.mem_tmc)+1)) 






class dshap_transformer(TransformerMixin):
    """
    Uses shapley algorithm to value individual observations in an internal CV fashion. 
    Poorly valued data points are then removed in transform function.
    
    
    Attributes:
    ----------
    model : sklearn estimator object
        model used to estimate shapley values. Recommend cheap models
    scoring: sklearn compatible scoring function
        used to decide the value of the datapoint as well as decide 
        whether it is a classification or regression problem
    cv: int
        cross validation splits to calculate shapley values
    
    
    Example
    -------
    dshap_ = dshap_transformer(lreg, mse_scorer, cv=2, shap_iter=100, shap_tol=0.1)
    dshap_.fit(X_train.values, y_train.values)
    shapley_filtered_x = dshap_.transform(X_train.values)
    """
    def __init__(self, model, scoring, cv=3, shap_iter=500, shap_tol = 0.05, pct_deletion=0.8):
        self.model = model
        self.scoring = scoring
        self.cv = cv
        self.shap_iter = shap_iter
        self.shap_tol = shap_tol
        self.pct_deletion = pct_deletion
        
        
    def fit(self, X, y):
        """
        Parameters
        ----------
        X : np.array
            covariates to run the shapley algorithm on
        y : array-like
            labels / y to run the shapley algorithm on
        """
        # check array
        X, y = check_X_y(X,y)

        # store shapley values in array here
        self.vals_tmc = np.zeros(len(X))
        
        # get split indices
        kf = KFold(n_splits=self.cv)
        kf.get_n_splits(X)

        # split by split index
        for value_index, train_index in kf.split(X):
            X_train, X_value = X[train_index], X[value_index]
            y_train, y_value = y[train_index], y[value_index]
            # print(f"Proportion of train to value sets: {len(X_train)} vs {len(X_value)}")
            
            # run the Data shapley algorithm
            dshap = DShap(X_train, y_train, 
                          X_value, y_value, 

                          model=self.model, 
                          scoring=self.scoring,
                          verbose=False
                          )
            dshap.run(self.shap_iter, tolerance = self.shap_tol)
            
            # store the shapley values
            self.vals_tmc[train_index] = dshap.vals_tmc
            
        return self
            
    def transform(self, X, y, pct_deletion=None):
        """
        filter obs based on pct_deletion

        TODO: quantile removal
        """
        assert len(X) == len(self.vals_tmc)
        if pct_deletion is None:
            pct_deletion = self.pct_deletion
        lowest_shapley_value_to_tolerate = self.vals_tmc.min() * (1-self.pct_deletion)
        
        higher_than_tolerance_mask = [self.vals_tmc>lowest_shapley_value_to_tolerate]
        
        return X[higher_than_tolerance_mask], y[higher_than_tolerance_mask]