import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class SimpleBayesClassifier:

    def __init__(self, n_pos, n_neg):
        
        """
        Initializes the SimpleBayesClassifier with prior probabilities.

        Parameters:
        n_pos (int): The number of positive samples.
        n_neg (int): The number of negative samples.
        
        Returns:
        None: This method does not return anything as it is a constructor.
        """

        self.n_pos = n_pos
        self.n_neg = n_neg
        self.prior_pos = n_pos / (n_pos + n_neg)
        self.prior_neg = n_neg / (n_pos + n_neg)

    def fit_params(self, x, y, n_bins = 10):

        """
        Computes histogram-based parameters for each feature in the dataset.

        Parameters:
        x (np.ndarray): The feature matrix, where rows are samples and columns are features.
        y (np.ndarray): The target array, where each element corresponds to the label of a sample.
        n_bins (int): Number of bins to use for histogram calculation.

        Returns:
        (stay_params, leave_params): A tuple containing two lists of tuples, 
        one for 'stay' parameters and one for 'leave' parameters.
        Each tuple in the list contains the bins and edges of the histogram for a feature.
        """

        self.stay_params = [(None, None) for _ in range(x.shape[1])]
        self.leave_params = [(None, None) for _ in range(x.shape[1])]

        # INSERT CODE HERE
        
        # Get the features for the positive and negative classes
        pos_features = x[y == 1]
        neg_features = x[y == 0]
        
        n_feats = x.shape[1]
        
        # Calculate the params for each feature
        for i in range(n_feats):
            
            # Prepare features (Positive)
            pos_features_i = pos_features[:, i]
            pos_features_i = pos_features_i[~np.isnan(pos_features_i)]
            # Prepare features (Negative)
            neg_features_i = neg_features[:, i]
            neg_features_i = neg_features_i[~np.isnan(neg_features_i)]
            
            # Calculate max_val and min_val to be boundaries of the histogram(Positive)
            max_val_pos = max(pos_features_i)
            min_val_pos = min(pos_features_i)
            # Calculate max_val and min_val to be boundaries of the histogram(Negative)
            max_val_neg = max(neg_features_i)
            min_val_neg = min(neg_features_i)
            
            # Make a basis for the histogram (Positive)
            basis_pos = np.linspace(min_val_pos, max_val_pos, n_bins-1) # Subtract 1 because it start from left edge
            # Make a basis for histogram (Negative)
            basis_neg = np.linspace(min_val_neg, max_val_neg, n_bins-1)

            # Append lower bound and upper bound for avoiding the edge case (Positive)
            basis_pos = np.append(basis_pos, 1e6)
            basis_pos = np.insert(basis_pos, 0, -1e6)
            # Append lower bound and upper bound for avoiding the edge case (Negative)
            basis_neg = np.append(basis_neg, 1e6)
            basis_neg = np.insert(basis_neg, 0, -1e6)
            
            # Discretize the features (Positive)
            digitized_pos = np.digitize(pos_features_i, basis_pos) # Default doesn't include right edge
            bins_pos = np.bincount(digitized_pos)[1:] # Since we don't want left edge which is [-inf, -1e6]
            # Discretize the features (Negative)
            digitized_neg = np.digitize(neg_features_i, basis_neg)
            bins_neg = np.bincount(digitized_neg)[1:]
            
            # Normalize the bins (Positive)
            bins_pos = bins_pos / sum(bins_pos)
            # Normalize the bins (Negative)
            bins_neg = bins_neg / sum(bins_neg)
            
            # Flooring the zero bins value (Positive)
            bins_pos = np.where(bins_pos == 0, 1e-6, bins_pos)
            # Flooring the zero bins value (Negative)
            bins_neg = np.where(bins_neg == 0, 1e-6, bins_neg)
            
            self.leave_params[i] = (bins_pos, basis_pos)
            self.stay_params[i] = (bins_neg, basis_neg)
            
        
        return self.stay_params, self.leave_params

    def predict(self, x, thresh = 0):

        """
        Predicts the class labels for the given samples using the non-parametric model.

        Parameters:
        x (np.ndarray): The feature matrix for which predictions are to be made.
        thresh (float): The threshold for log probability to decide between classes.

        Returns:
        result (list): A list of predicted class labels (0 or 1) for each sample in the feature matrix.
        """

        y_pred = []

        # INSERT CODE HERE
        
        # Loop through each feature's sample
        n_feats = x.shape[1]
        
        # Make a zeros array
        x_computed = np.zeros_like(x)
        
        for i in range(n_feats):
                
                # Get the bins and edges of the histogram for the feature
                bins_pos, basis_pos = self.leave_params[i]
                bins_neg, basis_neg = self.stay_params[i]
                
                # Check if there is null value if so drop the value from vector x_i
                x_i = x[:, i]
                nan_mask_x_i = np.isnan(x_i)
                x_i = x_i[~nan_mask_x_i]
                
                # Discretize the features
                digitized_pos = np.digitize(x_i, basis_pos)
                digitized_neg = np.digitize(x_i, basis_neg)
                
                # Get likelihood value w.r.t to bins
                likelihood_pos_i = bins_pos[digitized_pos-1]
                likelihood_neg_i = bins_neg[digitized_neg-1]
                
                # Calculate the difference between the likelihoods (log)
                log_likelihood_diff = np.log(likelihood_pos_i) - np.log(likelihood_neg_i)
                
                # Store the computed value
                x_computed[~nan_mask_x_i, i] = log_likelihood_diff
                x_computed[nan_mask_x_i, i] = 0
            
        # Sum for each sample and add the prior
        log_prior = np.log(self.prior_pos) - np.log(self.prior_neg)
        log_posterior = np.sum(x_computed, axis=1) + log_prior
        
        # Classify the sample based on the threshold in list
        y_pred = np.where(log_posterior > thresh, 1, 0).tolist()

        return y_pred
    
    def fit_gaussian_params(self, x, y):

        """
        Computes mean and standard deviation for each feature in the dataset.

        Parameters:
        x (np.ndarray): The feature matrix, where rows are samples and columns are features.
        y (np.ndarray): The target array, where each element corresponds to the label of a sample.

        Returns:
        (gaussian_stay_params, gaussian_leave_params): A tuple containing two lists of tuples,
        one for 'stay' parameters and one for 'leave' parameters.
        Each tuple in the list contains the mean and standard deviation for a feature.
        """

        self.gaussian_stay_params = [(0, 0) for _ in range(x.shape[1])]
        self.gaussian_leave_params = [(0, 0) for _ in range(x.shape[1])]

        # INSERT CODE HERE
        
        # Get the features for the positive and negative classes
        pos_features = x[y == 1]
        neg_features = x[y == 0]
        
        n_feats = x.shape[1]
        
        # Calculate the params for each feature with assumed Gaussian distribution
        for i in range(n_feats):
            # Calculate the mean and standard deviation for each feature
            mean_pos = np.nanmean(pos_features[:, i])
            std_pos = np.nanstd(pos_features[:, i])
            mean_neg = np.nanmean(neg_features[:, i])
            std_neg = np.nanstd(neg_features[:, i])
            
            self.gaussian_leave_params[i] = (mean_pos, std_pos)
            self.gaussian_stay_params[i] = (mean_neg, std_neg)
        
        return self.gaussian_stay_params, self.gaussian_leave_params
    
    def gaussian_predict(self, x, thresh = 0):

        """
        Predicts the class labels for the given samples using the parametric model.

        Parameters:
        x (np.ndarray): The feature matrix for which predictions are to be made.
        thresh (float): The threshold for log probability to decide between classes.

        Returns:
        result (list): A list of predicted class labels (0 or 1) for each sample in the feature matrix.
        """

        y_pred = []

        # INSERT CODE HERE
        
        # Loop through each feature's sample
        n_feats = x.shape[1]
        
        # Make a zeros array
        x_computed = np.zeros_like(x)
        
        for i in range(n_feats):
            
            # Get the mean and standard deviation for the feature
            mean_pos, std_pos = self.gaussian_leave_params[i]
            mean_neg, std_neg = self.gaussian_stay_params[i]
            
            # Check if there is null value if so drop the value from vector x_i
            x_i = x[:, i]
            nan_mask_x_i = np.isnan(x_i)
            x_i = x_i[~nan_mask_x_i]
            
            # Calculate the likelihood value w.r.t to Gaussian distribution
            likelihood_pos_i = stats.norm(mean_pos, std_pos).pdf(x_i)
            likelihood_neg_i = stats.norm(mean_neg, std_neg).pdf(x_i)
            
            # Calculate the difference between the likelihoods (log)
            log_likelihood_diff = np.log(likelihood_pos_i + 1e-9) - np.log(likelihood_neg_i + 1e-9)
            
            # Store the computed value
            x_computed[~nan_mask_x_i, i] = log_likelihood_diff
            
            # Set the value of null value to be 0
            x_computed[nan_mask_x_i, i] = 0
        
        # Sum for each sample and add the prior
        log_prior = np.log(self.prior_pos) - np.log(self.prior_neg)
        log_posterior = np.sum(x_computed, axis=1) + log_prior
        
        # Classify the sample based on the threshold in list
        y_pred = np.where(log_posterior > thresh, 1, 0).tolist()

        return y_pred