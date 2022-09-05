import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def plot_residuals(y, yhat):
    """ 
    """

    residuals = y - yhat

    plt.scatter(x=y, y=residuals)
    plt.xlabel('Tax Value')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Tax Value');
    
def regression_errors(y , yhat):
    """ 
    """

    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = sqrt(MSE)
    ESS = ((yhat - y.mean()) ** 2).sum()
    TSS= ESS + SSE

    return SSE, ESS, TSS, MSE, RMSE 

def baseline_mean_errors(y):
    """ 
    Purpose
        Computes the SSE, MSE, RMSE for a baseline model
    """
    baseline = np.repeat(y.mean(), len(y))

    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = sqrt(MSE)

    return SSE, MSE, RMSE
    
def better_than_baseline(y, yhat):
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y, yhat)
    
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    
    if SSE < SSE_baseline:
        print('My model performs better than baseline')
    else:
        print('My  model performs worse than baseline. :( )')