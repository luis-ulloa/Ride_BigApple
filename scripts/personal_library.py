import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from geopy.distance import geodesic

from sklearn.metrics import root_mean_squared_error


def plot_hist_box(df, col_name, title = None, xlabel = None):
    '''
    This function plots a histogram and a boxplot of a column.
    
    Parameters:
    col_name: string, this is the column's name
    title: string, the plot title.  Default value is a blank string.
    xlabel: string, the x-axis label.  Default value is a blank string.
    
    Return:
    It returns two plots.  A histogram on the left and a boxplot on the right.
    '''

    plt.figure(figsize = (16, 6))

    plot1 = plt.subplot(1, 2, 1)  # for histogram
    plot2 = plt.subplot(1, 2, 2)  # for box plot

    # histogram ---------------------------------------------------------------------------------------------
    plot1.hist(df[col_name], bins = 25, color = '#003366', edgecolor = '#ffffff');
    plot1.set_title(title, fontweight = 'bold', color = '#6e6e6e', fontsize = 14)
    plot1.set_xlabel(xlabel, color = '#6e6e6e', fontsize = 14)
    plot1.spines['top'].set_visible(False)
    plot1.spines['right'].set_visible(False)
    plot1.tick_params(axis = 'both', which = 'both', colors = '#6e6e6e', labelsize = 12);
    
    
    # box plot -----------------------------------------------------------------------------------------------
    sns.boxplot(data = df[col_name], ax = plot2, orient = 'h', color = '#003366', medianprops = {'color':'#FFFFFF'});
    plot2.set_title(title, fontweight = 'bold', color = '#6e6e6e', fontsize = 14);
    plot2.spines['top'].set_visible(False)
    plot2.spines['left'].set_visible(False)
    plot2.spines['right'].set_visible(False)
    plot2.set_xlabel(xlabel, color = '#6e6e6e', fontsize = 14)
    plot2.tick_params(axis = 'x', which = 'both', colors = '#6e6e6e', labelsize = 12);



def plot_heatmap(df, columns):
    '''
    This function plots a heatmap of the number of columns passed in.
    
    Parameters:
    columns: list, a list of columns to plot.
    
    Return:
    It returns a heatmap of the columns.
    '''

    plt.figure(figsize = (6, 6))

    # heatmap setup
    corrs = round(df[columns].corr(), 2)
    mask = np.zeros_like(corrs)
    mask[np.triu_indices_from(mask)] = True

    # plot heatmap
    sns.heatmap(corrs,
                square = True,
                annot = True,
                cmap = 'bone',
                mask = mask,
                vmin = -1,
                vmax = 1)




def get_geo_distance(long1, lat1, long2, lat2):
    '''
    This function measures the geodesic distance between two sets of coordinates.

    Parameters:
    long1: float, longitude, in degrees, of starting location.
    lat1: float, latitude, in degrees, of starting location.
    long2: float, longitude, in degrees, of ending location.
    lat2: float, latitude, in degrees, of ending location.

    Return:
    Geodesic distance (straight line over earth's surface) between starting and 
    ending locations.  It is in kilometers.
    '''

    pickup_point = (lat1, long1)
    dropoff_point = (lat2, long2)
    
    return geodesic(pickup_point, dropoff_point).kilometers



def model_evaluation(model, X_train, X_test, y_train, y_test):
    '''
    This function calculates and spits out the R2 scores for the
    training set and test set, as well as the RMSE.

    Parameters:
    model: object, the actual trained model
    X_train, X_test: dataframes, the training and test sets
    y_train, y_test: series, the training and test outputs

    Return: R2 scores and RMSE
    '''
    
    train_r2 = round(model.score(X_train, y_train), 3)
    test_r2 = round(model.score(X_test, y_test), 3)
    
    y_preds = model.predict(X_test)
    rmse = round(root_mean_squared_error(y_test, y_preds), 3)
    
    print(f'training r2: {train_r2}, test r2: {test_r2}, rmse: {rmse}')
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': rmse
    }
