#!/usr/bin/python
# -*- coding: utf-8 -*
import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")


from sklearn.decomposition import PCA, KernelPCA
from scipy.spatial import distance
from sklearn.cluster import * #DBSCAN,KMeans,MeanShift,Ward,AffinityPropagation,SpectralClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import kneighbors_graph
#from sklearn.mixture import GMM, DPGMM
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from xlrd import open_workbook,cellname,XL_CELL_TEXT
#import openpyxl as px
import scipy as sp
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import  TimeSeriesSplit

#from mvpa2.suite import SimpleSOMMapper # http://www.pymvpa.org/examples/som.html
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import euclidean_distances

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

import numpy as np
import pylab as pl
import matplotlib.pyplot as pl
import pandas as pd
import os
import sys
import re
from scipy import stats
#from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import glob
import seaborn as sns
#from sklearn.cross_validation import cross_val_score, ShuffleSplit, LeaveOneOut, LeavePOut, KFold
#-------------------------------------------------------------------------------
def mean_std_percentual_error(y_true, y_pred):    
  y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
  s=np.mean(np.abs(y_pred - y_true)/np.abs(y_true))*100, np.std(np.abs(y_pred - y_true)/np.abs(y_true))*100
  return s
#-------------------------------------------------------------------------------
def max_div_cv(n):
  l=[]
  for i in range (1, n): 
    if (n%i == 0  and i<8): 
      l.append(i)	
      
  return max(4,max(l))
#-------------------------------------------------------------------------------
def read_iraq_solar(filename, model='M1'):
    #%%
    #filename='./data/data_iraq_solar/Baghdad-POWER_SinglePoint_Daily_19900101_20200331_033d32N_044d40E_9911c21f-.csv'
    X=pd.read_csv(filename, header=21, index_col=None, na_values=-999) 
    dt=X[['YEAR','MO', 'DY' ]]; dt.columns=['year', 'month', 'day']
    idx=pd.to_datetime(dt, yearfirst=True)
    X.index=idx
    X.drop(['LAT','LON', 'YEAR','MO', 'DY',], axis=1, inplace=True)
    
    for c in X.columns:
       #print(c)
       X[c].interpolate(method='linear', inplace=True)
       #X[c].plot(); pl.show()
       
    target='ALLSKY_SFC_LW_DWN'
    
    train_date = X.index <= '31-12-2017'
    test_date  = X.index >  '31-12-2017'
    
    y = X[[target]]
    X.drop(target, axis=1, inplace=True)
    
    cols=['RH2M', 'T2M_MAX', 'T2M_MIN', 'T2M', 
          'WS10M', 'TS', 'WS10M_MIN',
          'WS10M_MAX', 'KT', 'CLRSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DWN'
          ]
    #cols=['RH2M',  'WS10M',  'KT',   'WS10M_MIN', 'WS10M_MAX', ]
    X=X[cols]
    
    import seaborn as sns
    df=X.copy(); df[target]=y.values
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = pl.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap =  cmap="YlGnBu"
    sns.heatmap(corr, mask=mask, cmap=cmap, #vmax=.3, center=0, 
                annot=True, #fmt="d")
                square=True, linewidths=.5, #cbar_kws={"shrink": .5},
            )
    #pl.show()
    
    X_train, X_test, y_train, y_test = X[train_date], X[test_date], y[train_date], y[test_date] 
    
    n_samples, n_features = X_train.shape
    variable_names=np.array(X_train.columns)
    target_names=np.array(y_train.columns)
    data_description = ['var_'+str(i) for i in range(X_train.shape[1])]
    sn = os.path.basename(filename).split('-')[0].split('/')[-1]
    dataset=  {
      'task':'forecast',
      'name':'Solar Radiation '+sn,
      'feature_names':variable_names,'target_names':target_names,
      'n_samples':n_samples, 'n_features':n_features,
      'X_train':X_train.values,
      'X_test':X_test.values,
      'y_train':np.array([y_train.values.ravel()]),
      'y_test':np.array([y_test.values.ravel()]),      
      'targets':target_names,
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'items':None,
      'reference':"",      
      'stations':[sn],
      'normalize': 'MinMax',
      }   
    #%%
    return dataset
#-------------------------------------------------------------------------------
def read_iraq_solar_baghdad(default_path = "./", filename='data/data_iraq_solar/Baghdad-POWER_SinglePoint_Daily_19900101_20200331_033d32N_044d40E_9911c21f-.csv'):
    return(read_iraq_solar(filename=os.path.join(default_path, filename)))    
#-------------------------------------------------------------------------------
def read_iraq_solar_basrah(default_path = "./", filename='data/data_iraq_solar/Basrah-POWER_SinglePoint_Daily_19900101_20200331_030d48N_047d82E_c50b78a0.csv'):
    return(read_iraq_solar(filename=os.path.join(default_path, filename)))    
#-------------------------------------------------------------------------------
def read_iraq_solar_dahuk(default_path = "./", filename='data/data_iraq_solar/Dahuk-POWER_SinglePoint_Daily_19900101_20200331_036d91N_043d70E_56116557.csv'):
    return(read_iraq_solar(filename=os.path.join(default_path, filename)))    
#-------------------------------------------------------------------------------
def read_iraq_solar_kut(default_path = "./", filename='data/data_iraq_solar/Kut-POWER_SinglePoint_Daily_19900101_20200331_032d51N_045d83E_fb85cc4e.csv'):
    return(read_iraq_solar(filename=os.path.join(default_path, filename)))    
#-------------------------------------------------------------------------------
def read_iraq_solar_mosul(default_path = "./", filename='data/data_iraq_solar/Mosul-POWER_SinglePoint_Daily_19900101_20200331_036d32N_043d11E_07086b4e.csv'):
    return(read_iraq_solar(filename=os.path.join(default_path, filename)))    
#-------------------------------------------------------------------------------
def read_iraq_solar_najaf(default_path = "./", filename='data/data_iraq_solar/Najaf-POWER_SinglePoint_Daily_19900101_20200331_031d99N_044d32E_6c953428.csv'):
    return(read_iraq_solar(filename=os.path.join(default_path, filename)))    
#-------------------------------------------------------------------------------
def read_iraq_solar_nasiriyah(default_path = "./", filename='data/data_iraq_solar/Nasiriyah-POWER_SinglePoint_Daily_19900101_20200331_031d60N_046d25E_825fd410.csv'):
    return(read_iraq_solar(filename=os.path.join(default_path, filename)))    
#-------------------------------------------------------------------------------
def read_iraq_solar_rutbah(default_path = "./", filename='data/data_iraq_solar/Rutbah-POWER_SinglePoint_Daily_19900101_20200331_033d50N_040d38E_e1037ea7.csv'):
    return(read_iraq_solar(filename=os.path.join(default_path, filename)))    
#-------------------------------------------------------------------------------
def read_iraq_solar_sulaimaniayah(default_path = "./", filename='data/data_iraq_solar/Sulaimaniayah-POWER_SinglePoint_Daily_19900101_20200331_035d56N_045d45E_46b0a3db.csv'):
    return(read_iraq_solar(filename=os.path.join(default_path, filename)))    
#-------------------------------------------------------------------------------
def read_iraq_solar_tikrit(default_path = "./", filename='data/data_iraq_solar/Tikrit-POWER_SinglePoint_Daily_19900101_20200331_034d61N_043d69E_00199300.csv'):
    return(read_iraq_solar(filename=os.path.join(default_path, filename)))    
#-------------------------------------------------------------------------------

def read_iraq_stations():
    #%%
    datasets = [
            read_iraq_solar_baghdad(),
            read_iraq_solar_basrah(),
            read_iraq_solar_dahuk(),
            read_iraq_solar_kut(),
            read_iraq_solar_mosul(),
            read_iraq_solar_najaf(),
            read_iraq_solar_nasiriyah(),
            read_iraq_solar_rutbah(),
            read_iraq_solar_sulaimaniayah(),
            read_iraq_solar_tikrit(),            
           ]
    
    for ds in datasets:
        print('%19s'%ds['name'], ds['X_train'].shape, ds['X_test'].shape, )
    
    n_features, n_samples, n_datasets = ds['X_train'].shape[1],ds['X_train'].shape[0],len(datasets)
    X_train = np.zeros((n_samples, n_features, n_datasets,))
    y_train = np.zeros((n_samples, n_datasets,))
    
    stations=[]
    for i,ds in enumerate(datasets):
        n_features = len(ds['feature_names'])
        n_samples  = ds['X_train'].shape[0]
        #print(n_features,n_samples)
        X_train[:,:,i] =  ds['X_train']
        y_train[:,i] =  ds['y_train']
        stations.append([ds['name']])
    
    n_features, n_samples, n_datasets = ds['X_test'].shape[1],ds['X_test'].shape[0],len(datasets)
    X_test = np.zeros((n_samples, n_features, n_datasets,))
    y_test = np.zeros((n_samples, n_datasets,))
    
    for i,ds in enumerate(datasets):
        n_features = len(ds['feature_names'])
        n_samples  = ds['X_test'].shape[0]
        #print(n_features,n_samples)
        X_test[:,:,i] =  ds['X_test']
        y_test[:,i] =  ds['y_test']

    n_samples, n_features, _ = X_train.shape
    variable_names=['x_'+str(i) for i in range(n_features)] #np.array(X_train.columns)
    target_names=['y_'+str(i) for i in range(n_datasets)]#np.array(y_train.columns)
    data_description = ['var_'+str(i) for i in range(X_train.shape[1])]
    #sn = os.path.basename(filename).split('-')[0].split('/')[-1]
    dataset=  {
      'task':'forecast',
      'name':'Solar Radiation Iraq',
      'feature_names':variable_names,'target_names':target_names,
      'n_samples':n_samples, 'n_features':n_features,
      'X_train':X_train,
      'X_test':X_test,
      'y_train':y_train,
      'y_test':y_test,
      'targets':target_names,
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'items':None,
      'reference':"",      
      'stations':stations,
      'normalize': 'MinMax',
      }   
    #%%
    return dataset    
#%%
#
#
#
if __name__ == "__main__":
    datasets = [
                read_iraq_solar_baghdad(),
                read_iraq_solar_basrah(),
                read_iraq_solar_dahuk(),
                read_iraq_solar_kut(),
                read_iraq_solar_mosul(),
                read_iraq_solar_najaf(),
                read_iraq_solar_nasiriyah(),
                read_iraq_solar_rutbah(),
                read_iraq_solar_sulaimaniayah(),
                read_iraq_solar_tikrit(),
            ]
    for D in datasets:
        print('='*80+'\n'+D['name']+'\n'+'='*80)
        print(D['reference'])
        print( D['y_train'])
        print('\n')
#%%-----------------------------------------------------------------------------
