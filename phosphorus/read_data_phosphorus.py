# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#pl.rc('text', usetex=True)
#pl.rc('font',**{'family':'serif','serif':['Palatino']})

dataset='P'; plot=True; seed=0

def read_ucs(dataset):
    
    if dataset=='P':
        return read_phosphorus(target='P', dataset=dataset)    
    
#%%
def read_phosphorus(target='P', dataset=None, seed=None, plot=True):
    #%% 
    fn='./data/data_phosphorus_yangtze/phosphorus_yangtze.xlsx'
    X = pd.read_excel(fn)
    cols_to_drop=['Sample ID']
    X.drop(cols_to_drop,axis=1,inplace=True)
    target_names=['P (mg/L)']
    
    variable_names=list(X.columns.drop(target_names))
       
    X=X[variable_names+target_names]
    X.dropna(inplace=True)
       
    for cc in X.columns:
        #print(cc)       
        X[cc] = X[cc].astype(float)
        
        
    n=220
    X_train, y_train = X[variable_names][:n], X[target_names][:n]
    X_test , y_test  = X[variable_names][n:], X[target_names][n:]
    X_train, X_test, y_train, y_test = train_test_split(X[variable_names], X[target_names], test_size=0.2, random_state=seed)

    y=X[target_names]

    for (nam, X_, y_) in [('All', X,y), ('Training', X_train,y_train), ('Test', X_test,y_test)]:

        df = X_[variable_names].copy()
        df[target_names]= y_[target_names]
       
        if plot:
            pl.figure(figsize=(4, 4))
            corr = df.corr()
            mask = np.triu(np.ones_like(corr, dtype=np.bool_))
            heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
            #heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
            heatmap.set_title(dataset+': Correlation Heatmap ('+nam+')', fontdict={'fontsize':12}, pad=12);
            pl.savefig(dataset+'_heatmap_correlation_'+nam+'.png',  bbox_inches='tight', dpi=300)
            pl.show()
        
    n=len(y_train);     
    n_samples, n_features = X_train.shape 

    df_train=X_train.copy(); df_train[target_names]=y_train
    stat_train = df_train.describe().T
    #print(stat_train.to_latex(),)
    stat_train.to_latex(buf=(dataset+'_train'+'.tex').lower(), index=True, caption='Basic statistics for dataset '+dataset+'.')

    
    task = 'regression' if target_names[0]=='UCS' else 'classification'
    task = 'regression'
    regression_data =  {
      'task'            : task,
      'name'            : dataset,
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train.values,
      'y_train'         : y_train.values.T,
      'X_test'          : X_test.values,
      'y_test'          : y_test.values.T,
      'targets'         : target_names,
       #'true_labels'     : classes,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'reference'       : "",
      'items'           : None,
      'normalize'       : None,
      }
    #%%
    return regression_data
        
#%%


if __name__ == "__main__": 
    
    #D = read_gajurel(target='UCS', treatment='Lime')
    #D = read_gajurel(target='UCS', treatment='Cement')
    #D = read_kardani()
    ds_names=['P' for i in range(1)]
    for d in ds_names:
        D=read_ucs(d)
        print(D['name'], D['n_samples'])
        print(D['feature_names'])
    
