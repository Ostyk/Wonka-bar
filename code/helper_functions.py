import os
import numpy as np
import pandas as pd
import random
from matplotlib import rc
import scipy.stats as st
import matplotlib.pyplot as plt
import time
from uncertainties import unumpy
import itertools

plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.unicode']=True
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.metrics import classification_report, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

import imblearn
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

import seaborn as sns



def scale_set(train,test):
    """uses sklearn standard sclar to normalize data"""
    sc = StandardScaler()
    fitted = sc.fit(train)
    return sc.transform(train), sc.transform(test)

def standardize_data(X):
    sc = StandardScaler()
    fitted = sc.fit(X)
    return sc.transform(X)
    X = standardize_data(X)

def clssifier_type(X_train, X_test,y_train, y_test, p1=None, p2=None, classifier='SVM'):
    ''' Cross validation of different alogirthms
    p1, p2 - are hyperparameters
    '''
    if classifier=='SVM':
        clf = SVC(probability=True, C=p1,gamma=p2)
    elif classifier=='RANDOM-FOREST':
        clf = RandomForestClassifier(n_estimators=p1, max_depth=p2,random_state=0)

    clf.fit(X_train, y_train)
    y_predict=clf.predict(X_test)
    
    x = {"accuracy": accuracy_score(y_test, y_predict),
         "recall": precision_score(y_test, y_predict),
         "precision": recall_score(y_test, y_predict),
         "f1-score": f1_score(y_test, y_predict, average='weighted', labels=np.unique(y_predict))}
    metrics_output = pd.DataFrame(list(x.values()),index=list(x.keys()),columns=['model']).T
    
    #ROC
    y_score = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
    roc = [fpr, tpr]
    return metrics_output, roc

def plot_roc(rocs,balanced=True):
    '''plots the roc curve for a given model'''
    plt.figure(figsize=(10,7))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='cyan',
         label='Luck', alpha=.8)
    colors = ['b','k','r']
    for index, values in enumerate(rocs):
        tprs = values['tprs']
        aucs = values['aucs']
        mean_fpr = values['mean_fpr']
        model_name1 = values['model_name']

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr,
                 label='{} - AUC = {:.4f} $\pm$ {:.4f}'.format(model_name1, mean_auc, std_auc),
                 lw=2, alpha=.8,color=colors[index])

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.2,color=colors[index])


    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic of the {} model'.format(mode_name))
    plt.legend(loc="lower right")
    plt.show()
    
def correlation(dataset, threshold):
    '''checks the correlation of a dataframe'''
    threshold_pos = threshold
    threshold_neg = - threshold
    pos, neg = 0, 0
    deleted = []
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    print("before: ", dataset.shape)
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold_pos) and (corr_matrix.columns[j] not in col_corr):
                
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    pos+=1
                    deleted.append(colname)
                    del dataset[colname] # deleting the column from the dataset
                    #dataset.drop(colname, axis=1)
            elif (corr_matrix.iloc[i, j] <= threshold_neg) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    neg+=1
                    deleted.append(colname)
                    del dataset[colname] # deleting the column from the dataset
                    #dataset.drop(colname, axis=1)
    print("after: ", dataset.shape)
    print("pos: {}\tneg: {}".format(pos,neg))
    return dataset, deleted



def plot_2d_space(X, y, label='Classes'): 
    '''plots x and y in a 2D-spae'''
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    
def apply_oversampling(X_train, y_train, seed):
    
    sm = SMOTE(random_state=12, ratio = 1.0)
    x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    return x_train_res, y_train_res

def model(X, y, n_splits=5, seed = 42, model_name='SVM', over_sampling = False):
    '''
    Args: X-data, y-labels,
    n_splits - K-fold cross validation splits
    '''
    if model_name=='SVM':
        hp1 = [10]# 10, 100, 1000] #C
        hp2 = [1e-3]#, 1e-2, 1/8] # Gamma
    elif model_name=='RANDOM-FOREST':
        hp1 = [100,200,300] #n_estimators
        hp2 = [2,4,6,8] #max_depth


    permutations = [(x,y) for x in hp2 for y in hp1]

    kf = StratifiedKFold(n_splits=n_splits,random_state=seed, shuffle=True)
    np.random.seed(seed)
    baseline = pd.DataFrame(np.zeros(4)).T
    baseline.columns = ['accuracy','recall','precision','f1-score']
    for p1,p2 in permutations:
        #print("Gamma: {}, C = {}".format(p2,p1))
        empty = pd.DataFrame([],columns=['accuracy','recall','precision','f1-score'])
        tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 100)

        for train_index, test_index in kf.split(X,y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if over_sampling:
                X_train, y_train = apply_oversampling(X_train, y_train, seed)
          
            X_train, X_test = scale_set(X_train,X_test)

            #metrics calcs
            performance, roc = clssifier_type(X_train, X_test, y_train, y_test, p1, p2, classifier=model_name)
            empty = pd.concat([empty,performance])
            #roc curve calcs
            tprs.append(np.interp(mean_fpr, roc[0], roc[1]))
            tprs[-1][0] = 0.0
            roc_auc = auc(roc[0], roc[1])
            aucs.append(roc_auc)

        permuation_performance = empty.mean() #mean of scores for all CVs

        if np.array([permuation_performance['f1-score']])>=np.array([baseline['f1-score']]): #maximizing f1 score
            baseline = pd.DataFrame(unumpy.uarray(permuation_performance, empty.std())).T

            baseline.columns = ['accuracy','recall','precision','f1-score']
            baseline['param 1'],baseline['param 2'] = p1, p2
            tprs_best = tprs
            aucs_best = aucs
            mean_fpr_best = mean_fpr

    r = {"tprs":tprs_best,
         "aucs":aucs_best,
         "mean_fpr":mean_fpr_best,
         "model_name":model_name} #roc

    #returning part
    baseline.insert(0, 'Model', model_name)

    return baseline,r

def corr_heatmap(df):
    sns.set(style="white")

    # Generate a large random dataset
    # Compute the correlation matrix
    corr = df.corr()
    #corr = corr[corr>]

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
def plot_days(df):
    e = pd.concat([df['day'], df['GREEN']], axis=1)
    day_1 = e['day'].iloc[0]
    day_last = e['day'].iloc[-1]
    october = dict()
    for i in range(int(day_1), 31):
        normal, green = e[e['day']==i]['GREEN'].value_counts()
        #october.append([normal, green, i, 10])
        october.update({str(i): {0:normal,1:green}})
    november = dict()
    for i in range(1, int(day_last+1)):
        val = e[e['day']==i]['GREEN'].value_counts()
        if len(val)>1:
            november.update({str(i): {0:normal,1:green}})
        else:
            november.update({str(i): {0:normal,1:0}})
    q1, q2 = pd.DataFrame(october).T, pd.DataFrame(november).T
    q = pd.concat([q1,q2])
    p = q.plot(kind='bar', figsize=(20,5), logy=True, fontsize=24)
    p.set_xlabel("Days of production", fontsize=24)
    p.set_ylabel("Number of bars", fontsize=24)
    p.set_title("October to November comparision Wonka bars production $0 -$ normal, $1 -$green", fontsize=24)
    p.legend(loc='best', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True, fontsize=24)
    
def data_processing(df):
    use_time = True
    if use_time:
        to_vec=[]
        for i in df['TIME']:
            t = time.strptime(i, '%Y-%m-%d %H:%M:%S')
            #print(t.tm_mon)
            to_vec.append({"day": t.tm_mday,
                           "dayWeek": t.tm_wday,
                           "hour": t.tm_hour,
                           "min": t.tm_min,
                           "month": t.tm_mon,
                           })

        vec = DictVectorizer()
        time_1 = vec.fit_transform(to_vec).toarray()
        time_ = pd.DataFrame(time_1)
        time_.columns = ['day', 'dayWeek', 'hour', 'min', 'month']
        df = pd.concat([time_, df],  axis=1)
    return df