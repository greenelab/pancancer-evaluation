#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# In[2]:


n = 200
p = 500
n_informative = 20
n_redundant = 20
n_clusters_per_class = 2
class_weights = [0.6, 0.4]

standardize = True
preprocess_pca = False


# In[3]:


def generate_dataset(preprocess_pca=False,
                     pca_n_components=50,
                     random_state=42):
    X, y = make_classification(n_samples=n,
                               n_features=p,
                               n_informative=n_informative,
                               n_redundant=n_redundant,
                               n_clusters_per_class=n_clusters_per_class,
                               weights=class_weights,
                               class_sep=1.75,
                               flip_y=0.0,
                               random_state=random_state)
    if standardize:
        X = StandardScaler().fit_transform(X)
    if preprocess_pca:
        X = PCA(n_components=pca_n_components).fit_transform(X)
    return X, y

X, y = generate_dataset(preprocess_pca)

print(X.shape)
print(X[:5, :5])


# In[4]:


print(y.shape)
print(y[:5])


# In[5]:


def get_prob_metrics(y_train, y_test, y_pred_train, y_pred_test):
    """Given predicted scores and true labels for train and test samples,
    calculate classification metrics.

    y_pred_train and y_pred_test should be continuous (higher = more likely
    to be positively labeled, e.g. output of logistic regression); true values
    are binary.
    """
    train_auroc = roc_auc_score(y_train, y_pred_train, average="weighted")
    test_auroc = roc_auc_score(y_test, y_pred_test, average="weighted")

    train_aupr = average_precision_score(y_train, y_pred_train, average="weighted")
    test_aupr = average_precision_score(y_test, y_pred_test, average="weighted")

    return {
        'train_auroc': train_auroc,
        'test_auroc': test_auroc,
        'train_aupr': train_aupr,
        'test_aupr': test_aupr,
    }


# In[6]:


def get_loss(y_train,
             y_pred_train,
             penalty=None,
             weights=None,
             param=None):
    log_likelihood = log_loss(y_train, y_pred_train)
    penalty_fn = 0.0
    if penalty == 'l1':
        penalty_fn = param * np.sum(np.absolute(weights))
    elif penalty == 'l2':
        penalty_fn = (0.5 * param * np.sum(weights ** 2))
    return log_likelihood, penalty_fn

def get_eta0(lr_schedule):
    eta0 = 0.0
    if lr_schedule in ['invscaling', 'adaptive']:
        eta0 = 0.1
    elif lr_schedule == 'constant':
        eta0 = 0.005
    return eta0


# In[7]:


results = []
results_cols = None
seeds = [42, 1]
coefs = {
    'constant': [],
    'optimal': [],
    'invscaling': [],
    'adaptive': []
}
# no regularization for now
for seed in seeds:
    
    X, y = generate_dataset(preprocess_pca=preprocess_pca, random_state=seed)
    
    kf = KFold(n_splits=4)
    for fold, (train_ix, test_ix) in enumerate(kf.split(X)):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        for lr_schedule in coefs.keys():
            
            eta0 = get_eta0(lr_schedule)
            
            sgd = SGDClassifier(
                loss='log_loss',
                random_state=seed,
                learning_rate=lr_schedule,
                eta0=eta0
            ).fit(X_train, y_train)

            y_pred_train = sgd.predict(X_train)
            y_pred_test = sgd.predict(X_test)

            prob_metrics = get_prob_metrics(y_train, y_test,
                                            y_pred_train, y_pred_test)

            if results_cols is None:
                results_cols = list(prob_metrics.keys()) + ['lr_schedule', 'seed', 'fold']
            else:
                assert results_cols == (list(prob_metrics.keys()) + ['lr_schedule', 'seed', 'fold'])
            results.append(list(prob_metrics.values()) + [lr_schedule, seed, fold])

            coefs[lr_schedule].append([seed, fold, sgd.coef_])

    results_df = (
        pd.DataFrame(results, columns=results_cols)
          .melt(id_vars=['lr_schedule', 'seed', 'fold'], var_name='metric')
    )

print(results_df.shape)
results_df.head()


# In[8]:


# get coefficient vector magnitudes
baseline_coefs_df = []

for lr_schedule in coefs.keys():
    for [seed, fold, coef] in coefs[lr_schedule]:
        baseline_coefs_df.append([lr_schedule, seed, fold, np.sum(np.absolute(coef))])
    
baseline_coefs_df = pd.DataFrame(
    baseline_coefs_df, columns=['lr_schedule', 'seed', 'fold', 'sum_coefs']
)

print(baseline_coefs_df.shape)
baseline_coefs_df.head()


# In[9]:


sns.set({'figure.figsize': (10, 6)})

sns.boxplot(data=results_df.sort_values(by='metric', ascending=False),
            x='metric', y='value', hue='lr_schedule')


# In[10]:


results = []
results_cols = None

coefs = {
    'constant': [],
    'optimal': [],
    'invscaling': [],
    'adaptive': []
}

losses = []

seeds = [42, 1]
sgd_params = np.logspace(-8, 1, 19)

for seed in seeds:
    
    X, y = generate_dataset(preprocess_pca=preprocess_pca, random_state=seed)
    
    kf = KFold(n_splits=4)

    # L1 regularization
    for fold, (train_ix, test_ix) in enumerate(kf.split(X)):
        print('seed:', seed, 'fold:', fold)
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        for lr_schedule in coefs.keys():
            
            eta0 = get_eta0(lr_schedule)
            
            for sgd_param in sgd_params:
                sgd = (
                    SGDClassifier(loss='log_loss',
                                  penalty='l1',
                                  alpha=sgd_param,
                                  random_state=seed,
                                  learning_rate=lr_schedule,
                                  eta0=eta0,
                                  max_iter=500)
                ).fit(X_train, y_train)
                y_pred_train = sgd.predict(X_train)
                y_pred_test = sgd.predict(X_test)
                prob_metrics = get_prob_metrics(y_train, y_test,
                                                y_pred_train, y_pred_test)
                if results_cols is None:
                    results_cols = (list(prob_metrics.keys()) +
                                    ['lr_schedule', 'seed', 'fold', 'lasso_param'])
                else:
                    assert results_cols == (list(prob_metrics.keys()) +
                                            ['lr_schedule', 'seed', 'fold', 'lasso_param'])
                results.append(list(prob_metrics.values()) +
                               [lr_schedule, seed, fold, sgd_param])
                coefs[lr_schedule].append([seed, fold, sgd_param, sgd.coef_])

                sgd_logl, sgd_penalty = get_loss(
                    y_train, y_pred_train,
                    penalty='l1', weights=sgd.coef_, param=sgd_param
                )
                losses.append([lr_schedule, seed, fold, sgd_param, sgd_logl, sgd_penalty])

results_df = (
    pd.DataFrame(results, columns=results_cols)
      .melt(id_vars=['lr_schedule', 'seed', 'fold', 'lasso_param'], var_name='metric')
)

print(results_df.shape)
results_df.head()


# In[11]:


sns.set_style('ticks')

plot_df = results_df[results_df.metric.isin(['train_aupr', 'test_aupr'])]

with sns.plotting_context('notebook', font_scale=1.6):
    g = sns.relplot(
        data=plot_df,
        x='lasso_param', y='value', hue='metric',
        hue_order=['train_aupr', 'test_aupr'],
        marker='o', kind='line', col='lr_schedule',
        col_wrap=2, height=5, aspect=1.6,
        facet_kws={'sharex': False}
    )
    g.set(xscale='log', xlim=(10e-9, 10), ylim=(-0.05, 1.05))
    g.set_xlabels('LASSO parameter (lower = less regularization)')
    g.set_ylabels('AUPR')
    sns.move_legend(g, 'center', bbox_to_anchor=[1.035, 0.55], frameon=True)
    g._legend.set_title('Dataset')
    new_labels = ['train', 'test']
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)
    g.set_titles('Learning rate schedule: {col_name}')
     
    plt.suptitle(f'LASSO parameter vs. AUPR, simulated data, L1 regularization', y=1.0)

plt.tight_layout()


# In[12]:


# get coefficient vector magnitudes
coefs_df = []

for lr_schedule in coefs.keys():
    for [seed, fold, param, coef] in coefs[lr_schedule]:
        coefs_df.append([lr_schedule, seed, fold, param, np.sum(np.absolute(coef))])
    
coefs_df = pd.DataFrame(
    coefs_df, columns=['lr_schedule', 'seed', 'fold', 'lasso_param', 'sum_coefs']
)

print(coefs_df.shape)
coefs_df.head()


# In[13]:


sns.set_style('ticks')

with sns.plotting_context('notebook', font_scale=1.6):
    g = sns.relplot(
        data=coefs_df,
        x='lasso_param', y='sum_coefs',
        marker='o', kind='line', col='lr_schedule',
        col_wrap=2, height=5, aspect=1.6,
        facet_kws={'sharex': False}
    )
    g.set(xscale='log', yscale='log', xlim=(10e-9, 10))
    g.set_xlabels('LASSO parameter (lower = less regularization)')
    g.set_ylabels('Sum of coefficient weights')
    g.set_titles('Learning rate schedule: {col_name}')
    for i, ax in enumerate(g.axes):
        lr_schedule = list(coefs.keys())[i]
        ax.axhline(baseline_coefs_df[baseline_coefs_df.lr_schedule == lr_schedule].sum_coefs.mean(), linestyle='--')
     
    plt.suptitle(f'LASSO parameter vs. sum of coefficient weights, simulated data, L1 regularization', y=1.0)

plt.tight_layout()


# In[14]:


# get coefficient vector magnitudes
losses_df = pd.DataFrame(
    losses,
    columns=['lr_schedule', 'seed', 'fold', 'lasso_param',
             'log_likelihood', 'weight_penalty']
)
losses_df['total_loss'] = losses_df.log_likelihood + losses_df.weight_penalty

losses_df = losses_df.melt(
    id_vars=['lr_schedule', 'seed', 'fold', 'lasso_param'],
    var_name='loss_component', value_name='loss_value'
)

print(losses_df.shape)
losses_df.head()


# In[15]:


sns.set_style('ticks')

with sns.plotting_context('notebook', font_scale=1.6):
    g = sns.relplot(
        data=losses_df,
        x='lasso_param', y='loss_value', hue='loss_component',
        hue_order=['log_likelihood', 'weight_penalty', 'total_loss'],
        marker='o', kind='line', col='lr_schedule',
        col_wrap=2, height=5, aspect=1.6,
        facet_kws={'sharex': False}
    )
    g.set(xscale='log',  xlim=(10e-9, 10))
    g.set_xlabels('LASSO parameter (lower = less regularization)')
    g.set_ylabels('Sum of coefficient weights')
    g.set_titles('Learning rate schedule: {col_name}')
    g.set_ylabels('Loss value')
    sns.move_legend(g, "center", bbox_to_anchor=[1.075, 0.55], frameon=True)
    g._legend.set_title('Loss component')
     
    plt.suptitle(f'LASSO parameter vs. loss, simulated data, L1 regularization', y=1.0)

plt.tight_layout()

