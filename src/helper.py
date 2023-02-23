#########################################
# Import Libraries & Option Setup
#########################################

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
import itertools
import re
#import pyreadr
#import plotly.express as px

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

################################################
# 1. Exploratory Data Analysis
################################################
def check_df(dataframe, head=5):
    """
    Helper function to print out basic data properties such as:
    dataframe shape (rows, columns)
    dataframe dtypes | data types of the features
    top n entries in the dataframe (n specified with the variable head)
    last n entries in the dataframe (n specified with the variable head)
    total number of null values in each feature
    statistical values of numerical features in the dataframe, quantiles are specified as: [0, 0.05, 0.50, 0.95, 0.99, 1]

    Parameters
    ----------
    dataframe - dataframe to be invested
    head - number of entries to be printed out (head and tail)

    Returns
    -------

    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)



############## CAT_COLS, NUM_COLS ###############

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
        Returns the names of categorical, numerical and categorical but cardinal variables in the dataframe

        Parameters
        ------
            dataframe: dataframe
                    dataframe
            cat_th: int, optional
                    threshold for the variables that have numerical values but are categorical
            car_th: int, optinal
                    threshold for the variables that are categorical but cardinal

        Returns
        ------
            cat_cols: list
                    list of categorical variables
            num_cols: list
                    list of numerical variables
            cat_but_car: list
                    list of cardinal variables

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = all variables
            num_but_cat is inside cat_cols

        """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


# # Define Variables
# cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

############## CATEGORICAL ###############

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        b= sns.countplot(x=dataframe[col_name], data=dataframe)
        b.set_xlabel(col_name, fontsize=50)
        b.tick_params(labelsize=5)
        plt.show(block=True)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

### bu fonksiyon önemli!
#rare_analyser(df, "TARGET", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

#new_df = rare_encoder(df, 0.01)

#rare_analyser(new_df, "TARGET", cat_cols)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

############## NUMERICAL ###############

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

# # Investigating Categorical features
# for col in cat_cols:
#     cat_summary(df, col)
#
# # Investigating Numerical features
# df[num_cols].describe().T
#
# # for col in num_cols:
# #     num_summary(df, col, plot=True)


def create_boxplot(num_df):
    num_df = num_df.dropna()
    red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

    fig, axs = plt.subplots(1, len(num_df.columns), figsize=(20, 10))

    for i, ax in enumerate(axs.flat):
        ax.boxplot(num_df.iloc[:, i], flierprops=red_circle)
        num_df.iloc[:, i]
        ax.set_title(num_df.columns[i], fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()

# in order to call this first create a num_df with only numerical columns
# num_df


############## CORRELATION ###############

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

# Drop Highly Correlated Values

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

# # Correlation between numerical features
# correlation_matrix(df, num_cols)

# Analyse Target with Categorical & Numerical Features

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

# # Investigating the relationship between Target features - Numerical Features
# for col in num_cols:
#     target_summary_with_num(df, "Outcome", col)


################################################
# 2. Data Preprocessing & Feature Engineering
################################################
############# OUTLIERS ###############

def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquartile_range)
    low_limit = round(quartile1 - 1.5 * interquartile_range)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquartile_range)
    low_limit = round(quartile1 - 1.5 * interquartile_range)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

def remove_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquartile_range)
    low_limit = round(quartile1 - 1.5 * interquartile_range)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

############# MISSING VALUES ###############

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

# missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


#missing_vs_target(df, "Survived", na_cols)

#df["Embarked"].fillna("missing")

# filling with median values median ile oldurma
#df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# fill categorical with mode
#df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# fill categorical values in relation with numerical features
#df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()



############# LABEL - ONE HOT ENCODER ###############

def label_encoder(dataframe, binary_col):
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe







#########################################
# ML MODELS
#########################################

#########################################
# CLASSIFICATION
#########################################

def base_classification_models(X, y, scoring="roc_auc"):
    # Base Classification Models
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, \
        AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    from sklearn.model_selection import cross_validate

    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

#base_models(X, y, scoring="accuracy")


#check if you find a better plot!


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.rcParams.update({'font.size': 19})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontdict={'size':'16'})
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=12,color="blue")
    plt.yticks(tick_marks, classes,fontsize=12,color="blue")
    rc('font', weight='bold')
    fmt = '.1f'
    thresh = cm.max()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red")

    plt.ylabel('True label',fontdict={'size':'16'})
    plt.xlabel('Predicted label',fontdict={'size':'16'})
    plt.tight_layout()

# plot_confusion_matrix(confusion_matrix(y_test, y_pred=y_pred), classes=['Non Fraud','Fraud'],
#                       title='Confusion matrix')


# Auc Roc Curve
def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr)
    plt.show()
    pass

#generate_auc_roc_curve(model, X_test)



#########################################
# REGRESSION
#########################################

def base_regression_models(X, y, scoring="neg_root_mean_squared_error"):
    # Base Classification Models
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    from sklearn.model_selection import cross_validate
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    print("Base Models....")
    regressions = [('LIN', LinearRegression()),
                   ('RIDGE', Ridge()),
                   ('LASSO', Lasso()),
                   ('ELAS', ElasticNet()),
                   ('KNN', KNeighborsRegressor()),
                   ("SVR", SVR()),
                   ("RF", RandomForestRegressor()),
                   ('GBM', GradientBoostingRegressor()),
                   ('XGBoost', XGBRegressor()),
                   ('LightGBM', LGBMRegressor()),
                   ('CatBoost', CatBoostRegressor(verbose=False))
                   ]

    for name, regression in regressions:
        cv_results = cross_validate(regression, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


#########################################
# PCA
#########################################

def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

#pca_df = create_pca_df(X, y)

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

#plot_pca(pca_df, "diagnosis")