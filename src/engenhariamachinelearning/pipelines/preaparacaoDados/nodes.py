import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn.metrics as metrics
import mlflow
from sklearn.metrics import mean_absolute_error, r2_score
import math
from xgboost import XGBRFRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pycaret.classification import ClassificationExperiment
from pycaret.classification import *
from pycaret.regression import *
from sklearn.metrics import log_loss

mlflow.start_run()

def x_y(conformed):
    x = conformed.iloc[:, :-1]
    return x

def y_test(conformed):
    y = conformed.iloc[:, 6:]
    return y

# Filtrar a coluna shot_type == 2PT Field Goal
def filtro_shortType():
    data1 = pd.read_csv('/engenhariamachinelearning/data/data.csv')
    data01 = onformed = data1 [['lat', 'shot_type', 'lon', 'minutes_remaining', 'period',
                                 'playoffs', 'shot_distance', 'shot_made_flag']]
    data01.dropna(inplace=True)
    filtrados = data1.loc[data1['shot_type'] == '2PT Field Goal']
    return filtrados

# Filtrar a coluna shot_type == 3PT Field Goal
def filtro_shortType3pt():
    data1 = pd.read_csv('/engenhariamachinelearning/data/data.csv')
    data01 = onformed = data1 [['lat', 'shot_type', 'lon', 'minutes_remaining', 'period',
                                 'playoffs', 'shot_distance', 'shot_made_flag']]
    data01.dropna(inplace=True)
    filtrados3 = data1.loc[data1['shot_type'] == '3PT Field Goal']
    return filtrados3

# Filtrar a coluna shot_type == 2PT Field Goal - MLFLOW
def shortType2pt():
    data2 = pd.read_csv('/engenhariamachinelearning/data/data.csv')
    data01 = onformed = data2 [['lat', 'shot_type', 'lon', 'minutes_remaining', 'period',
                                 'playoffs', 'shot_distance', 'shot_made_flag']]
    data01.dropna(inplace=True)
    filtrados2pt = data2.loc[data2['shot_type'] == '2PT Field Goal']
    return filtrados2pt

#Filtrar a coluna shot_type == 3PT Field Goal - MLFLOW
def shortType3pt():
    data3 = pd.read_csv('/engenhariamachinelearning/data/data.csv')
    data01 = onformed = data3 [['lat', 'shot_type', 'lon', 'minutes_remaining', 'period',
                                 'playoffs', 'shot_distance', 'shot_made_flag']]
    data01.dropna(inplace=True)
    filtrados3pt = data3.loc[data3['shot_type'] == '2PT Field Goal']
    return filtrados3pt

# testar csv
def data_conformada():
    data2 = pd.read_csv('/engenhariamachinelearning/data/data.csv')
    dfconf = data2 [['lat', 'shot_type', 'lon', 'minutes_remaining', 'period',
                        'playoffs', 'shot_distance', 'shot_made_flag']]
    dfconf.dropna(inplace=True)
    return dfconf

# função selecionar variaveis
def ConformData():
    data2 = pd.read_csv('/engenhariamachinelearning/data/data.csv')
    conformed = data2 [['lat', 'shot_type', 'lon', 'minutes_remaining', 'period',
                        'playoffs', 'shot_distance', 'shot_made_flag']]
    conformed.dropna(inplace=True)
    return conformed

# função retornar e salvar treino e teste
def svc(x,y):
    scaler = StandardScaler()
    scaler.fit(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test

# dimensões metricas
def pegar_metricas(y_test, x_train):
    metricay = y_test.shape[0]
    metricax = x_train.shape[0]
    return metricay, metricax

# tamanho da base de dados da predição
def dimData(conformed):
    dimensao_data = conformed.shape[0]
    return dimensao_data

# Algoritimos de classificação
def pycaret_mlsflow(conformed):
    s = setup(conformed, target = 'shot_made_flag', session_id = 123,
               n_jobs=-2, log_experiment = 'mlflow', experiment_name = 'classificador_nbakobe')
    exp = ClassificationExperiment()
    exp.setup(conformed, target = 'shot_made_flag', session_id = 123, n_jobes=-2, 
              log_experiment = 'mlflow', experiment_name = 'classificador_nbakobe')
    exp.add_metric('logloss', 'log Loss', log_loss, greater_is_better=False)

    exp.compare_models()

    return exp

# Algoritimos de regressão
def classificacao_pycaret(conformed):
    s = setup(conformed, target = 'shot_made_flag', session_id = 123,
                n_jobs=-2, log_experiment = 'mlflow', experiment_name = 'regressor_nbakobe')
    exp = RegressionExperiment()
    exp.setup(conformed, target = 'shot_made_flag', session_id = 123, n_jobes=-2, 
              log_experiment = 'mlflow', experiment_name = 'regressor_nbakobe')
    exp.add_metric('logloss', 'log Loss', log_loss, greater_is_better=False)

    exp.compare_models()

    return exp

# Algoritimos de regressão logistica
def regressaoLogistic(x_train):
    exp = RegressionExperiment()
    exp.setup(x_train, session_id = 456, n_jobs=-2, log_experiment = 'mlflow',
               experiment_name = 'logistic_refression')
    exp.add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False)
    model_lr = exp.create_model('lr')

    return model_lr