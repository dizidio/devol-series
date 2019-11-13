import pandas as pd
import operator
import math

df = pd.read_excel("./data/txcambio_error_predict.xlsx")

lag = 10

for i in range(1,lag):
    df['VALOR REAL {}'.format(i)] = df['VALOR REAL'].shift(i)

df['ERRO_ARIMA'] = df['VALOR REAL'] - df['ARIMA']
df['ERRO_SVM'] = df['SVR'] - df['ERRO_ARIMA']
df['ERRO_MLP'] = df['MLP'] - df['ERRO_ARIMA']
df['DIFF_ERRO'] = df['ERRO_SVM'] - df['ERRO_MLP']

for i in range(1,lag):
    df['MA{}_ERRO_ARIMA 1'.format(i)] = df['ERRO_ARIMA'].rolling(i).mean()
    df['MA{}_ERRO_ARIMA 1'.format(i)] = df['MA{}_ERRO_ARIMA 1'.format(i)].shift(1)
    df['MA{}_ERRO_SVM 1'.format(i)] = df['ERRO_SVM'].rolling(i).mean()
    df['MA{}_ERRO_SVM 1'.format(i)] = df['MA{}_ERRO_SVM 1'.format(i)].shift(1)
    df['MA{}_ERRO_MLP 1'.format(i)] = df['ERRO_MLP'].rolling(i).mean()
    df['MA{}_ERRO_MLP 1'.format(i)] = df['MA{}_ERRO_MLP 1'.format(i)].shift(1)
    df['MA{}_DIFF_ERRO 1'.format(i)] = df['DIFF_ERRO'].rolling(i).mean()
    df['MA{}_DIFF_ERRO 1'.format(i)] = df['MA{}_DIFF_ERRO 1'.format(i)].shift(1)

for i in range(1,lag):
    df['ERRO_ARIMA {}'.format(i)] = df['ERRO_ARIMA'].shift(i)
    df['ERRO_SVM {}'.format(i)] = df['ERRO_SVM'].shift(i)
    df['ERRO_MLP {}'.format(i)] = df['ERRO_MLP'].shift(i)
    df['DIFF_ERRO {}'.format(i)] = df['DIFF_ERRO'].shift(i)
    
df = df.dropna()
df = df.reset_index()
    
output = df['VALOR REAL']
df = df.drop(columns=['ERRO_ARIMA', 'ERRO_SVM', 'ERRO_MLP', 'DIFF_ERRO', 'VALOR REAL', 'index'])
#df = df.drop(columns=['VALOR REAL'])

index_test = -14;

dftrain = df[:index_test]
outtrain = output[:index_test]
dftest = df[index_test:]
outtest = output[index_test:]

from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error as mse

est_gp = SymbolicRegressor(population_size=15000,
                           generations=50, stopping_criteria=0.0,
                           const_range=(-1,1), init_depth=(1,10),
                           init_method='half and half',
                           p_crossover=0.9, p_subtree_mutation=0.01,
                           p_hoist_mutation=0.01, p_point_mutation=0.01,
                           p_point_replace=0.05,max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.001, random_state=50,
                           function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg',
                                         'inv','max','min','sin','cos','tan'),
                           metric='mse',n_jobs=-1)
est_gp.fit(dftrain, outtrain)

predictions = est_gp.predict(dftest)

meansq = mse(predictions, outtest)
print("MSE (Test): {}".format(meansq))

