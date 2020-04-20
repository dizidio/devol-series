import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
import time
import csv

#genome = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 64, 0, 2048, 1, 256, 1, 32, 'relu', 'adam'] #out_svm
genome = [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 256, 0, 4, 0, 1, 0, 256, 'relu', 'adam'] #out_mlp

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def load_data(filename, index_test, lag_max, split_ratio=0.2):
    df = pd.read_csv(filename)
    print(df)

    for i in range(1,lag_max):
        df['SARIMA_{}'.format(i)] = df['SARIMA'].shift(i)
        df['out_mlp_{}'.format(i)] = df['out_mlp'].shift(i)
        df['out_svm_{}'.format(i)] = df['out_svm'].shift(i)

    df = df.dropna()
    df = df.reset_index()
    weekday = df['Weekday']

    df = df.drop(columns=[c for c in df if c.startswith('out_mlp')]) ## MUDAR

    output = df['TARGET']
    df = df.drop(columns=['Unnamed: 0', 'TARGET', 'SPLIT', 'index', 'Weekday', 'Hour'])

    dftrain = df.iloc[:index_test]
    outtrain = output.iloc[:index_test]
    dftest = df.iloc[index_test:]
    outtest = output.iloc[index_test:]
    weektest = weekday.iloc[index_test:]

    split_index = int((1-split_ratio)*len(dftrain))
    dft = dftrain.iloc[:split_index]
    opt = outtrain.iloc[:split_index]
    dfv = dftrain.iloc[split_index:]
    opv = outtrain.iloc[split_index:]

    return ((dft, opt), (dfv, opv)), (dftest, outtest), weektest

max_dense_nodes = 2048;
max_layers = 4;
pop_size = 100;
n_tries = 3;
nGens = 50;
crossoverRatio = 0.5;
lag_max = 10;
split_ratio = 0.2

input_active = [0, 1]
activation = ['identity', 'logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
layer_active = [0, 1]
layer_nodes = [2**i for i in range(0, int(np.log2(max_dense_nodes))+1)]


def get_model(genome):
    layers = []
    for i in range(input_size, input_size+max_layers*2, 2):
        layers.append(genome[i]*genome[i+1])
    layers = tuple([l for l in layers if l!=0])
    return MLPRegressor(hidden_layer_sizes=layers, activation=genome[-2], solver=genome[-1], max_iter = 1500, learning_rate='adaptive')

f = "./combined_arima_svm_mlp_weekday.csv"
dataset, testdataset, weektest = load_data(f, lag_max=lag_max, index_test = -670, split_ratio=split_ratio)
print("Carregando arquivo: {}".format(f))

X_train = dataset[0][0]
y_train = dataset[0][1]
X_val = dataset[1][0]
y_val = dataset[1][1]
X_test = testdataset[0]
y_test = testdataset[1]
input_size = X_train.shape[1]


lmse = []
ltest_mse = []
lmae = []
ltest_mae = []
lmape = []
ltest_mape = []
lr2 = []
ltest_r2 = []

for _ in range(100):
    model = get_model(genome)
    model.fit(X_train.iloc[:,genome[:input_size]], y_train.values.ravel())
    
    lmse.append(mse(y_val.values.ravel(),model.predict(X_val.iloc[:,genome[:input_size]])))
    ltest_mse.append(mse(y_test.values.ravel(), model.predict(X_test.iloc[:,genome[:input_size]])))
    
    lmae.append(mae(y_val.values.ravel(),model.predict(X_val.iloc[:,genome[:input_size]])))
    ltest_mae.append(mae(y_test.values.ravel(), model.predict(X_test.iloc[:,genome[:input_size]])))
    
    lmape.append(mape(y_val.values.ravel(),model.predict(X_val.iloc[:,genome[:input_size]])))
    ltest_mape.append(mape(y_test.values.ravel(), model.predict(X_test.iloc[:,genome[:input_size]])))
    
    lr2.append(r2_score(y_val.values.ravel(),model.predict(X_val.iloc[:,genome[:input_size]])))
    ltest_r2.append(r2_score(y_test.values.ravel(), model.predict(X_test.iloc[:,genome[:input_size]])))

    if lmse[-1] <= min(lmse):
        best_model = model
    
argmin = np.argmin(lmse)
print('MSE: {} - MAE: {} - MAPE: {} - R2: {}'.format(ltest_mse[argmin], ltest_mae[argmin], ltest_mape[argmin], ltest_r2[argmin]))
#print("Training: {} - Testing: {}".format(mse(y_val.values.ravel(),model.predict(X_val.iloc[:,genome[:input_size]])), mse(y_test.values.ravel(), model.predict(X_test.iloc[:,genome[:input_size]]))))

for i in range(7):
    print("Weekday: {}".format(i))
    int_ytest = y_test[weektest==i]
    int_Xtest = X_test[weektest==i]
    print("{}".format(mse(int_ytest.values.ravel(), best_model.predict(int_Xtest.iloc[:,genome[:input_size]]))))
    print("{}".format(mae(int_ytest.values.ravel(), best_model.predict(int_Xtest.iloc[:,genome[:input_size]]))))
    print("{}".format(mape(int_ytest.values.ravel(), best_model.predict(int_Xtest.iloc[:,genome[:input_size]]))))
    print("{}".format(r2_score(int_ytest.values.ravel(), best_model.predict(int_Xtest.iloc[:,genome[:input_size]]))))
    
