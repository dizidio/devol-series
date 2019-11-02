import tensorflow as tf
##from keras.backend.tensorflow_backend import set_session
##config = tf.ConfigProto()
##config.gpu_options.per_process_gpu_memory_fraction = 0.4
##config.gpu_options.visible_device_list = "0"
##set_session(tf.Session(config=config))

from devol.devol import DEvol
from devol.genome_handler import GenomeHandler
import pandas as pd

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

index_test = -52;

dftrain = df[:index_test]
outtrain = output[:index_test]
dftest = df[index_test:]
outtest = output[index_test:]

split_ratio = 0.2
split_index = int((1-split_ratio)*len(dftrain))
dft = dftrain[:split_index]
opt = outtrain[:split_index]
dfv = dftrain[split_index:]
opv = outtrain[split_index:]

dataset = ((dft, opt), (dfv, opv))

genome_handler = GenomeHandler(max_dense_layers=4, # includes final layer
                               max_dense_nodes=128,
                               max_input_size=dft.shape[1])

devol = DEvol(genome_handler)
model = devol.run(dataset=dataset,
                  num_generations=5,
                  pop_size=5,
                  epochs=5,
                  n_tries=5,
                  metric='loss')
print(model.summary())
