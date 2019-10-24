from __future__ import print_function

import time
import numpy as np
#import pandas as pd
import os
import os.path as path
import keras
import struct
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Flatten, Dropout, Reshape, LocallyConnected2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, LSTM
from keras.models import model_from_json
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

import matplotlib.pyplot as plt
#import seaborn as sns

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


import tensorflow as tf

from sklearn.metrics import confusion_matrix, log_loss, auc
from sklearn.model_selection import train_test_split


##def mean_pred(y_true, y_pred):
##    points_test = loc_real.astype('float32');
##    result_test = model.predict(data_lt)
##    points_result = tf.matmul(result_test, locs_pts.astype('float32'));
##    distance_x = tf.square(tf.subtract(points_test[:,0],points_result[:,0]));
##    distance_y = tf.square(tf.subtract(points_test[:,1],points_result[:,1]));
##    distance_z = tf.square(tf.subtract(points_test[:,2],points_result[:,2]));
##    distance_total = distance_x + distance_y + distance_z;
##    return tf.reduce_mean(distance_total);

def get_loc(ponto):
    return locs_apk[locs_apk[:,0]==ponto, 1:]
    
## LIMIT GPU MEMORY
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


#sess = tf.Session()
#K.set_session(sess)

num_rows = 10;#
num_cols = 15;#
min_val = -95;
max_val = -20;
n_bins = 25;#
N_VZS = 30;
epochs = 150;
learn_rate = 0.0001;
    

timestr = time.strftime("%Y%m%d_%H%M%S")
stamp = '_{}_{}rows_{}aps_{}bins'.format(timestr, num_rows, num_cols, n_bins);

MODEL_NAME = 'DNN'+stamp;#

############################
####### TRAIN CONFIG #######
batch_size = 64;
num_classes = 71;

def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out'+stamp, \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out{}/'.format(stamp) + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out{}/'.format(stamp) + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out{}/'.format(stamp) + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out{}/frozen_'.format(stamp) + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out{}/frozen_'.format(stamp) + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out{}/opt_'.format(stamp) + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")
    
############################
##### PRE-PROCESSING  ######

filename_train = 'hists_train_150518_15aps_25bins_10rows_diogo.csv' #
filename_test = 'hists_new_test_050718_15aps_25bins_10rows.txt' #

locs_apk = np.genfromtxt('coords_app_020818.csv',delimiter=",");
data_final = np.genfromtxt(filename_train,delimiter=",");
data_l = data_final[:,:-1];
classes_l = data_final[:,-1];

cont = 0;
locs_l = np.empty([classes_l.shape[0], 3]);
for i in classes_l:
    locs_l[cont] = get_loc(i);
    cont = cont + 1;
    

data_l = data_l/num_rows;
            
data_train, class_train = data_l,classes_l;

timestr = time.strftime("%Y%m%d_%H%M%S")

data_train = data_train.reshape(data_train.shape[0], num_cols, n_bins, 1);
input_shape = (num_cols, n_bins,1);

class_train = class_train - 1;
class_train = keras.utils.to_categorical(class_train, num_classes);

data_final_test = np.genfromtxt(filename_test,delimiter=",");#
data_lt = data_final_test[:,:-1];

classes_lt = data_final_test[:,-1];

cont = 0;
loc_real = np.empty([classes_lt.shape[0], 3]);
for i in classes_lt:
    loc_real[cont] = get_loc(i);
    cont = cont + 1;

#loc_real = data_final_test[:,-3:];
data_lt = data_lt/num_rows;
data_lt = data_lt.reshape(data_lt.shape[0], num_cols, n_bins, 1);

cont = 0;
locs_pts = np.empty([71, 3]);
for i in range(71):
    locs_pts[cont] = get_loc(i+1);
    cont = cont + 1;
    

os.mkdir('./out'+stamp)

#############################
########## MODEL ############

class calc_dist_test(keras.callbacks.Callback):

  def __init__(self, patience):
      self.patience = patience;
      self.history_auc = [];
      self.history_dist = [];
      self.history_weights = [];
      self.cont_distance = 0;
          
  def on_epoch_end(self, batch, logs={}):
      score_results = self.model.predict(data_lt)
      points_test = loc_real
      points_result = np.dot(score_results, locs_pts)
      distance = np.sqrt((points_test[:,0]-points_result[:,0])**2 + (points_test[:,1]-points_result[:,1])**2 + (points_test[:,2]-points_result[:,2])**2);
      
      sorted_ = np.sort(distance)
      yvals = np.arange(len(sorted_))/float(len(sorted_))
      sorted_ = np.append(sorted_, 100);
      yvals = np.append(yvals, 1);
      
      auc_value = auc(sorted_, yvals);
      print("Dist. Media: ", np.mean(distance));
      print("AUC (test): ", auc_value);
      
      if (self.history_auc and (auc_value <= max(self.history_auc))):
          self.cont_distance = self.cont_distance + 1;
      else:
          self.cont_distance = 0;

      self.history_auc.append(auc_value);
      self.history_dist.append(distance);
      self.history_weights.append(self.model.get_weights())
      
      index = np.argmax(self.history_auc);
      self.best_distance = self.history_dist[index];
      self.best_weights = self.history_weights[index];
     
      if (self.cont_distance>self.patience):
          self.model.stop_training = True
          print('Early Stopping');
#c_test = calc_dist_test(patience=5)
best_distance = 200000;

#model = load_model('best-model.h5');

#model.add(Dense(128, activation='tanh', kernel_initializer = 'he_normal'));
#model.add(Dropout(rate=0.2))

for ik in range(N_VZS):
    cback_dist = calc_dist_test(patience=5);
    print("\n Instance {}/{}".format(ik+1, N_VZS));
    model = Sequential()
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(32, activation='tanh', kernel_initializer = 'he_normal'));
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.6))
    #model.add(Dense(256, activation='relu', kernel_initializer = 'he_normal'));
    #model.add(BatchNormalization())
    model.add(Dense(128, activation='tanh', kernel_initializer = 'he_normal'));
    model.add(Dropout(rate=0.2))
    model.add(Dense(512, activation='relu', kernel_initializer = 'he_normal'));
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer = 'he_normal'))

    model.summary()

    #learn_rate = 0.5;
    #optimizer = keras.optimizers.Adadelta()
    optimizer = keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, decay=0.0)
    model.compile(loss=keras.losses.poisson,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(data_train, class_train,
                  batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[cback_dist])
                  
    max_auc = max(cback_dist.history_auc);
    print("AUC: {}".format(max_auc));

    #score_results = model.predict(data_lt)
    #points_test = loc_real
    #points_result = np.dot(score_results, locs_pts)
    #distance = np.sqrt((points_test[:,0]-points_result[:,0])**2 + (points_test[:,1]-points_result[:,1])**2 + (points_test[:,2]-points_result[:,2])**2 );
    print("Model Distance: {}".format(np.mean(cback_dist.best_distance)));

    if np.mean(cback_dist.best_distance) < best_distance:
        best_distance = np.mean(cback_dist.best_distance);
        model.set_weights(cback_dist.best_weights);
        best_model = model;        
        error_distance = cback_dist.best_distance;
        #best_history = history;
        best_index = ik;
        try:
          os.remove('./out'+stamp+'/best-model.h5')
        except OSError:
          pass
        model.save('./out'+stamp+'/best-model.h5')


np.savetxt("./out"+stamp+"/error_DNN"+stamp+".txt", error_distance, '%5.2f', delimiter = ",")

print("Best Distance = {}".format(best_distance))

config_training = best_model.get_config() + [type(optimizer)] + [optimizer.get_config()] + \
                  ["learn_rate = {}; epochs = {}; n_bins = {}; num_rows = {}; num_cols = {}; batch_size = {}; N_VZS = {}".format(learn_rate, epochs, n_bins, num_rows, num_cols, batch_size, N_VZS)] + \
                  [filename_train] + [filename_test] + ["Best Distance = {}".format(best_distance)] + ["Best Index = {}".format(best_index)];

np.savetxt("./out"+stamp+"/model_config"+stamp+".txt", config_training, '%s')


plt.figure()
sorted_ = np.sort(error_distance)
yvals = np.arange(len(sorted_))/float(len(sorted_))
plt.title('DNN'+stamp)
plt.ylabel('Probabilidade Acumulada')
plt.xlabel('Erro (m)')

plt.plot(sorted_, yvals)
plt.savefig('./out'+stamp+'/cumulated_distance_test'+stamp+'.pdf'.format(timestr, num_rows, num_cols, n_bins))

