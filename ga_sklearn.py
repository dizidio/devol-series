import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse
import time
import csv

def load_data(filename, index_test, lag_max, split_ratio=0.2):
    df = pd.read_excel(filename)
    
    df['VALOR REAL'] = df.TARGET - df.SARIMA
    df = df[['VALOR REAL']]

    for i in range(1,lag_max):
        df['VALOR REAL {}'.format(i)] = df['VALOR REAL'].shift(i)

    df = df.dropna()
    df = df.reset_index()

    output = df['VALOR REAL']
    #df = df.drop(columns=['ERRO_ARIMA', 'ERRO_SVM', 'ERRO_MLP', 'DIFF_ERRO', 'VALOR REAL', 'index'])
    df = df.drop(columns=['VALOR REAL', 'index'])

    dftrain = df.iloc[:index_test]
    outtrain = output.iloc[:index_test]
    dftest = df.iloc[index_test:]
    outtest = output.iloc[index_test:]

    split_index = int((1-split_ratio)*len(dftrain))
    dft = dftrain.iloc[:split_index]
    opt = outtrain.iloc[:split_index]
    dfv = dftrain.iloc[split_index:]
    opv = outtrain.iloc[split_index:]

    return ((dft, opt), (dfv, opv)), (dftest, outtest)   


class Population:
    def __init__(self, pop_size, max_layers, n_tries, dataset):
        self.pop = []
        self.n_tries = n_tries
        self.pop_size = pop_size
        self.max_layers = max_layers
        self.X_train = dataset[0][0]
        self.y_train = dataset[0][1]
        self.X_val = dataset[1][0]
        self.y_val = dataset[1][1]
        self.input_size = self.X_train.shape[1]
        self.error_list = []
        self.gen = 1
        for _ in range(pop_size):
            ind = []
            for _ in range(self.input_size):
                ind.append(np.random.choice(input_active))
            for _ in range(self.max_layers):
                ind.append(np.random.choice(layer_active))
                ind.append(np.random.choice(layer_nodes))
            ind.append(np.random.choice(activation))
            ind.append(np.random.choice(solver))
            self.pop.append(ind)
            
    def get_model(self,genome):
        layers = []
        for i in range(self.input_size, self.input_size+self.max_layers*2, 2):
            layers.append(genome[i]*genome[i+1])
        layers = tuple([l for l in layers if l!=0])
        return MLPRegressor(hidden_layer_sizes=layers, activation=genome[-2], solver=genome[-1], max_iter = 1500, learning_rate='adaptive')

    def evaluate(self,genome):
        error_list = []
        for _ in range(self.n_tries): 
            mlp = self.get_model(genome)
            mlp.fit(self.X_train.iloc[:,genome[:self.input_size]], self.y_train.values.ravel())
            error_list.append(mse(self.y_val.values.ravel(), mlp.predict(self.X_val.iloc[:,genome[:self.input_size]])))
        print(np.min(error_list))
        return np.min(error_list)

    def evaluate_pop(self):
        error_list = []
        for ind in range(self.pop_size):
            error_list.append(self.evaluate(self.pop[ind]))
        self.gen_avg = np.mean(error_list)
        self.gen_std = np.std(error_list)
        self.best_ind = np.min(error_list)
        self.error_list = error_list
        return error_list

    def mutate(self, genome):
        index = np.random.randint(len(genome))
        if index < self.input_size:
            genome[index] = 1 - genome[index]
        elif index < self.input_size + self.max_layers*2:
            if (index - self.input_size) % 2 == 0:
                genome[index] = 1 - genome[index]
            else:
                genome[index] = np.random.choice(layer_nodes)
        elif index == len(genome)-2:
            genome[index] = np.random.choice(activation)
        elif index == len(genome)-1:
            genome[index] = np.random.choice(solver)
        return genome

    def crossover(self, genome1, genome2):
        cross_ind = np.random.randint(0, len(genome1))
        child = genome1[:cross_ind] + genome2[cross_ind:]
        return child

    def update(self):
        newPop = []

        max_mutations = max(3, self.gen // 4)
        
        inverse_error_list = [ 1/i for i in self.error_list]
        probabilities = inverse_error_list / np.sum(inverse_error_list)
        
        for _ in range(self.pop_size):

            if np.random.random() <= crossoverRatio:
                ind1, ind2 = np.random.choice(len(self.pop), size=2, p = probabilities, replace=False)
                newPop.append(self.crossover(self.pop[ind1],self.pop[ind2]))
            else:
                ind = np.random.choice(len(self.pop), p = probabilities)
                newPop.append(self.pop[ind])

            num_mutations = np.random.choice(max_mutations)
            
            for _ in range(num_mutations):
                newPop[-1] = self.mutate(newPop[-1])
                
        self.pop = newPop
        self.gen = self.gen+1

        return newPop

    def get_avg(self):
        return self.gen_avg

    def get_std(self):
        return self.gen_std

    def get_best(self):
        return self.best_ind
    
    def get_pop(self):
        return self.pop

    def get_best_model(self):
        return self.pop[np.argmin(self.error_list)]

    def get_error_list(self):
        return self.error_list

# CONFIGS

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

# EXECUTION
f = "./data/PREVS_SMART.xlsx"


filename = 'Log_' + time.strftime("%Y%m%d_%H%M%S") + '.txt'
with open(filename, 'w', newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['max_dense_nodes={}'.format(max_dense_nodes), 'max_layers={}'.format(max_layers), 'pop_size={}'.format(pop_size), 'n_tries={}'.format(n_tries),
                    'nGens={}'.format(nGens), 'crossoverRatio={}'.format(crossoverRatio), 'lag_max={}'.format(lag_max), 'split_ratio={}'.format(split_ratio)])
    writer.writerow(['File', 'Generation', 'Individual', 'Genome', 'MSE (Fitness)'])


dataset, testdataset = load_data(f, lag_max=lag_max, index_test = -670, split_ratio=split_ratio)
print("Carregando arquivo: {}".format(f))
best_score = np.inf
gen = Population(pop_size, max_layers, n_tries, dataset)
for g in range(nGens):
    start_time = time.time()
    gen.evaluate_pop()
    print("Generation {} - Avg : {:.4f} - Std : {:.4f} - Best: {:.4f}".format(g+1, gen.get_avg(), gen.get_std(), gen.get_best()))
    print("--- %s seconds ---" % (time.time() - start_time))
    if gen.get_best() < best_score:
        best_ind = gen.get_best_model()

    with open(filename, 'a', newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i, d in enumerate(zip(gen.get_pop(), gen.get_error_list())):
            writer.writerow([f, g+1, i+1, d[0], d[1]])
        writer.writerow(['------ Generation {} '.format(g+1), 'Avg: {} '.format(gen.get_avg()), 'Std: {} '.format(gen.get_std()),'Best: {} ------'.format(gen.get_best())])

    gen.update()
