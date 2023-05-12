import sys
import os
import pandas as pd
import datetime
import shutil
import time
import io
import glob
import matplotlib.pyplot as plt
from   sklearn.model_selection       import train_test_split

import tensorflow                    as     tf
from   tensorflow                    import keras
from   tensorflow.keras              import losses
from   tensorflow.keras              import metrics
from   tensorflow.keras.callbacks    import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from   tensorflow.keras.initializers import GlorotNormal, GlorotUniform, HeNormal
from   tensorflow.keras.layers       import Dense, Dropout, Input
from   tensorflow.keras.models       import load_model, Sequential
from   tensorflow.keras.optimizers   import Adam


def model_l():
    initializer = HeNormal()
    optimizer = Adam(learning_rate=0.0001)

    model = tf.keras.Sequential([
        Input(shape=(5,)), #tmn entrada
        Dense(400,activation='relu',kernel_initializer=initializer),
        Dense(500,activation='relu',kernel_initializer=initializer),
        Dense(400,activation='relu',kernel_initializer=initializer),
        Dense(4, activation='linear')
        ])

    model.compile(optimizer=optimizer,loss=losses.MAPE,metrics=[metrics.MAPE])
    return model

def model_mo():
    initializer = HeNormal()
    optimizer = Adam(learning_rate=0.0001)

    inputLayer = Input(shape=(5,))
    firstDensec0 = Dense(200,activation='relu', kernel_initializer=initializer)(inputLayer)
    firstDensec1 = Dense(200,activation='relu', kernel_initializer=initializer)(inputLayer)
    firstDensec2 = Dense(200,activation='relu', kernel_initializer=initializer)(inputLayer)
    firstDensec3 = Dense(200,activation='relu', kernel_initializer=initializer)(inputLayer)

    secondDensec0 = Dense(300,activation='relu', kernel_initializer=initializer)(firstDensec0)
    secondDensec1 = Dense(300,activation='relu', kernel_initializer=initializer)(firstDensec1)
    secondDensec2 = Dense(300,activation='relu', kernel_initializer=initializer)(firstDensec2)
    secondDensec3 = Dense(300,activation='relu', kernel_initializer=initializer)(firstDensec3)

    thirdDensec0 = Dense(200,activation='relu', kernel_initializer=initializer)(secondDensec0)
    thirdDensec1 = Dense(200,activation='relu', kernel_initializer=initializer)(secondDensec1)
    thirdDensec2 = Dense(200,activation='relu', kernel_initializer=initializer)(secondDensec2)
    thirdDensec3 = Dense(200,activation='relu', kernel_initializer=initializer)(secondDensec3)

    c0Output = Dense(1,activation='linear', name='c0')(thirdDensec0)
    c1Output = Dense(1,activation='linear', name='c1')(thirdDensec1)
    c2Output = Dense(1,activation='linear', name='c2')(thirdDensec2)
    c3Output = Dense(1,activation='linear', name='c3')(thirdDensec3)

    model = tf.keras.Model(inputs=inputLayer, outputs=[c0Output, c1Output, c2Output, c3Output])
    model.compile(optimizer=optimizer, 
                  loss={'c0': losses.MAPE, 'c1': losses.MAPE, 'c2': losses.MAPE, 'c3': losses.MAPE},
                  metrics={'c0': metrics.MAPE, 'c1': metrics.MAPE, 'c2': metrics.MAPE, 'c3': metrics.MAPE})
    
    return model

def model_b():
    initializer = HeNormal()
    optimizer = Adam(learning_rate=0.0001)

    inputLayer = Input(shape=(5,))
    dense0 = Dense(300,activation='relu',kernel_initializer=initializer)(inputLayer)
    dense1 = Dense(375,activation='relu',kernel_initializer=initializer)(dense0)
    dense2 = Dense(300,activation='relu',kernel_initializer=initializer)(dense1)
    output = Dense(144, activation='sigmoid')(dense2)

    model = tf.keras.Model(inputs=inputLayer, outputs=[output])

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy', metrics.binary_crossentropy, keras.metrics.Precision(), keras.metrics.Recall()])

    return model

# Generator necessário para treinamento em memória externa, caso o dataset não couber na memória principal
def dataset_generator(filename, batch_size, n_entries=7):
    csvfile = open(filename)
    # chunksize define a quantidade de linhas do dataset que serão lidos em conjunto
    # chunksize = batch_size -> lê um lote do disco por vez, pode ser lido mais de um lote, mas precisa de mais um laço, 1 yield por batch
    reader = pd.read_csv(csvfile, chunksize=batch_size, header=None, dtype="float64")
    while True:
        for chunk in reader:
            w = chunk.values
            x = w[:, :n_entries]    # Primeiras 'n_entries' colunas -> entrada da rede
            y = w[:, n_entries:]     # Ultima coluna -> saída desejada
            yield x, y
        csvfile = open(filename)
        reader = pd.read_csv(csvfile, chunksize=batch_size, header=None, dtype="float64")


# Treina um modelo de dnn
# model: nome do modelo
# n_batch: tamanho do lote
# dataset: diretório contendo os arquivos csv de treinamento, teste e validação
# n_epochs_max: quantidade limite de épocas do treinamento
# patience: tolerância de epocas sem melhora no erro, se excedido
def train_dnn(model, n_batch, dataset, n_epochs_max=2000, patience=100, dir_log=''):
    model_name = 'model' + '_' + str(n_batch) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = dir_log + model_name

    model_weights_file_name = dir_name + '_weights.hdf5'
    model_save_file_name = dir_name + '_model.h5'

    model = model()

    train_set = dataset + '/train_set.csv'
    validation_set = dataset + '/validation_set.csv'
    test_set = dataset + '/test_set.csv'

    # Callback de parada antecipada, para a quantidade de épocas passada no parâmetro 'patience'
    early_stop = EarlyStopping(
        monitor='val_binary_crossentropy',
        mode='min',
        verbose=1,
        patience=patience,
        restore_best_weights=True)

    # Callback para checkpoint do melhor modelo, baseado no erro MAPE
    checkpoint_min_loss = ModelCheckpoint(
        filepath=dir_name + '_checkpoint_{epoch}_{loss:.4f}.hdf5',
        monitor='val_binary_crossentropy',
        verbose=1,
        save_best_only=True,
        mode='auto')

    callbacks_list = [early_stop, checkpoint_min_loss]

    entrySize = 5

    # Cria os geradores de dados para treinamento, validação e teste
    train_generator = dataset_generator(train_set, n_batch, entrySize)
    validation_generator = dataset_generator(validation_set, n_batch, entrySize)
    test_generator = dataset_generator(test_set, n_batch, entrySize)

    # ---- Cálculo da quantidade de lotes de treinamento, teste e validação -------- #
    # Alterar manualmente se necessário

    map_size = 300 ** 2 # tamanho do mapa
    sampling = 0.10     # amostragem por instância do mapa
    num_maps = 1      # quantidade de instâncias
    # dataset_size = ((map_size * sampling) * (map_size * sampling -1) * num_maps) / 2
    dataset_size = 166176

    # Quantidade de 'leituras' realizadas para passar pelo arquivo inteiro (1 por lote)
    ntrain = (dataset_size * 0.7) // n_batch
    nval = (dataset_size * 0.15) // n_batch
    ntest = (dataset_size * 0.15) // n_batch
    # ------------------------------------------------------------------------------- #

    # Realiza o treinamento
    history = model.fit(
        train_generator,
        steps_per_epoch=ntrain,
        epochs=n_epochs_max,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=nval,
        verbose=2
    )

    # Salva o modelo após treinamento
    model.save(model_save_file_name)
    model.save_weights(model_weights_file_name)

    score = model.evaluate(test_generator, steps=ntest, verbose=1)

    csvfile = open(test_set)
    reader = pd.read_csv(csvfile, nrows=10, header=None, dtype="float64")

    for row in reader:
        print(row)

    results = model.predict(row[:5])

    for result in results:
        print(result)

    return score, history


# grava o conteudo de data_io num csv
def grava(filename,data_io):
    # grava os resultados num csv para analise posterior
    with open(filename, 'a') as file:
        data_io.seek(0)
        shutil.copyfileobj(data_io, file)

def main():
    args = sys.argv

    out_dir = "D:\Juliano Pasa\Pesquisa\codigo-do-cristian-theta-\\results"
    dataset_location = "D:\Juliano Pasa\Pesquisa\codigo-do-cristian-theta-\dataset\\binarioMaior"

    data_io_stats = io.StringIO()
    stats_file_name = out_dir + '/results.csv'

    # Verifica se a GPU está ativada pro treinamento
    print(tf.config.list_physical_devices())
    models = [model_b] # Lista de modelos, 1 modelo sendo treinado

    # Tamanho do lote de treinamento
    batch_size = 32

    print('Treinamento iniciado')
    start = time.time()
    try:
        for model in models:
            model().summary()

            t1_start = time.time()
            # Realiza o treinamento do modelo
            score, history = train_dnn(model, batch_size, dataset_location, n_epochs_max=100, patience=30, dir_log=out_dir+'\\')
            print(history.history.keys())
            t1_stop = time.time()
            diff_time = t1_stop - t1_start

            data_io_stats.write("""%s,%s,%s,%s,%s\n""" % ('model',batch_size,history.history,diff_time,score))


            # Plotando acerto
            plt.plot(history.history['val_binary_crossentropy'])
            plt.plot(history.history['val_binary_crossentropy'])
            plt.title("Mean absolute percentage error of the Model")
            plt.ylabel("Mean absolute percentage error")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Val"], loc='upper left')
            plt.show()
            
            # Plotando perda
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title("Loss of the Model")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Val"], loc='upper left')
            plt.show()
            print(score)

    finally:
        total_training_time = time.time() - start
        print("Treinamento finalizado! Tempo total: " + str(total_training_time))
        # Grava os resultados num csv
        grava(stats_file_name, data_io_stats)

def normalizeDatasets(path):
    df = pd.read_csv(path, header=None)

    for i in range(5, 9):
        df.iloc[:, i] = (df.iloc[:, i] - df.iloc[:, i].mean())/df.iloc[:, i].std()

    outputName = path[:-4] + "_output.csv"
    df.to_csv(outputName, header=None, index=None)

    print(df.head()) 


if __name__ == '__main__':
    #normalizeDatasets("D:\Importantes\Projetos\Artigo revista\codigo-do-cristian-theta-\dataset\cluster\\validation_set.csv")
    main()
    #tfenv2
    #python .\