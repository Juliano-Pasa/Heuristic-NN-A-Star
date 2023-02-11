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
        Input(shape=(7,)), #tmn entrada
        Dense(400,activation='relu',kernel_initializer=initializer),
        Dense(500,activation='relu',kernel_initializer=initializer),
        Dense(400,activation='relu',kernel_initializer=initializer),
        Dense(1, activation='linear')
        ])

    model.compile(optimizer=optimizer,loss=losses.MAPE,metrics=[metrics.MAPE])
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
            y = w[:, n_entries]     # Ultima coluna -> saída desejada
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
        monitor='val_mean_absolute_percentage_error',
        mode='min',
        verbose=1,
        patience=patience,
        restore_best_weights=True)

    # Callback para checkpoint do melhor modelo, baseado no erro MAPE
    checkpoint_min_loss = ModelCheckpoint(
        filepath=dir_name + '_checkpoint_{epoch}_{loss:.4f}.hdf5',
        monitor='mean_absolute_percentage_error',
        verbose=1,
        save_best_only=True,
        mode='auto')

    callbacks_list = [early_stop, checkpoint_min_loss]

    # Cria os geradores de dados para treinamento, validação e teste
    train_generator = dataset_generator(train_set, n_batch)
    validation_generator = dataset_generator(validation_set, n_batch)
    test_generator = dataset_generator(test_set, n_batch)

    # ---- Cálculo da quantidade de lotes de treinamento, teste e validação -------- #
    # Alterar manualmente se necessário

    map_size = 300 ** 2 # tamanho do mapa
    sampling = 0.10     # amostragem por instância do mapa
    num_maps = 1      # quantidade de instâncias
    dataset_size = ((map_size * sampling) * (map_size * sampling -1) * num_maps) / 2

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

    score = model.evaluate(test_generator, steps=ntest, verbose=0)

    return score, history


# grava o conteudo de data_io num csv
def grava(filename,data_io):
    # grava os resultados num csv para analise posterior
    with open(filename, 'a') as file:
        data_io.seek(0)
        shutil.copyfileobj(data_io, file)

def main():
    args = sys.argv

    out_dir = args[1]   # Diretório de saída
    dataset_location = args[2]  # Diretório contendo train_set.csv, test_set.csv e validation_set.csv

    # os.chdir(dataset_location)

    data_io_stats = io.StringIO()
    stats_file_name = out_dir + '.csv'

    # Verifica se a GPU está ativada pro treinamento
    print(tf.config.list_physical_devices())
    models = [model_l] # Lista de modelos, 1 modelo sendo treinado

    # Tamanho do lote de treinamento
    batch_size = 64 * 4096

    print('Treinamento iniciado')
    start = time.time()
    try:
        for model in models:
            model().summary()

            t1_start = time.time()
            # Realiza o treinamento do modelo
            score, history = train_dnn(model, batch_size, dataset_location, n_epochs_max=100, patience=30, dir_log=out_dir+'/')
            t1_stop = time.time()
            diff_time = t1_stop - t1_start

            # Plotando acerto
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title("Accurracy of the Model")
            plt.ylabel("Accuracy")
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

            data_io_stats.write("""%s,%s,%s,%s,%s\n""" % ('model',batch_size,history.history,diff_time,score))

            print(score)
    finally:
        total_training_time = time.time() - start
        print("Treinamento finalizado! Tempo total: " + str(total_training_time))
        # Grava os resultados num csv
        grava(stats_file_name, data_io_stats)


if __name__ == '__main__':
    main()