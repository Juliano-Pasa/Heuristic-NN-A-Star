import random
import io
import sys
import os
from tqdm import tqdm
import csv
import glob
from time import process_time

# Código para embaralhar uma quantidade grande de dados que não cabem em memória
# Fonte: https://stackoverflow.com/a/62566435
def shuffle(file_in, file_out='out'):
    file_out = './out/' + file_out
    files_out = []

    NUM_OF_FILES = 2_00 # Quantidade de fragmentos do arquivo .csv original

    for i in range(NUM_OF_FILES):
        f_ = file_out + str(i) + '.csv'
        files_out.append(io.open(f_, 'w', encoding='utf-8'))

    with io.open(file_in, 'r', encoding='utf-8') as source:
        for f in tqdm(source):
            files_out[random.randint(0, NUM_OF_FILES - 1)].write(f)
        for i in range(NUM_OF_FILES):
            files_out[i].close()

    for i in range(NUM_OF_FILES):
        f_ = file_out + str(i) + '.csv'
        data = []
        with io.open(f_, 'r', encoding='utf-8') as file:
            data = [(random.random(), line) for line in tqdm(file)]
        data.sort()
        with io.open(f_, 'w', encoding='utf-8') as file:
            for _, line in tqdm(data):
                file.write(line)

# Junta os N .csv da etapa de shuffle em 3 arquivos de treinamento, teste e validação
# Params: folder = diretório onde o dataset embaralhado segmentado está localizados
#          train, test e val = % de divisão do dataset (70/15/15)
#          Se NUM_OF_FILES = 100 -> 70 arquivos são unidos em um train_set.csv, 15 arquivos unidos em test_set.csv e 15 arquivos unidos em validation_set.csv
def merge_train_test_validation(folder, train=0.7, test=0.20, val=0.10):
    isExist = os.path.exists("./dataset/")
    if not isExist:
        os.makedirs("./dataset/")

    extension = 'csv'
    all_filenames = [i for i in glob.glob('./'+folder+'/*.{}'.format(extension))]
    train_size = len(all_filenames) * train
    test_size = train_size + len(all_filenames) * test
    count = 0
    for file in all_filenames:
        start = process_time()
        print(file)
        with open(file, 'r') as f:
            f_csv = csv.reader(f)
            if count < train_size:
                out_file = './dataset/train_set.csv'
            elif count < test_size:
                out_file = './dataset/test_set.csv'
            else:
                out_file = './dataset/validation_set.csv'
            with open(out_file, 'a', newline='') as out:
                writer = csv.writer(out)
                for row in f_csv:
                    if row:
                        writer.writerow(row)
        print(process_time() - start)
        count = count + 1

def main():
    args = sys.argv
    f_in = args[1]  # .csv contendo os dados do dataset

    isExist = os.path.exists("./out/")
    if not isExist:
        os.makedirs("./out/")

    shuffle(f_in)   # Embaralha o dataset e salva em N arquivos separados em direstório "./out/" (NUM_OF_FILES)

    os.remove(f_in) # Remove o arquivo original (opcional, depende da disponibilidade de espaço no disco)

    merge_train_test_validation('out')  # Combina as partições embaralhadas criadas em arquivos de treinamento, teste e validação

    # Remove as partições criadas
    files = glob.glob('./out/*.csv')
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    os.rmdir('out')

if __name__ == "__main__":
    main()