ARQUIVOS:

1_generate_dataset.py:
Código para realizar a geração do dataset de treinamento. Utiliza o código auxiliar de ./AUXILIARES/generate_viewshed.py para geração dos mapas de visibilidade
salvos no diretório ./VIEWSHEDS
Comando para rodar:
$ python 1_generate_dataset.py <arquivo .tif do recorte do terreno>

2_shuffle_split_dataset.py:
Código para realizar a preparação do dataset: embaralha os dados e separa em conjuntos de treinamento, validação e teste
Comando:
$ python 2_shuffle_split_dataset.py <arquivo .csv do dataset gerado>

3_train.py:
Código para treinamento de um modelo de DNN
Comando:
$ python 3_train.py <diretório de saída> <caminho para o diretório dos conjuntos de treinamento/teste/validação>

4_test.py:
Código para realização dos testes do(s) modelo(s) de DNN treinado(s)
Comando:
$python 4_test.py <.tif do recorte do terreno> <DNN modelo 1> <DNN modelo 2>

Outros códigos auxiliares:
./AUXILIARES/cut_mde.py:
Produz um recorte de um terreno maior

./AUXILIARES/view.py
Código para visualização do terreno, nodos abertos, caminho percorrido e projeção da visibilidade do observador

./AUXILIARES/generate_viewshed.py
Código utilizado na geração do dataset em 1_generate_dataset.py