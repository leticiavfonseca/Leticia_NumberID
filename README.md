# Letícia Programação Aplicada
### Professor : Major Renault 
### 2021.2 Eletrônica SE/3



##  Projeto
O projeto foi desenvolvido pensando nas recentes aplicações do aprendizado de máquina e sua crescente na sociedade
contemporânea. Uma possível aplicação seria a identificação de números de residências através de imagens do google street view

## Função
Indicar o valor numérico de uma imagem 28x28 pixels, sendo porta de entrada para mecanismos mais complexos de processamento gráfico como
redes convolucionais.


### O código
O projeto foi desenvolvido a partir de classes da lib TensorFlow, utilizando o conceito de herança. A partir dai, o código realiza um treinamento do modelo sequencial através das imagens de teste, com batches e epochs de tamanho definidos empíricamente após testagem. A rede é treinada por meio do conceito de backpropagation, com fundamentos de cálculo multidimensional para minimização da entropia cruzada.
Após isso, já no arquivo DNN_Run.py, é elaborada uma classe que consolida dois métodos para o usuário final: uma testagem rápida acerca da 
precisão do modelo e um método para o usuário escolher um item do conjunto de validação para acompanhar qual o palpite do algoritmo.
Toda a interface gráfica do programa é feita através do matplotlib, uma lib clássica de python para construção gráfica e elaboração de reports.
O usuário, por definição, se comunica com o programa através do terminal em que roda o programa, em um input.

### Instalação

Foi incluido no projeto um arquivo .txt com todos os requisitos em termos de bibliotecas para rodar a aplicação de forma que o usuário precisa 
apenas ter o python já instalado em sua máquina. Com esse critério satisfeito, basta rodar o arquivo python DNN_Run.py


