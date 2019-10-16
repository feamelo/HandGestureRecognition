# HandGestureRecognition
Image processing algorithm based on Meenakshi Panwar paper

[Este projeto foi desenvolvido em outra plataforma de controle de versionamento e exportada para o GitHub após o término da disciplina Introdução ao Processamento de Imagens - UnB]

Extração de features da mão utilizando técnicas de processamento de imagem para detecção de gestos. O algoritmo segue a seguinte estrutura:

![fluxogram](images/fluxogram.png?raw=true)

Input Image: Conjunto de imagens localizadas na pasta "dataset"
## Pré-processamento e segmentação
Conversão da matriz para o espaço de cores YCbCr e clusterização por Kmeans, dividindo a imagem em duas categorias - fundo e mão.

![kmeans](images/kmeans.png?raw=true)

## Determinação da orientação
1. Varredura da imagem procurando o primeiro pixel de cada extremidade 
2. Delimitação da região da mão através da união das 4 coordenadas encontradas no passo anterior
3. Cálculo do aspect ratio do retângulo encontrado (ratio >1 significa que a mão está na vertical)

![ratio](images/ratio.png?raw=true)

## Extração das features
  
1. Centroide: Calculado a partir do momento da imagem
   
    ![eq1](images/eq1.gif?raw=true)

    ![eq2](images/eq2.gif?raw=true)


2. Detecção do dedão: Uma faixa lateral nas duas bordas da imagem é avaliada. Se menos de 10% da faixa avaliada for da label mão, significa que existe um dedão neste canto da imagem.

    ![thumb](images/thumb.png?raw=true)

3. Detecção dos demais dedos
    + Boundary matrix: É gerada uma "matriz de bordas" que é um array contendo o index dos pixels localizados nas extremidades das mãos.
    + Peak detection: Algoritmo que compara o valor de cada pixel nas bordas com os seus vizinhos (janela de 15 pixels) para determinar o valor de pico.
    + Picos na região do dedão são desconsiderados (20% do tamanho da mão)
    + Cálculo da distância euclidiana de cada pico ao centroide. 70% do maior valor - correspondente ao maior dedo - é considerado como threshold para determinar se o dedo está abaixado ou não.

        ![fingers](images/fingers.png?raw=true)


## Classificação

A determinação das features supracitada gera uma sequência binária de 5 valores indicando a posição dos dedos - 0 para abaixado e 1 para levantado. 
Por questões anatômicas, se algum dedo não for detectado, pressupõe-se que o mesmo está abaixado e do lado da mão no qual não se encontra o dedão.

![final](images/final.png?raw=true)

## Resultado

![result](images/result.png?raw=true)


## Instalação e uso

Install python
```
sudo apt-get update
sudo apt-get install python3.6
```

Create a virtual environment called "venv" and activate it
```
python3 -m venv venv
source venv/bin/activate
```

Install the required packages
```
python3 -m pip install -r requirements.txt
```

Run 
```
python3 recognizer.py
```
