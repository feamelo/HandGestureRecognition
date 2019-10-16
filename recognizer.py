#################################### Recognizer #####################################
# Autores: Fernanda Amaral Melo, Luiz Fernando Neves de Araújo   				
# Contato: fernanda.amaral.melo@gmail.com										
#          luizfna@gmail.com                                                        
#																				
# Script detecção de gestos baseado na extração de features da mão				
#																				
#####################################################################################

import os
import cv2
import sys
import math
import imutils
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

plot_ =  True
error_ = False

class gestures(object):
    #classe que realiza as operações
    def __init__(self,file_name):
        #Inicialização lendo a imagem e convertendo pra YCbCr
        self.pic = cv2.imread(file_name)
        #Tivemos um problema, que o celular da colega salvva imagens em 4k
        #Para resolver isso, fizemos essa conversão para sanar o problema
        if (self.pic.shape[0]>1100):
            self.pic = cv2.resize(self.pic,(720,1080))
        self.ycbcr = cv2.cvtColor(self.pic, cv2.COLOR_RGB2YCR_CB)
        x = int(self.ycbcr.shape[0]/65)
        y = int(self.ycbcr.shape[1]/65)
        self.kernel = np.ones((x,y))

    def runner(self):

        #metodo que roda as operações
        #clusterizacao pra definicao de backgroun e foreground
        self.kmeans_cluster()
        #caso a definicao de background e foreground esteja invertid, é verificado
        #o primeiro pixel para a inversão
        if self.thresh[1,1] > 0:
            self.thresh = cv2.bitwise_not(self.thresh)

        self.thresh = self.morph_bin()
        #definição dos contornos da mão
        self.coords = self.find_contours()

        x,y,w,h = self.coords
        cv2.rectangle(self.thresh,(x,y),(w,h),(255,0,0),2)
        self.width = w - x
        self.length = h - y
        self.img_ratio = self.length/self.width

        #detecção da presença do dedão
        thumbs = self.thumb_detect()
        tl,tu,tr,td,thumb = thumbs
        self.boundary_detect()
        #detecção dos outros dedos
        self.find_peaks(thumbs)

        return

    def kmeans_cluster(self):
        #função que clusteriza a imagem em dois, separando duas grandes regiões
        conver = np.float32(self.ycbcr.reshape((-1,3)))
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, 20, 1)
        ret,label,center = cv2.kmeans(conver,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((self.ycbcr.shape))
        thresh1 = cv2.threshold(res2,100,255, cv2.THRESH_BINARY_INV)[1]
        self.thresh,unp_var,unp_var2 = cv2.split(thresh1)

        return

    def morph_bin(self):
        #metodo para operações morfologicas, no caso basicamente remove
        #ruidos
        opening = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, self.kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)

        return closing

    def find_contours(self):

        major = cv2.__version__.split('.')[0]
        if major == '3':
            _, contours, _ = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        c = max(contours, key=cv2.contourArea)

        left = tuple(c[c[:, :, 0].argmin()][0])
        right = tuple(c[c[:, :, 0].argmax()][0])
        top = tuple(c[c[:, :, 1].argmin()][0])
        bot = tuple(c[c[:, :, 1].argmax()][0])
        M = cv2.moments(c)
        self.cx = int(M['m10']/M['m00'])
        self.cy = int(M['m01']/M['m00'])
        self.boundary_matrix = c
        #retorna as coordenadas dos limites da função
        return left[0],top[1],right[0],bot[1]


    def thumb_detect(self):
        #função que detecta o dedão, if define se a mão está na vertical ou
        #horizontal
        x,y,w,h = self.coords
        thumb_left = False
        thumb_right = False
        thumb_up = False
        thumb_down = False
        thumb = False

        if self.img_ratio > 1:
            left = np.mean(self.thresh[y:h,x:(x+int((w-x)/20))])
            right = np.mean(self.thresh[y:h,(w-int((w-x)/20)):w])
            if left < 38:
                thumb_left = True
                thumb = True
            elif right <  38:
                thumb_right = True
                thumb = True
        else:
            up = np.mean(self.thresh[y:(y+(int((h-y)/20))),x:w])
            down = np.mean(self.thresh[(h-int((h-y)/20)):h,x:w])
            if up < 38:
                thumb_up = True
                thumb = True
            elif down < 38:
                thumb_down = True
                thumb = True

        #retorna booleanos para as possiveis saidas
        return thumb_left,thumb_up,thumb_right,thumb_down,thumb

    def boundary_detect(self):
        #vê na matriz de bordas os pontos interessantes
        #para detecção dos dedos ignorando o que está fora da região
        #definida pelo centroide e das bordas dos dedos

        y = self.boundary_matrix[:, :, 1]
        x = self.boundary_matrix[:, :, 0]

        xmin,ymin,xmax,ymax = self.coords
        self.boundary_matrix = []
        for i in range(0,np.shape(y)[0]):
            if (x[i,0]>xmin and x[i,0]<xmax and y[i,0]>ymin and y[i,0]<ymax): # if inside hand area
                if (self.img_ratio > 1 and y[i]<self.cy) : # Vertical
                    self.boundary_matrix.append([x[i,0],y[i,0]])
                elif (self.img_ratio < 1 and x[i]>self.cx): # Horizontal
                    self.boundary_matrix.append([x[i,0],y[i,0]])

        self.boundary_matrix = np.asarray(self.boundary_matrix)

        return


    def find_peaks(self,thumbs):
        #Metodo que detecta os picos nas bordas, possibilitando assim
        #a identificção dos dedos na região de interesse]
        isVertical = 1 if self.img_ratio>1 else 0

        # acha os picos
        peaks = argrelextrema(self.boundary_matrix[:,1 if isVertical else 0], np.less_equal if isVertical else np.greater_equal,order=20)

        # pega a distancia euclidiana e defini os limites verticais e horizontais
        distances = self.euclidean_dist(self.boundary_matrix[peaks,0],self.cx,self.boundary_matrix[peaks,1],self.cy)
        self.threshold = np.where(distances == np.max(distances))
        self.threshold = self.boundary_matrix[peaks[0][self.threshold[1][0]]]
        self.threshold = [self.cx + (self.threshold[0]-self.cx)*0.70, self.cy - (self.cy - self.threshold[1])*0.6]
        thresh_dist = self.width/8 if isVertical else self.length/8


        # decide os picos relevantes
        fingers = []
        for i in range (0, np.shape(peaks[0])[0]-1 if peaks[0][0]==0 else np.shape(peaks[0])[0]):
            if(self.analyze_thumb(thumbs,self.boundary_matrix[peaks[0][i],:])):
                if(not fingers or all(self.euclidean_dist(self.boundary_matrix[finger,0], self.boundary_matrix[peaks[0][i],0], self.boundary_matrix[finger,1], self.boundary_matrix[peaks[0][i],1]) > thresh_dist for finger in np.array(fingers)[:,0])):
                    if(isVertical):
                        fingers.append([peaks[0][i], 1 if self.boundary_matrix[peaks[0][i],1] < self.threshold[1] else 0])
                    else:
                        fingers.append([peaks[0][i], 1 if self.boundary_matrix[peaks[0][i],0] > self.threshold[0] else 0])

        self.fill_sequence(fingers,thumbs)
        return

    def euclidean_dist(self,x1,x2,y1,y2):
        return np.sqrt(np.power(x1-x2,2) + np.power(y1-y2,2))

    def analyze_thumb(self, thumbs, index):
        # desconsidera picos na região do polegar
        if (self.img_ratio>1):
            if (thumbs[0] and index[0]<self.coords[0]+self.width/5):
                return False
            elif (thumbs[2] and index[0]>self.coords[2]-self.width/5):
                return False
        else:
            if (thumbs[1] and index[1]<self.coords[1]+self.length/5 and index[1]>self.coords[1]):
                return False
            elif (thumbs[3] and index[1]<self.coords[3]-self.length/5):
                return False

        return True

    def fill_sequence(self,fingers,thumbs):
        isVertical = 1 if self.img_ratio>1 else 0
        fingers = np.array(fingers)
        # numero de dedos detectados
        size = np.shape(fingers)[0]
        self.finger_seq = fingers[[self.boundary_matrix[fingers[:,0],0].argsort()],1]

        # completa com 0s as posições que faltam
        zeros = np.array([])
        for i in range (0,4-size):
            zeros=np.append(zeros,0)

        if (size == 0 or size> 5):
            self.finger_seq = np.zeros(5)
        elif (size > 5):
            self.find_peaks(thumbs,5)
        elif size == 5:
            #substitui a posição do polegar
            self.finger_seq[0][4 if thumbs[2] or thumbs[3] else 0]=thumbs[4]
        elif((thumbs[2] or thumbs[1])):
            # adiciona o polegar para completar o vetor
            self.finger_seq = np.append(zeros,np.append(self.finger_seq[0],1))
        else:
            self.finger_seq = np.append(thumbs[4], np.append(self.finger_seq[0],zeros))

        self.classification()
        if(plot_):
            self.plot(fingers)

    def classification(self):
        #classificações dos gestos
        self.label = 'none'

        labels = {  '1': [0,1,0,0,0],
                    '2': [0,0,1,1,0],
                    'Rock': [0,1,0,0,1],
                    '3': [0,1,1,1,0],
                    'Super Rock': [1,1,0,0,1],
                    'Ta ok?!': [0,0,1,1,1],
                    '4': [1,1,1,1,0],
                    '5': [1,1,1,1,1],
                    'arma': [0,0,0,1,1],
        }

        for lab,seq in labels.items():
            if ((np.array(seq) == self.finger_seq).all() or (np.array(seq[::-1]) == self.finger_seq).all()):
                self.label = lab
                break

        return


    def plot(self,fingers):
        #função para plotar os resultados
        fig = plt.figure()
        plt.subplot(221)
        fig.suptitle('Gesture: ' + str(self.label) + ' -   Sequence: '+str(self.finger_seq))

        plt.imshow(self.pic)
        plt.subplot(222)
        plt.imshow(self.thresh)

        plt.subplot(223)
        if (self.img_ratio < 1): # horizontal
            plt.plot(self.boundary_matrix[:,0])
            plt.plot(fingers,self.boundary_matrix[fingers,0], "x")
            plt.hlines(self.threshold[0], 0, np.shape(self.boundary_matrix[:,0])[0], colors='r', linestyles="dashed")
        else: # vertical
            plt.plot(self.boundary_matrix[:,1])
            plt.plot(fingers,self.boundary_matrix[fingers,1], "x")
            plt.hlines(self.threshold[1], 0, np.shape(self.boundary_matrix[:,0])[0], colors='r', linestyles="dashed")

        plt.subplot(224)
        plt.plot(self.boundary_matrix[:,0], self.boundary_matrix[:,1])
        plt.plot(self.boundary_matrix[fingers,0], self.boundary_matrix[fingers,1], "x")
        plt.show()

for i in range(1,14):
    fileDir = os.path.dirname(os.path.abspath(__file__))
    detect = gestures(fileDir + '/dataset/'+str(i)+'.JPG')
    detect.runner()
