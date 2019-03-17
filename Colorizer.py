import numpy as np
import random
import cv2
from sklearn.cluster import KMeans
import math
import os

#Create a cluster space with 16 clusters
kmeans = KMeans(16)

#Read all images in Cluster folder and create 16 clusters based on all pixel values
path = 'Cluster'
for filename in os.listdir(path):
    print(filename)
    image = cv2.imread(filename)
    l,a,b = cv2.split(cv2.cvtColor(image,cv2.COLOR_BGR2Lab))
    pixels = np.squeeze(cv2.merge((a.flatten(),b.flatten())))
    kmeans.fit(pixels)

#Neural Network initialization
#Initial weights and biases
class Network:
    def __init__(self,neurons):
        self.layers = len(neurons)
        self.neurons = neurons
        self.weights = []
        self.biases = []
        
        for i in range(1,self.layers):
            rows = neurons[i]
            cols = neurons[i-1]
            
            #Creates a weight numpy array with dimension = dims
            #Initializes them with random normal distributed value between 0-1
            
#             layer_weight = np.zeros((rows,cols))
            layer_weight = np.random.randn(rows,cols)
            self.weights.append(layer_weight)
#             layer_bias = np.random.randn(rows,1)
            layer_bias = np.zeros((rows,1))
            self.biases.append(layer_bias)

#Sigmoid activation
def sigmoid(z):
    
    return (1.0/(1.0+np.exp(-z)))


#Feed forward
def feedforward(my_net,X):
    layer_activation_list = []
    weighted_sum_list = []
    for i in range(1,my_net.layers):
        if i==1:
            a = X
        else:
            a = layer_activation_list[-1]
            
        weight = my_net.weights[i-1]
        biases = my_net.biases[i-1]
        z = np.dot(weight,a) + biases
        weighted_sum_list.append(z)
        a = sigmoid(z)
        layer_activation_list.append(a)
        
    return (layer_activation_list,weighted_sum_list)
    
#Calculate the output for test data
def test(network,data):
    
    final_image = []
    
    for i in range(0,len(data)):
        activation_list,weighted_sum_list = feedforward(network,data[i][0])
        prediction = np.argmax(activation_list[-1],axis=0)
        final_image.append(prediction)
        
    return final_image

#Calculate cost gradient
def cost_gradient(activation,target):
    
    return (activation - target)

#Calculate sigmoid derivative
def sigmoid_derivative(z):
    
    return sigmoid(z)*(1-sigmoid(z))

#Updating weights while backprop
def update_weights(network,activation_list,delta_list,alpha,X):
    
    for i in range(network.layers-1):
        if i==0:
            dcdw = np.dot(delta_list[i],X.transpose())
        else:        
            dcdw = np.dot(delta_list[i],activation_list[i-1].transpose())
        dcdb = np.average(delta_list[i],axis=1).reshape(delta_list[i].shape[0],1)
        
        network.weights[i] -= alpha*dcdw
        network.biases[i] -= alpha*dcdb
        
    return

#Calculate error at all layers
def errorback(network,last_delta,weighted_list):
    
    delta_list = []
    delta_list.append(last_delta)
    
    for i in range(network.layers-2,0,-1):
        delta = (np.dot((network.weights[i]).transpose(),delta_list[-1]))*(sigmoid(weighted_list[i-1]))
        delta_list.append(delta)
        
    return delta_list

#Gradient Descent
def gradient_descent(network,train_data,alpha,epochs):
    
    for i in range(0,len(train_data)):
        X = train_data[i][0]
        target = train_data[i][1]

        activation_list, weighted_list = feedforward(my_net,X)

        last_layer_delta = cost_gradient(activation_list[-1],target)*sigmoid_derivative(weighted_list[-1])

        final_delta_list = errorback(my_net,last_layer_delta,weighted_list)
        final_delta_list = final_delta_list[::-1]

        update_weights(my_net,activation_list,final_delta_list,alpha,X)
        
        print ("Input {0}: ".format(i))

    return

#Hyperparameters
my_net = Network([49, 30, 16])
epochs = 50
alpha = 0.0000000000001
patch_size = 7

#Preparing training data
path = 'Input_Images'
for filename in os.listdir(path):
    image_path = filename
    input_image = cv2.imread(image_path)
    l, a, b = cv2.split(cv2.cvtColor(input_image, cv2.COLOR_BGR2Lab))
    pixels = np.squeeze(cv2.merge((a.flatten(),b.flatten())))
    rows = l.shape[0]
    cols = l.shape[1]
    print(rows,cols)
    predict_values = kmeans.predict(pixels)
    cluster_values = predict_values.reshape((rows,cols))
    input_data = []
    target = []
    for i in range(0,rows-(patch_size-1)):
        for j in range(0,cols-(patch_size-1)):
            input_data.append(l[i:i+patch_size,j:j+patch_size].reshape((patch_size*patch_size,1)))
            target_vector = np.zeros((16,1))
            target_vector[cluster_values[i+3,j+3]] = 1
            target.append(target_vector)

    train_data = []
    
    #Training phase
    for i in range(0,len(input_data)):
        train_data.append([input_data[i],target[i]])
        
    gradient_descent(my_net,train_data,alpha,epochs)


#Preparing testing data
test_image_path = 'img17.jpg'
test_image = cv2.imread(test_image_path)
tl, ta, tb = cv2.split(cv2.cvtColor(test_image, cv2.COLOR_BGR2Lab))
rows = tl.shape[0]
cols = tl.shape[1]
print(rows,cols)
# input_data = []
test_target = []
output_data = []
for i in range(0,rows-(patch_size-1)):
    for j in range(0,cols-(patch_size-1)):
        output_data.append(tl[i:i+patch_size,j:j+patch_size].reshape((patch_size*patch_size,1)))
        test_target.append(ta[i+3,j+3]/255)
        
test_data = []
for i in range(0,len(output_data)):
    test_data.append([output_data[i],test_target[i]])

final_image = test(my_net,test_data)
final_image = np.reshape(final_image, (test_image.shape[0]-(patch_size-1),test_image.shape[1]-(patch_size-1)))

#Writing and combining output to image

ALayer = np.zeros((test_image.shape[0]-(patch_size-1),test_image.shape[1]-(patch_size-1)))
BLayer = np.zeros((test_image.shape[0]-(patch_size-1),test_image.shape[1]-(patch_size-1)))

for i in range(0,test_image.shape[0]-(patch_size-1)):
    for j in range(0,test_image.shape[1]-(patch_size-1)):
        ALayer[i][j] = math.floor(kmeans.cluster_centers_[final_image[i][j]][0])
        BLayer[i][j] = math.floor(kmeans.cluster_centers_[final_image[i][j]][1])

cv2.imwrite('ALayer.jpg',ALayer)
cv2.imwrite('BLayer.jpg',BLayer)

A = cv2.imread('ALayer.jpg',cv2.IMREAD_GRAYSCALE)
B = cv2.imread('BLayer.jpg',cv2.IMREAD_GRAYSCALE)

L = tl[3:253,3:253]
LABmerge = cv2.merge((ourL, A, B))
imageBGR = cv2.cvtColor(LABmerge, cv2.COLOR_LAB2BGR)
cv2.imwrite('colorized.jpg',imageBGR)

#Comparing it with actual colored image

image = cv2.imread('img17.jpg')
l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2Lab))

cv2.imwrite('L.jpg',l)
cv2.imwrite('A.jpg',a)
cv2.imwrite('B.jpg',b)

ourA = cv2.imread('c.jpg',cv2.IMREAD_GRAYSCALE)
ourB = cv2.imread('d.jpg',cv2.IMREAD_GRAYSCALE)
ourL = l[3:253,3:253]

L1 = l[3:253,3:253]
A1 = a[3:253,3:253]
B1 = b[3:253,3:253]

LABmerge = cv2.merge((ourL,ourA,ourB))
LABmerge2 = cv2.merge((L1,A1,B1))

imageBGR = cv2.cvtColor(LABmerge, cv2.COLOR_LAB2BGR)
imageBGRoriginal = cv2.cvtColor(LABmerge2, cv2.COLOR_LAB2BGR)
cv2.imwrite('colorized2.jpg',imageBGR)
cv2.imwrite('colorized2original.jpg',imageBGRoriginal)
cv2.imwrite('merge.jpg',LABmerge)
cv2.imwrite('mergeoriginal.jpg',LABmerge2)