///////////////////Feature Extraction///////////////////////
import matplotlib.pyplot as pyplot
import numpy as np
import cv2
import itertools
from sklearn import preprocessing
from scipy.fftpack import dct
from scipy.cluster.vq import kmeans,vq
from sklearn.decomposition import PCA
import scipy.ndimage.filters
import pickle

SURF_WINDOW = 20
DCT_WINDOW = 20
windowSize = 10
gridSpacing = 7

class Colorizer(object):
    def __init__(self, ncolors=16, probability=False, npca=32, graphcut_lambda=1, ntrain=6550 ):
        #number of bins in the discretized a,b channels
        self.levels = int(np.floor(np.sqrt(ncolors)))
        #self.ncolors = self.levels**2 #recalculate ncolors in case the provided parameter is not a perfect square
        self.ncolors = ncolors
        self.ntrain = ntrain
        self.npca = npca

        self.scaler = preprocessing.MinMaxScaler() # Scaling object -- Normalizes feature array
        
        self.pca = PCA(npca)

        self.centroids = []
        self.probability = probability
        self.colors_present = []
        #self.surf = cv2.DescriptorExtractor_create('SURF')
        #self.surf.setBool('extended', True) #use the 128-length descriptors



    def load_image(self, path):
        '''
        Read in a file and separate into L*a*b* channels
        '''
        img = cv2.imread(path) 
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))

        return l, a, b


    '''Method to train the NN and train a kmean to get the 16 classification of colors'''
    def train(self, files):
        '''
        -- Reads in a set of training images. 
        -- Converts from RGB to LAB colorspace.
        -- Extracts feature vectors at each pixel of each training image.
        -- (complexity reduction?).
        '''

        features = []
        self.local_grads = []
        classes = []
        numTrainingExamples = 0

        kmap_a = []
        kmap_b = []


        # compute color map
        for f in files:

            l,a,b = self.load_image(f)
            kmap_a = np.concatenate([kmap_a, a.flatten()])
            kmap_b = np.concatenate([kmap_b, b.flatten()])

        # Training K Means using two Kmaps to form 'ncolors' clusters 
        self.train_kmeans(kmap_a,kmap_b,self.ncolors)

        for f in files:
            l,a,b = self.load_image(f)            
            a,b = self.quantize_kmeans(a,b)

            #dimensions of image
            m,n = l.shape 

            
            #extract features from training image
            # (Select uniformly-spaced training pixels)
            #for x in xrange(int(gridSpacing/2),n,gridSpacing):
            #    for y in xrange(int(gridSpacing/2),m,gridSpacing):
            #extract features from training image
            
            for i in range(self.ntrain):
                #choose random pixel in training image
                x = int(np.random.uniform(n))
                y = int(np.random.uniform(m))
    
                features.append(self.get_features(l, (x,y)))
                classes.append(self.color_to_label_map[(a[y,x], b[y,x])])

                numTrainingExamples = numTrainingExamples + 1

        # normalize columns
        self.features = self.scaler.fit_transform(np.array(features))
        classes = np.array(classes)

        #print ("Size, Pre-PCA")
        #print (np.shape(self.features))
        # reduce dimensionality
        self.features = self.pca.fit_transform(self.features)

        #print ("Size, Post-PCA")
        print (np.shape(self.features))
        print (self.features)
        return self


    def train_kmeans(self, a, b, k):
        # w,h = np.shape(a)
        #pixel = np.reshape((cv2.merge((a,b))),(w * h,2))
        pixel = np.squeeze(cv2.merge((a.flatten(),b.flatten())))
        
        #print ("pixel array size: ")
        #print (np.shape(pixel))

        # cluster
        self.centroids,_ = kmeans(pixel,k) # sixteen colors will be found
 
        # quantization
        qnt,_ = vq(pixel,self.centroids)

        #color-mapping lookup tables
        self.color_to_label_map = {c:i for i,c in enumerate([tuple(i) for i in self.centroids])} #this maps the color pair to the index of the color
        self.label_to_color_map = dict(zip(self.color_to_label_map.values(),self.color_to_label_map.keys())) #takes a label and returns a,b
          

    def quantize_kmeans(self, a, b):
        w,h = np.shape(a)
        
        # reshape matrix
        pixel = np.reshape((cv2.merge((a,b))),(w * h,2))

        # quantization
        qnt,_ = vq(pixel,self.centroids)

        # reshape the result of the quantization
        centers_idx = np.reshape(qnt,(w,h))
        clustered = self.centroids[centers_idx]

        a_quant = clustered[:,:,0]
        b_quant = clustered[:,:,1]
        return a_quant, b_quant


    def get_features(self, img, pos):
        intensity = np.array([img[pos[1], pos[0]]])
        position = self.feature_position(img, pos)
        meanvar = np.array([self.getMean(img, pos), self.getVariance(img, pos)]) #variance is giving NaN
        #feat = np.concatenate((position, meanvar, self.feature_surf(img, pos)))
        #feat = np.concatenate((meanvar, self.feature_surf(img, pos)))
        #feat = np.concatenate((meanvar, self.feature_surf(img, pos), self.feature_dft(img, pos)))
        feat = np.concatenate((meanvar, self.feature_dft(img, pos)))
        
        #print (feat)
        return feat


    def feature_position(self, img, pos):
        '''
        Returns position of the pixel in the image based on pos values
        '''
        m,n = img.shape
        x_pos = pos[0]/n
        y_pos = pos[1]/m

        return np.array([x_pos, y_pos])


    def getMean(self, img, pos):
        ''' 
        Returns mean value over a windowed region around (x,y)
        '''
        xlim = (max(pos[0] - windowSize,0), min(pos[0] + windowSize,img.shape[1]))
        ylim = (max(pos[1] - windowSize,0), min(pos[1] + windowSize,img.shape[0]))

        return np.mean(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])


    def getVariance(self, img, pos):
        '''
        Returns variance value over a windowed region around (x,y)
        '''
        xlim = (max(pos[0] - windowSize,0), min(pos[0] + windowSize,img.shape[1]))
        ylim = (max(pos[1] - windowSize,0), min(pos[1] + windowSize,img.shape[0]))

        return np.var(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])/1000 #switched to Standard Deviation --A


    '''  
    def feature_surf(self, img, pos):
    
        octave2 = cv2.GaussianBlur(img, (0, 0), 1)
        octave3 = cv2.GaussianBlur(img, (0, 0), 2)
        kp = cv2.KeyPoint(pos[0], pos[1], SURF_WINDOW)
        _, des1 = self.surf.compute(img, [kp])
        _, des2 = self.surf.compute(octave2, [kp])
        _, des3 = self.surf.compute(octave3, [kp])

        return np.concatenate((des1[0], des2[0], des3[0]))
    '''


    def feature_dft(self, img, pos):
        xlim = (max(pos[0] - windowSize,0), min(pos[0] + windowSize,img.shape[1]))
        ylim = (max(pos[1] - windowSize,0), min(pos[1] + windowSize,img.shape[0]))
        patch = img[ylim[0]:ylim[1],xlim[0]:xlim[1]]
        
        l = (2*windowSize + 1)**2

        #return all zeros for now if we're at the edge
        if patch.shape[0]*patch.shape[1] != l:
            return np.zeros(l)

        return np.abs(np.fft(patch.flatten()))  
        

if __name__ == '__main__':
    training_files = ['test.jpg' ]
    c = Colorizer()
    c.train(training_files)