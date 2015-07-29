"""
Authors: 
Joost van Amersfoort - <joost.van.amersfoort@gmail.com>
Otto Fabius - <ottofabius@gmail.com>

#License: MIT
"""

"""This script trains an auto-encoder on the MNIST dataset and keeps track of the lowerbound"""

#python trainmnist.py -s mnist.npy

import VariationalAutoencoder
import numpy as np
import argparse
import time
import gzip, cPickle
import scipy.io.wavfile
import os
from pydub import AudioSegment
from preprocess import pickle_loader,pickle_saver,map_to_range
import matplotlib.pyplot as plt
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("-d","--double", help="Train on hidden layer of previously trained AE - specify params", default = False)

args = parser.parse_args()

print "Loading MNIST data"
#Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz

f = gzip.open('mnist.pkl.gz', 'r')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

#x_train = pickle_loader("sound/test_data.pkl")
x_train = pickle_loader("sound/puretone_data.pkl")

#print all_data.shape
x_test = x_train

t = np.linspace(0, 2, 4001)

#x_train = np.array([numpy.sin(t*100), numpy.sin(t*200), numpy.sin(t*40), numpy.sin(t*50)])*.5

n_steps = 400

dimZ = 4
HU_decoder = 200
HU_encoder = HU_decoder

batch_size = 4
L = 1
learning_rate = 0.005
data = x_train
print "shape of data",data.shape

if args.double:
    print 'computing hidden layer to train new AE on'
    prev_params = np.load(args.double)
    data = (np.tanh(data.dot(prev_params[0].T) + prev_params[5].T) + 1) /2
    x_test = (np.tanh(x_test.dot(prev_params[0].T) + prev_params[5].T) +1) /2

[N,dimX] = data.shape
encoder = VariationalAutoencoder.VA(HU_decoder,HU_encoder,dimX,dimZ,batch_size,L,learning_rate)
encoder.continuous = True

if args.double:
    encoder.continuous = True

print "Creating Theano functions"
encoder.createGradientFunctions()

print "Initializing weights and biases"
encoder.initParams()
lowerbound = np.array([])
testlowerbound = np.array([])

begin = time.time()
plt.ion()
for j in xrange(n_steps):
    encoder.lowerbound = 0
    print 'Iteration:', j
    encoder.iterate(data)
    end = time.time()
    print("Iteration %d, lower bound = %.2f,"
          " time = %.2fs"
          % (j, encoder.lowerbound/N, end - begin))


    if j%100 == 0:

        mu_out3 = encoder.getTestOutput(data)
        for i, (d, mu) in enumerate(zip(data, mu_out3.T)):

            plt.subplot(3,2,i+1)
            plt.cla()
            plt.plot(d[:200])
            plt.plot(mu[:200])

        plt.subplot(3,2,5)
        plt.imshow(encoder.params[0], interpolation = 'nearest', cmap = 'gray')
        plt.subplot(3,2,6)
        plt.imshow(encoder.params[1], interpolation = 'nearest', cmap = 'gray')

        plt.draw()

    
    #mu_out = encoder.getTestOutput(test_point)
    begin = end
    #print mu_out.shape

test_point = np.array([data[0,:]])
#mu_out = encoder.getTestOutput(test_point)
#print mu_out
z_val = np.array([[1,2,3,4]]).T
mu_out = encoder.generateOutput(z_val,test_point)
zed = encoder.getZ(data)
while True:
    z_val = raw_input("Input value for Z")
    mu_out = encoder.generateOutput(eval(z_val),test_point)
    print "The zeds are:",zed,mu_out.shape    
    
    
    mu_out2 = encoder.getTestOutput(test_point)

    mu_out3 = encoder.getTestOutput(data)
    

    for i, (d, mu) in enumerate(zip(data, mu_out3.T)):
        plt.subplot(2,3,i+1)
        plt.cla()
        plt.plot(d[:200])
        plt.plot(mu[:200])
        
    plt.show()

    # plt.plot(data.T, 'b')
    # plt.plot(mu_out3.T, 'r--')
    # plt.show()

    #import pdb; pdb.set_trace()



    #pickle_saver(mu_out,"outputs/test_1.pkl")  
    scipy.io.wavfile.write("test"+z_val+".wav", 2000, mu_out)

    scipy.io.wavfile.write("test_"+z_val+".wav", 2000, mu_out2)
    scipy.io.wavfile.write("test_datapoint"+z_val+".wav", 2000, test_point[0])

    plt.subplot(2,3,5)
    plt.cla()
    plt.plot(mu_out[:200])

#scipy.io.wavfile.write("test2", 2000, test_point[0])
    #if j % 5 == 0:
    #    print "Calculating test lowerbound"
    #    testlowerbound = np.append(testlowerbound,encoder.getLowerBound(x_test))
