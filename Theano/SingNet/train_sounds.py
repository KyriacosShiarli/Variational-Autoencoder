"""
Authors: 
Joost van Amersfoort - <joost.van.amersfoort@gmail.com>
Otto Fabius - <ottofabius@gmail.com>

#License: MIT
"""

"""This script trains an auto-encoder on the MNIST dataset and keeps track of the lowerbound"""


import os
import VariationalAutoencoder
import numpy as np
import argparse
import time
import gzip, cPickle
import scipy.io.wavfile
import os
from pydub import AudioSegment
from preprocess import pickle_loader,pickle_saver,map_to_range,map_to_range_symmetric
import matplotlib.pyplot as plt
import numpy
from sn_plot import plot_reconstructed





parser = argparse.ArgumentParser()
parser.add_argument("-d","--double", help="Train on hidden layer of previously trained AE - specify params", default = False)

args = parser.parse_args()
#TOFO: Make all input convensions N*D
#x_train = pickle_loader("sound/test_data.pkl")
data_dictionary = pickle_loader("sound/mistakidis.pkl") # data dictionary contains a "data" entry and a "sample rate" entry
x_train = data_dictionary["data"][:5,:]
#print all_data.shape
x_test = x_train # x_test is the same as x_train


#x_train = np.array([numpy.sin(t*100), numpy.sin(t*200), numpy.sin(t*40), numpy.sin(t*50)])*.5

n_steps = 2

dimZ = 4
HU_decoder = 200
HU_encoder = HU_decoder
batch_size = 4
L = 1
learning_rate = 0.002
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
encoder.sample_rate = data_dictionary["sample_rate"]
if args.double:
    encoder.continuous = True

print "Creating Theano functions"
encoder.createGradientFunctions()

print "Initializing weights and biases"
encoder.initParams()
lowerbound = np.array([])
testlowerbound = np.array([])

begin = time.time()

# Declare figures for interactive mode
plt.ion()
f1 = plt.figure(1)

for j in xrange(n_steps):
    encoder.lowerbound = 0
    print 'Iteration:', j
    begin = time.time()
    encoder.iterate(data)
    end = time.time()
    print("Iteration %d, lower bound = %.2f,"
          " time = %.2fs"
          % (j, encoder.lowerbound/N, end - begin))
    if j%10 == 0:
        mu_out3 = encoder.getTestOutput(data)
        plot_reconstructed(data,mu_out3.T,f1,interactive=True)

pickle_saver(encoder,"encoder.pkl")
