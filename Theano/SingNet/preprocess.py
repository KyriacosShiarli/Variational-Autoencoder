import scipy.io.wavfile
import numpy as np
import cPickle as pickle

def pickle_saver(to_be_saved,full_directory):
	with open(full_directory,'wb') as output:
		pickle.dump(to_be_saved,output,-1)

def pickle_loader(full_directory):
	with open(full_directory,'rb') as input:
		return pickle.load(input)

def map_to_range(data,input_range,output_range,from_data = False):
	#data is a single numpy array please
	#assume data is greater or equal to 0
	if from_data==True:
		input_range =(np.amin(data),np.amax(data),)
	data = map(float,data)
	grad = (np.amax(output_range)-np.amin(output_range))/(np.amax(input_range)-np.amin(input_range))
	intercept = (np.amax(output_range) + np.amin(output_range))/2 - (np.amax(input_range) + np.amin(input_range))/2
	data_temp = np.copy(data)*grad
	return data_temp+intercept

def load_pure_tone_data():

	rate,data= scipy.io.wavfile.read("sound/toy_data_single_note/A.wav")
	all_data = map_to_range(data,[-32767. ,32767. ],[-1,1])
	rate,data= scipy.io.wavfile.read("sound/toy_data_single_note/B.wav")
	all_data = np.vstack((all_data,map_to_range(data,[-32767. ,32767. ],[-1,1])))
	rate,data= scipy.io.wavfile.read("sound/toy_data_single_note/E.wav")
	all_data = np.vstack((all_data,map_to_range(data,[-32767. ,32767. ],[-1,1])))
	rate,data= scipy.io.wavfile.read("sound/toy_data_single_note/D.wav")
	all_data = np.vstack((all_data,map_to_range(data,[-32767. ,32767. ],[-1,1])))
	pickle_saver(all_data,"sound/puretone_data.pkl")

if __name__ == "__main__":
	def bright_shinies():
		rate,data= scipy.io.wavfile.read("sound/brightshinies_sample.wav")
		print "MAX OF DATA", data.dtype
		#scipy.io.wavfile.write("reversal",44100,data)
		test = map_to_range(data[:,0],[-32767. ,32767. ],[-1,1])

		reverse = map_to_range(data[:,0],[-1,1],[-32767. ,32767. ])
		reverse = np.array(reverse,dtype = np.int16)
		#scipy.io.wavfile.write("reversal",44100,reverse)
		#fivefold subsampling 
		sample_length =20000
		num_datapoints = np.floor(test.shape[0]/sample_length)
		#print dat.shape
		rem = np.floor(test.shape[0]%sample_length)
		all_data = np.reshape(test[:-rem],(num_datapoints,sample_length))
		pickle_saver(all_data,"sound/test_data.pkl")
		scipy.io.wavfile.write("reversal",44100,all_data[0,:])
	load_pure_tone_data()
	data = pickle_loader("sound/puretone_data.pkl")
	print data.shape