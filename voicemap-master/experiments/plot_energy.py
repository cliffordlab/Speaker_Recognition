import numpy as np
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/sanmathikamath/projects/Speaker_Recognition/voicemap-master")
from voicemap.librispeech import LibriSpeechDataset

datafile = '/Users/sanmathikamath/projects/Speaker_Recognition/voicemap-master/data/LibriSpeech/train-clean-100/8419/286667/8419-286667-0001.flac'

#global pad_count
def get_energy(datafile, n_seconds=3, bin_size = 0.01, pad = True, max_len = 5):
	# Read file
	x, sr = sf.read(datafile)
	#Choose starting point at random
	fragment_length = sr*n_seconds
	max_len = sr*max_len
	fragment_start_index = np.random.randint(0, max(len(x) - max_len, 1))
	# fragment_start_index = 0
	selected_x = x[fragment_start_index : fragment_start_index + fragment_length]
	print(fragment_start_index)
	# Check for required length and pad if necessary
	if pad and len(selected_x) < fragment_length:
		less_timesteps = fragment_length - len(selected_x)
		# Stochastic padding, ensure instance length == self.fragment_length by appending a random number of 0s
		# before and the appropriate number of 0s after the instance
		before_len = np.random.randint(0, less_timesteps)
		after_len = less_timesteps - before_len
		selected_x = np.pad(selected_x, (before_len, after_len), "constant")
		#pad_count +=1

	#Form bins
	bin_l = int(sr*bin_size)
	window = signal.get_window('boxcar',bin_l)
	
	#Compute energy in bins
	energy_bins = np.convolve(np.power(selected_x,2) , window,'valid')[::bin_l]
	#print(np.sum(energy_bins))
	return np.sort(energy_bins)


if __name__ == '__main__':
	#Get sorted energy in bins of 10ms
	training_set = ["train-clean-100"]
	pad_count = 0
	n_seconds = 1
	bin_size = 0.01
	train = LibriSpeechDataset(training_set, n_seconds, pad=True)
	files = list(train.datasetid_to_filepath.values())
	energy_bins = np.zeros((len(files), int(n_seconds/bin_size)))
	i = 0
	np.random.seed(42)
	for datafile in files[0:5]:
		#print(i)
		energy_bins[i,:] = get_energy(datafile, n_seconds, bin_size)
		
		#print(np.sum(energy_bins[i,:]))
		i+=1



	# fig=plt.figure();
	# #Plt average
	# # img = plt.imshow(np.sum(energy_bins, axis =0).reshape(1,-1),aspect='auto',cmap='Blues');
	# # img = plt.imshow(energy_bins,aspect='auto',cmap='Blues');
	# # fig.colorbar(img);
	# # plt.show()
	

	# energy_bins_1 = np.load('energy_bins_1.npz.npy')
	# energy_bins_2 = np.load('energy_bins_2.npz.npy')
	# energy_bins_3 = np.load('energy_bins_3.npz.npy')
	# energy_bins_4 = np.load('energy_bins_4.npz.npy')
	# energy_bins_5 = np.load('energy_bins_5.npz.npy')
	
	# avg_en_1 = np.mean(energy_bins_1, axis =0)
	# avg_en_2 = np.mean(energy_bins_2, axis =0)
	# avg_en_3 = np.mean(energy_bins_3, axis =0)
	# avg_en_4 = np.mean(energy_bins_4, axis =0)
	# avg_en_5 = np.mean(energy_bins_5, axis =0)
	
	# #fig, ax = plt.subplots(5,1)
	# # img = ax[0].imshow(np.mean(energy_bins_1, axis =0).reshape(1,-1),aspect='auto',cmap='Blues')
	# # ax[0].set_title('1 seconds')	
	# # img = ax[1].imshow(np.mean(energy_bins_2, axis =0).reshape(1,-1),aspect='auto',cmap='Blues')
	# # ax[1].set_title('2 seconds')
	# # img = ax[2].imshow(np.mean(energy_bins_3, axis =0).reshape(1,-1),aspect='auto',cmap='Blues')
	# # ax[2].set_title('3 seconds')	
	# # img = ax[3].imshow(np.mean(energy_bins_4, axis =0).reshape(1,-1),aspect='auto',cmap='Blues')
	# # ax[3].set_title('4 seconds')
	# # img = ax[4].imshow(np.mean(energy_bins_5, axis =0).reshape(1,-1),aspect='auto',cmap='Blues')
	# # ax[4].set_title('5 seconds')
	# # [axi.get_yaxis().set_visible(False) for axi in ax.ravel()]
	# # plt.tight_layout()
	# # fig.colorbar(img, ax=ax.ravel().tolist())
	# # plt.show()

	# fig, ax = plt.subplots(5,1)
	# img = ax[0].plot(np.mean(energy_bins_1, axis =0))
	# ax[0].set_title('1 seconds')
	# img = ax[1].plot(np.mean(energy_bins_2, axis =0))
	# ax[1].set_title('2 seconds')
	# img = ax[2].plot(np.mean(energy_bins_3, axis =0))
	# ax[2].set_title('3 seconds')	
	# img = ax[3].plot(np.mean(energy_bins_4, axis =0))
	# ax[3].set_title('4 seconds')
	# img = ax[4].plot(np.mean(energy_bins_5, axis =0))
	# ax[4].set_title('5 seconds')
	# [axi.set_xlim((0,500)) for axi in ax.ravel()]
	# # [axi.get_yaxis().set_visible(False) for axi in ax.ravel()]
	# plt.tight_layout()
	# # fig.colorbar(img, ax=ax.ravel().tolist())
	# plt.show()

	# # #Plot Hist
	# # plt.figure()
	# # fig, ax = plt.subplots(2,3)
	# # x, bins, p = ax[1][1].hist(avg_en_5, 20, density=True)
	# # for item in p:
	# # 	item.set_height(item.get_height()/sum(x))
	# # ax[1][1].set_title('5 seconds')

	# # x, bins, p = ax[0][0].hist(avg_en_1, bins, density=True)
	# # for item in p:
	# # 	item.set_height(item.get_height()/sum(x))
	# # ax[0][0].set_title('1 seconds')	

	# # x, bins, p = ax[0][1].hist(avg_en_2, bins, density=True)
	# # for item in p:
	# # 	item.set_height(item.get_height()/sum(x))
	# # ax[0][1].set_title('2 seconds')

	# # x, bins, p = ax[0][2].hist(avg_en_3, bins, density=True)
	# # for item in p:
	# # 	item.set_height(item.get_height()/sum(x))
	# # ax[0][2].set_title('3 seconds')	

	# # x, bins, p = ax[1][0].hist(avg_en_4, bins, density=True)
	# # for item in p:
	# # 	item.set_height(item.get_height()/sum(x))
	# # ax[1][0].set_title('4 seconds')
	# # [axi.set_ylim((0,1)) for axi in ax.ravel()]
	# # fig.delaxes(ax[1][2])
	# # plt.tight_layout()
	# # #fig.colorbar(img, ax=ax.ravel().tolist())
	# # plt.show()


	# #Plot cumulative 
	# fig, ax = plt.subplots(5,1)
	# img = ax[0].plot(np.cumsum(avg_en_1))
	# ax[0].set_title('1 seconds')
	# img = ax[1].plot(np.cumsum(avg_en_2))
	# ax[1].set_title('2 seconds')
	# img = ax[2].plot(np.cumsum(avg_en_3))
	# ax[2].set_title('3 seconds')	
	# img = ax[3].plot(np.cumsum(avg_en_4))
	# ax[3].set_title('4 seconds')
	# img = ax[4].plot(np.cumsum(avg_en_5))
	# ax[4].set_title('5 seconds')
	# # [axi.get_yaxis().set_visible(False) for axi in ax.ravel()]
	# plt.tight_layout()
	# # fig.colorbar(img, ax=ax.ravel().tolist())
	# plt.show()


