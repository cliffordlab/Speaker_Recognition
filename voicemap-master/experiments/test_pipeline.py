import sys
sys.path.append('../voicemap/')
sys.path.append('../')
from librispeech import LibriSpeechDataset
from utils import whiten, contrastive_loss, preprocess_instances, BatchPreProcessor
from config import LIBRISPEECH_SAMPLING_RATE
from voicemap.models import get_baseline_convolutional_encoder, build_siamese_net
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from keras.models import load_model

from keras.models import Model
from keras.layers import Input
import keras.backend as K
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

def load_encoder(model_path, input_length):
    filters = 128
    embedding_dimension = 64
    dropout = 0.0
    #Load model
    encoder = get_baseline_convolutional_encoder(filters, embedding_dimension, dropout=dropout)
    siamese = build_siamese_net(encoder, (input_length, 1), distance_metric='uniform_euclidean')
    opt = Adam(clipnorm=1.)
    siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #siamese = load_model(model_path)
    siamese.load_weights(model_path)

    ## Load Encoder
    inputs = Input(shape=(input_length,1))
    encoded = siamese.layers[2](inputs)
    encoder = Model(inputs=inputs, outputs=encoded)
    encoder = siamese.layers[2]
    encoder.compile(loss='mse',optimizer='adam')
    return encoder

def split_files(dataset_instance):
    from sklearn.model_selection import train_test_split
    ## Split the train filenames to train, val_speakers
    train = dataset_instance
    speaker_list=train.df["speaker_id"]
    train_arr_filenames=train.df['filepath']
    print(train_arr_filenames.shape)

    indices = np.arange(len(train_arr_filenames))
    ind_train, ind_test = train_test_split(indices, test_size=0.33, random_state=42)
    train_arr_filenames[ind_train]
    X_train_files = train_arr_filenames[ind_train]
    X_test_files = train_arr_filenames[ind_test]
    y_train = speaker_list[ind_train]
    y_test = speaker_list[ind_test]
    return X_train_files, X_test_files, y_train, y_test

def read_data(input_data, train):
    X_array =[]
    #instance, sr = sf.read(input_data[i])
    for i in input_data.index.values:
        instance, sp = train[i]
        X_array.append(instance)
    X_array = np.array(X_array)
    return X_array

def extract_embedding(X_array, encoder, preprocessor):
    query_instance_ = preprocessor.instance_preprocessor(X_array[:,:,np.newaxis])
    emb = encoder.predict(query_instance_)
    return emb

if __name__ == '__main__':
    
    ## Initialising all Parameters
    n_seconds = 2
    downsampling = 4
    batchsize = 64
    save_dir = '../notebooks/saved_data_new/'
    input_length = int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)
    # model_path = '../models_new/siamese__filters_128__embed_64__drop_0.0__pad=True.hdf5'
    # model_path = '../models_e50_cluster/n_seconds/siamese__nseconds_1.0__filters_128__embed_64__drop_0.0__r_0.hdf5'
    model_path ='../models_retry/n_seconds/siamese__nseconds_2__filters_128__embed_64__drop_0.0__r_0_epochs_.hdf5'
    logfile = open('n_spkrs.log'+str(n_seconds), 'w')
    print(model_path, file = logfile)
    np.random.seed(2020)
    ## Training
    training_set = ['train-clean-100']
    train = LibriSpeechDataset(training_set, n_seconds, stochastic=False, pad=True)#, cache=False)

    batch_preprocessor = BatchPreProcessor("classifier", preprocess_instances(downsampling))
    preprocessor = batch_preprocessor

    X_train_files, X_test_files, y_train, y_test = split_files(train)

    train.read_audio_dataset()
    train.pre_read = True
    X_read = read_data(X_train_files, train)
    val_read = read_data(X_test_files,train)
    
    encoder = load_encoder(model_path, input_length)
    print("Extracting Embedding")
    saved = False
    if not saved: 
        emb = extract_embedding(X_read, encoder, preprocessor)
        val_emb = extract_embedding(val_read, encoder, preprocessor)
        np.save(save_dir+'emb_'+str(n_seconds)+'sec_sampling.npy',emb)
        np.save(save_dir+'val_emb_'+str(n_seconds)+ 'sec_sampling.npy',val_emb)
    else:
        emb = np.load(save_dir+'emb_'+str(n_seconds)+'sec_sampling.npy')
        val_emb = np.load(save_dir+'val_emb_'+str(n_seconds)+ 'sec_sampling.npy')
    print("Training classifier")
    ## Train Classifier
    clf = svm.SVC(gamma = 'scale', probability=True)
    clf.fit(emb, y_train)

    ## Print Accuracy
    y_pred = clf.predict(val_emb)
    
    print(accuracy_score(y_test, y_pred), file = logfile) 
    print(accuracy_score(y_test, y_pred)) 
    ## Saving classifier
    with open(save_dir+'clf_'+str(n_seconds)+'_sampling.pkl', 'wb') as f:
        pickle.dump(clf, f)

    run_num_experiment = True
    if run_num_experiment:
        from test_num_speakers import *
            run_exp_num_speakers(X_train_files, y_train, X_test_files, y_test, emb, val_emb, logfile, num_reps =100)

    logfile.close()
