import sys
sys.path.append('../voicemap/')
sys.path.append('../')
sys.path.append("/home/skamat2/SpeakerRecognition/voicemap-master")
from voicemap.librispeech import LibriSpeechDataset
from voicemap.utils import whiten, contrastive_loss, preprocess_instances, BatchPreProcessor
from config import LIBRISPEECH_SAMPLING_RATE, PATH
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
    
    downsampling = 4
    batchsize = 64
    save_dir = PATH+'/notebooks/saved_data_prev_2/'
    training_set = ['train-clean-100']
    
    batch_preprocessor = BatchPreProcessor("classifier", preprocess_instances(downsampling))
    preprocessor = batch_preprocessor

    models = ['/models_e50_full/n_seconds/siamese__nseconds_1.0__filters_128__embed_64__drop_0.0__r_0.hdf5',
            '/models_retry/n_seconds/siamese__nseconds_2__filters_128__embed_64__drop_0.0__r_0_epochs_.hdf5'] 
            #models = ['/models_retry/n_seconds/siamese__nseconds_3__filters_128__embed_64__drop_0.0__r_0.hdf5',
            #'/models_retry/n_seconds/siamese__nseconds_4__filters_128__embed_64__drop_0.0__r_0.hdf5',
            #'/models_retry/n_seconds/siamese__nseconds_5__filters_128__embed_64__drop_0.0__r_0.hdf5']
            
    
    total_seconds =[1,2]
    i=0
    for n_seconds in total_seconds:
        ## Initialising all Parameters
        input_length = int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)
        #model_path = PATH+'/models_360_250/n_seconds/siamese__nseconds_'+str(n_seconds)+'__filters_128__embed_64__drop_0.0__r_0.hdf5'
        model_path = PATH+models[i]
        logfile = open(PATH+'/logs/n_spkrs_500_prev_'+str(n_seconds)+'.log', 'w')
        print(model_path, file = logfile)
        np.random.seed(2020)
        ## Training
        print("training")
        if n_seconds == total_seconds[0]:
            train = LibriSpeechDataset(training_set, n_seconds, stochastic=False, pad=True)#, cache=False)
            #train.read_audio_dataset()
            #train.pre_read = True
        else:
            train.fragment_seconds = n_seconds
            train.fragment_length = int(n_seconds * LIBRISPEECH_SAMPLING_RATE)
        
        X_train_files, X_test_files, y_train, y_test = split_files(train)
                
        encoder = load_encoder(model_path, input_length)
        print("Extracting Embedding")
        saved = True
        if not saved: 
            X_read = read_data(X_train_files, train)
            val_read = read_data(X_test_files,train)
            emb = extract_embedding(X_read, encoder, preprocessor)
            val_emb = extract_embedding(val_read, encoder, preprocessor)
            np.save(save_dir+'emb_'+str(n_seconds)+'sec'+str(i)+'.npy',emb)
            np.save(save_dir+'val_emb_'+str(n_seconds)+ 'sec'+str(i)+'.npy',val_emb)
        else:
            print("Loading embedding")
            emb = np.load(save_dir+'emb_'+str(n_seconds)+'sec.npy')
            val_emb = np.load(save_dir+'val_emb_'+str(n_seconds)+ 'sec.npy')
        print("Training classifier")
        saved = True
        if not saved: 
            ## Train Classifier
            clf = svm.SVC(gamma = 'scale', probability=True)
            print(len(emb), len(y_train))
            clf.fit(emb, y_train)
            ## Saving classifier
            with open(save_dir+'clf_'+str(n_seconds)+'sec_r'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(clf, f)
            y_pred = clf.predict(val_emb)
         
            print(accuracy_score(y_test, y_pred), file = logfile) 
            print(accuracy_score(y_test, y_pred)) 

        else:
            with open(save_dir+'clf_'+str(n_seconds)+'sec.pkl', 'rb') as f:
                clf = pickle.load(f)

        run_num_experiment = True
        if run_num_experiment:
            from experiments.test_num_speakers import *
            run_exp_num_speakers(X_train_files, y_train, X_test_files, y_test, emb, val_emb, logfile,n_seconds, num_reps =500)

        logfile.close()
        i+=1
