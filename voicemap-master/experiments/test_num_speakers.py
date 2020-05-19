from sklearn.metrics import accuracy_score
from test_pipeline import *
import numpy as np

def num_speakers_train(X_train_files, y_train, two_spkr, emb=None):
    X_two_train =[]
    y_two_train =[]
    if emb is None:
        spkr1=X_train_files[y_train == two_spkr[0]]
        spkr1_emb = extract_embedding(spkr1, train)
    else:
        spkr1_emb = emb[y_train == two_spkr[0]]
    X_two_train = spkr1_emb
    y_spkr1 = np.stack([two_spkr[0]]*len(spkr1_emb))
    y_two_train = y_spkr1
    
    for i in range(1, len(two_spkr)):
        if emb is None:
            spkr1=X_train_files[y_train == two_spkr[i]]
            spkr1_emb = extract_embedding(spkr1, train)
        else:
            spkr1_emb = emb[y_train == two_spkr[i]]
        y_spkr1 = np.stack([two_spkr[i]]*len(spkr1_emb))
        X_two_train = np.vstack((X_two_train,spkr1_emb))
        y_two_train = np.hstack((y_two_train,y_spkr1))
    
    return (X_two_train, y_two_train)

def num_speakers_val(X_test_files, y_test, two_spkr, val_emb = None):
    X_two_val =[]
    y_two_val =[]
    if val_emb is None:
        spkr1=X_test_files[y_test == two_spkr[0]]
        spkr1_emb = extract_embedding(spkr1, train)
    else:
        spkr1_emb = val_emb[y_test == two_spkr[0]]
    X_two_val = spkr1_emb
    y_spkr1 = np.stack([two_spkr[0]]*len(spkr1_emb))
    y_two_val = y_spkr1

    for i in range(1, len(two_spkr)):
        if val_emb is None:
            spkr1=X_test_files[y_test == two_spkr[i]]
            spkr1_emb = extract_embedding(spkr1, train)
        else:
            spkr1_emb = val_emb[y_test == two_spkr[i]]
        y_spkr1 = np.stack([two_spkr[i]]*len(spkr1_emb))
        X_two_val = np.vstack((X_two_val,spkr1_emb))
        y_two_val = np.hstack((y_two_val,y_spkr1))
    
    return (X_two_val, y_two_val)

def run_exp_num_speakers(X_train_files, y_train, X_test_files, y_test, emb, val_emb,  logfile, num_reps =100):
    speaker_range = [2, 5, 10, 50, 100]

    total_acc_list =[]
    for num_speakers in speaker_range:
        acc=[]
        for i in range(num_reps):
            two_spkr = np.random.choice(np.unique(y_train),num_speakers, replace = False)
            #print("Trial:", i)
            #, "Speakers:", two_spkr)
            X_two_train, y_two_train = num_speakers_train(X_train_files, y_train, two_spkr, emb)
            clf2 = svm.SVC(gamma = 'scale')
            clf2.fit(X_two_train, y_two_train) 

            X_two_val, y_two_val = num_speakers_val(X_test_files, y_test, two_spkr, val_emb)
            y_pred = clf2.predict(X_two_val)
            y_val = y_two_val
            acc.append(accuracy_score(y_val, y_pred))
            #print("Accuracy:",acc[i])
        print("Num Speakers:", num_speakers, "Accuracy: mean", np.round(np.mean(np.array(acc)),4),"std:", np.round(np.std(np.array(acc)),4), file = logfile)
        print("Num Speakers:", num_speakers, "Accuracy: mean", np.round(np.mean(np.array(acc)),4),"std:", np.round(np.std(np.array(acc)),4))
        total_acc_list.append(acc)
        np.save('total_acc_list_'+str(num_speakers), total_acc_list)

