## Speaker Identification 

#### Code Structure:

- Voicemap:
     Contains dataloader class and model structure definitions and utilities
- Experiments:
     Contains main code which uses the defined classes above for the required task
     - train_siamese: Trains the encoder using Siamese network
     - n_seconds_accuracy: Trains multiple encoder depending on input window lengths
     - test_num_speakers: Defines experiment with varying number of speakers for each model
     - test_pipeline: Trains an SVM classifier and experiments with varying number of speakers
     - plot_energy: Data Visualisation of energy of selected audio sample
- Notebooks:
     Contains jupyter notebooks for visualisation of embeddings, creating required plots from the results and additional experiments
     

Some parts of the codebase is a modified version of the original codebase from 
https://github.com/oscarknagg/voicemap
