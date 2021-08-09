# A Robust Hybrid Approach for Textual Document Classification

This repository is an Implemetation for research paper  'A Robust Hybrid Approach for Textual Document Classification'.

### Paper Description : 
Text document classification is an important task for diverse natural language processing based applications. Traditional machine learning approaches mainly focused on reducing dimensionality of textual data to perform classification. This although improved the overall classification accuracy, the classifiers still faced sparsity problem due to lack of better data representation techniques. Deep learning based text document classification, on the other hand, benefitted greatly from the invention of word embeddings that have solved the sparsity problem and researchers focus mainly remained on the development of deep architectures. Deeper architectures, however, learn some redundant features that limit the performance of deep learning based solutions. In this paper, we propose a two stage text document classification methodology which combines traditional feature engineering with automatic feature engineering (using deep learning). The proposed methodology comprises a filter based feature selection (FSE) algorithm followed by a deep convolutional neural network.


Further information about the paper and filter used for selection of important words from sentences is available in the paper given this repository. Please got through the same.


### Code/File Explanation : 

##### training_config_data.json : 
This is a configuration file which having all variables which need to alter based on the personal requirement during stagesl like dataset creation, model training and model testing.
This configuration file having different key values pairs. Here i given the example values for respect key and uses of the same.

######  key :  values   >>>>  Description

###### "data_folder_path" :&nbsp;&nbsp;&nbsp;&nbsp; "C:/Users/Bharathraj C L/Downloads/paper to implement/robust/merge_data" &nbsp;&nbsp;&nbsp;&nbsp; >>>> &nbsp;&nbsp;&nbsp;&nbsp; File path for folder which contains all text files in .txt format

###### "base_file_path" :&nbsp;&nbsp;&nbsp;&nbsp; "C:/Users/Bharathraj C L/Downloads/paper to implement/robust/data_all.csv", &nbsp;&nbsp;&nbsp;&nbsp; >>>> &nbsp;&nbsp;&nbsp;&nbsp;  File path for .csv file which contains all file name with respect to the labels, which will used in the training process

###### "epochs" :&nbsp;&nbsp;&nbsp;&nbsp;  1,  &nbsp;&nbsp;&nbsp;&nbsp; >>>>   &nbsp;&nbsp;&nbsp;&nbsp;   Number of epochs for model training

###### "raw_dataset_folder_path" :&nbsp;&nbsp;&nbsp;&nbsp; "./main_dataset/", &nbsp;&nbsp;&nbsp;&nbsp; >>>>  &nbsp;&nbsp;&nbsp;&nbsp;   File path for folder which contains all raw data in the form of tsv file.

###### "batch_size":&nbsp;&nbsp;&nbsp;&nbsp; 100,  &nbsp;&nbsp;&nbsp;&nbsp;  >>>>  &nbsp;&nbsp;&nbsp;&nbsp;  Number of rows per batchs

###### "number_raw_file_per_labels" :&nbsp;&nbsp;&nbsp;&nbsp; 1000, &nbsp;&nbsp;&nbsp;&nbsp;   >>>>  &nbsp;&nbsp;&nbsp;&nbsp;  Number of rows selected for the training process

###### "model1_path":&nbsp;&nbsp;&nbsp;&nbsp; "save_model1.h5", &nbsp;&nbsp;&nbsp;&nbsp;  >>>> &nbsp;&nbsp;&nbsp;&nbsp;  model1 is saved in this file  path

###### "model2_path":&nbsp;&nbsp;&nbsp;&nbsp; "save_model2.h5", &nbsp;&nbsp;&nbsp;&nbsp;  >>>> &nbsp;&nbsp;&nbsp;&nbsp; model2 is saved in this file path

###### "model3_path":&nbsp;&nbsp;&nbsp;&nbsp; "save_model3.h5", &nbsp;&nbsp;&nbsp;&nbsp;  >>>> &nbsp;&nbsp;&nbsp;&nbsp;  model3  is saved this file path

###### "use_saved_model1":&nbsp;&nbsp;&nbsp;&nbsp; false,  &nbsp;&nbsp;&nbsp;&nbsp; >>>>  &nbsp;&nbsp;&nbsp;&nbsp; It gives option select model either from buid from scractch or load the saved model for model1   

###### "use_saved_model2":&nbsp;&nbsp;&nbsp;&nbsp; false,  &nbsp;&nbsp;&nbsp;&nbsp; >>>> &nbsp;&nbsp;&nbsp;&nbsp;  It gives option select model either from buid from scractch or load the saved model for model2

###### "use_saved_model3":&nbsp;&nbsp;&nbsp;&nbsp; false,  &nbsp;&nbsp;&nbsp;&nbsp; >>>> &nbsp;&nbsp;&nbsp;&nbsp;  It gives option select model either from buid from scractch or load the saved model for model3

###### "tokenizer_path":&nbsp;&nbsp;&nbsp;&nbsp; "tokenizer.pkl",  &nbsp;&nbsp;&nbsp;&nbsp;  >>>> &nbsp;&nbsp;&nbsp;&nbsp;  File path for saving tokenizer in pkl format

###### "labelencoder_path": &nbsp;&nbsp;&nbsp;&nbsp; "labelencoder.pkl",  &nbsp;&nbsp;&nbsp;&nbsp; >>>> &nbsp;&nbsp;&nbsp;&nbsp; File path for saving label encoder in pkl format

###### "preprocessed_dataset_training": &nbsp;&nbsp;&nbsp;&nbsp; "train_data.pkl", &nbsp;&nbsp;&nbsp;&nbsp;  >>>>  &nbsp;&nbsp;&nbsp;&nbsp; Preprocessed data with list of sentences and labels in pkl format for training process

###### "preprocessed_dataset_testing":&nbsp;&nbsp;&nbsp;&nbsp; "train_data.pkl",  &nbsp;&nbsp;&nbsp;&nbsp; >>>> &nbsp;&nbsp;&nbsp;&nbsp;  Preprocessed data with list of sentenecs and labels in pkl format for testing process

###### "select_model": &nbsp;&nbsp;&nbsp;&nbsp;  1    &nbsp;&nbsp;&nbsp;&nbsp;  >>>>  &nbsp;&nbsp;&nbsp;&nbsp;  It gives option to select particular model out of 3, For model1 -> 1,model2 -> 2,model3 -> 3

###### "length": &nbsp;&nbsp;&nbsp;&nbsp; 10000  &nbsp;&nbsp;&nbsp;&nbsp;>>>>  &nbsp;&nbsp;&nbsp;&nbsp; Maximum number of words in sentences are considered.


##### dataset_creation.py : 
This file contains all functions which will be used for dataset creation in .txt format using raw dataset .tsv file, which is downloaded from internet.



##### ndm_code.py :
This file contains all functions which will be used for filtering process which creating preprocessed dataset in .pkl file from .txt files.



##### with_ndm_dataset.py :
This file is used to initiate the filtering process and creation of both training and testing dataset in .pkl file format.


##### model_build.py :
This file contains all functions for defining  model architecture.

##### model_train.py :
This file contains all functions for declaring neural network model, loading training file, initiating training process and saving trained model at the end.


##### model_test.py :
This file contains all functions for model evaluation with the consideration of models and testing dataset.

##### util_code.py 
This file contains all essential functions which will be using in different process.


###  How to Use :

##### For Dataset creation :
Please follow the steps given below : \
Load file in to a folder and update that folder path in key of 'raw_dataset_folder_path' in training_config_data.json file.\
create an empty folder and update that folder path in key of 'data_folder_path' in training_config_data.json file.\
Update csv file name with path in key of 'base_file_path' in training_config_data.json file, where we need to save the particular file.\
execute the dataset_creation.py file. It will create the dataset in .txt format and respective csv file.


##### For preprocessed dataset with filtering process :
Please follow the steps given below :\
Update .pkl file path for both training and testing  files in training_config_data.json file.\
execute the with_ndm_dataset.py file. It will create the dataset in .pkl format for training and testing dataset with consideration of all preprocessing steps.


##### For training process :
Please follow the steps given below :\
Update all key:values in training_config_data.json file which related to the model selection, model usage and model training purpose.\
execute the model_train.py file. It will initiate the training process and saves the model, once model training completes.

##### For testing process :
Please follow the steps given below :\
Update all key:values in training_config_data.json file which related to the model selection, model usage and model testing purpose.\
execute the model_test.py file. It will initiate the testing process and display the model accuracy.
