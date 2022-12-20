# Introductory Remarks

The current version of this program is preliminary and intended for my NLP students at NYU.

Before trying this baseline system, download baselinedata.zip from the shared data directory
in google drive.  Unzip this file in the same directory as this README.  This material is subject
to licensing restrictions.

# Collaboration
Each team working on this project will have a branch allocated on this repository.  
Please work only on your respective branches. 
The branch name will be of the format `<team_number>-<student1-netid>-<student2-netid>-...`.  
The main brach containing the baseline system will be locked (read-only). 

Please ensure that you checkout your respective branches and work (adding your own code, or updating the baseline code)
Obviously you can revamp the code completely in your own group's branch.  

By the end of the project, have a README in your branch letting us know how to run your code. Add all the information such as package dependencies.  
We will be running/testing your models on google colab (which allows us to support a wide variety of pre-requisites and hadrware such as Multi-GPU, High-RAM etc.).  

# Task Description

Our main objective is to create a model that can classify/identify ARG1 given an input sentence
There are 3 main subdirectories in this folder corresponding to the different tasks at hand, 
1. percentage  
2. part  
3. all_nombank  

# Understanding directory contents  
Each of these subdirectories will inturn have a total of 6 files. 
1. `train.data` - This is the primary input file that is used for creating the embeddings, features and training the model  
2. `dev.data` - This is the dev set/ validation set to test the performace of the model on  
3. `test.data` - This is the testing data  
4. `train.features` - These are the features created to be provided to the model while training. Thus the input to out model will be of the shape `X[: trainFeatures]`  
5. `dev.features`  - These are the features created to be provided to the model while testing.  
6. `train.embeddings` - These files are all connected with predicting ARG1 and sometimes this includes calculating and storing an average value for ARG1 embeddings. These values are saved and reloaded when we run feature creation multiple times.  

# Steps to perform the task  
1. We would want to create the `.features` files from the given data. To do this, we use the `create_embedding_and_features.py` code.    
    - The code snippet that was used to call and run this file is  
    ```
    import create_embedding_and_features
    create_embedding_and_features.make_percent_train_feature_file()
    ```  
    - The second command can be replaced with the corresponding command for your task e.g. `create_embedding_and_features.make_part_dev_feature_file()`  
    - This can help you find the entry point of the code and trace back to understand what is happening
    - Functions have been created only for percentage and partitive tasks. Creating your own function for the other nombank tasks is very easy. Look at the code snippet [here](https://github.com/TAbishekS/Nombank-Tasks-Baseline-System/blob/main/create_embedding_and_features.py#L1456). You can reuse this code snippet, change the directory name (to your corresponding nombank taks) and run the baseline system. Please note that the baseline system was only tested on the percentage, partitve and all nombank datasets. 
    - Remember that creating embedding files is a computationally intensive task. To test if the creation of the embedding files work, try with a smaller input dataset.  
2. Using these `.features` files, we would want to create and train a model. To do this, we will use the `build_model.py` code.  
    - The code snippet that was used to create and train the the model is: 
    ```
    import build_model
    build_model.train_and_run("percentage", ada=True)
    ```  
    - Again, the second command can be replaced with the corresponding command for your task e.g. `build_model.train_and_run("part", ada=True)`  
    - When the model is built and trained, a `classifier.model` file will be created in the corresponding directory along with an output file called `dev.output` that will contain the words labelled with `arg1`

To visually represent the different steps in our task:  

Input File -> Features and Embeddings -> Build and train Model -> `dev.output` file -> score Model.  

# Extra files and other instructions  
1. When running the `create_embedding_and_features.py` file which uses `spacy`, we may run into an error, as the `en_core_web_md` model that this uses may not be downloaded yet. To get around this problem, after installing `spacy` using `pip` or `conda`, run the command: `python -m spacy download en_core_web_md`  
2. There are other files in the dependency folder, that are used in the `create_embedding_and_features.py` and the `build_model.py` files, so please ensure that you do not delete the folder.  
3. The ML model that is built and scored (precision,recall,f-measure) against the answer key.  An analysis and the results are reported in the talk: `https://cs.nyu.edu/courses/fall22/CSCI-GA.2590-001/priv/partitive_talk.pdf`  

# Questions to think about:  

1. Are there better ways to handle embedding features?  How and why?

2. Some students in previous semester used embeddings of the whole
sentences to predict classes.  Can this strategy be implemented in a
way such that the results can be ensembled with the approach assumed
here?

3. How would you handle path features from a treebank or a parser? How
much better or worse are the results than the chunk-based method
implemented for the initial baseline? Why?

4. In addition to representing the path features directly, it may also
be possible to use path-based embeddings.  Is this effective?

5. What additional features yield positive results.

6. Are there other parameters that need to be tuned and if so, how?

7. Can you find principled reasons why particular ML algorithms
improve results?  Are there some more sophisticated algorithms that do
not improve the results? What are some limiting factors?

8. Initial experiments on the baseline system focussed on a small
subset of the tasks for which data is provided: predicting support;
identifying ARG1s of other classes of nouns; identifying other
arguments. It may be worthwhile to run systems on some of the other
tasks.  Comparisons may lead to insights.

9. The baseline system does poorly for "all ARG1" task.  An analsysis
of errors could lead to a better system or, perhaps, a better
understanding.

10. The use of some of the other data provided may result in better
results.  From Google Drive, the following additional files may be
helpful:

* WSJ_CHUNK_FILES.zip -- may provide evidence of random co-occurances,
   that can be compared to occurances of arguments

* Parse Tree and Treebank Data -- (gold) treebank trees and (nongold) parse trees for all Nombank tasks.

11. So far these questions, have assumed tasks with mostly gold data.
However, experiments with non-gold data may suggest more realistic
results, i.e., results of running a system on raw text.


