import sklearn
from sklearn.linear_model import LogisticRegression, SGDClassifier  ## LogisticRegression is the same as Maximum Entropy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os
import re
import numpy as np
from tools.get_part_class_list_training import *
from tools.feature_lists import *
from tools.arg_scorer import *


word_class_dict = make_all_class_dict()

monocase = True

## We assumes the following for feature files
## 1) features are divided by tabs
## 2) The system will predict a single feature, which we will list
## 3) That feature is always in the last column (other "competing" features
##      could be in the last column as well
## 4) Each line corresponds to a token, except for blank lines which represent sentence
## breaks

def get_feature_dictionary (instream,answer_feature):
    ## The distinction of feature vs answer does not matter
    ## for the construction of the dictionary
    current_number = 2
    feature_dict = {}
    not_answer = 'not_'+answer_feature
    ## feature_dict['Answer'] = {answer_feature:1,not_answer_feature:0}
    ## hard code 1 and 0
    ## hard code for answer feature
    line_number = 0
    for line in instream:
        if monocase:
            line = line.lower()
        line = line.strip(os.linesep)
        feature_list = line.split('\t')
        column_number = 0
        for feature_value in feature_list:
            if feature_value.lower() == answer_feature.lower():
                feature_value = feature_value.lower()
            if feature_value == answer_feature:
                continue ## this is already taken care of
            elif column_number == 0:
                feature = 'word'
                value = feature_value
                if monocase:
                    value = value.lower()
                column_number = 1 ## really just first column vs not first column
            elif feature_value.startswith('arg') and (not '=' in feature_value):
                continue ## ignore other arguments
            elif not '=' in feature_value:
                print('bad feature value:', feature_value)
                input('Pause1 -- check data')
            else:
                feature,value = feature_value.split('=')
                feature = feature.lower()
            if (feature in feature_dict) and (not feature in numeric_valued_features):
                child_dict = feature_dict[feature]
                ## 57
                if value in child_dict:
                    pass
                else:
                    child_dict[value] = current_number
                    current_number += 1
            elif (not feature in numeric_valued_features):
                child_dict = {}
                feature_dict[feature] = child_dict
                child_dict[value]=current_number
                current_number +=1
    return(feature_dict)

def number_string(value):
    try:
        return(float(value))
    except:
        return(False)

def get_key_and_value_info (instream,answer_feature,use_OOV=False,minimum_word_frequency = 2,\
                            minimum_key_frequency=100,minimum_value_frequency=20,convert_class_features=False,trim_values=True):
    ## return key_list and value_dictionary
    global value_frequency
    key_list = []
    value_dictionary = {}
    current_number = 2
    not_answer = 'not_'+answer_feature
    line_number = 0
    word_dict = {}
    value_frequency = {}
    key_frequency = {}
    for line in instream:
        if monocase:
            line = line.lower()
        line = line.strip(os.linesep)
        feature_list = line.split('\t')
        column_number = 0
        for feature_value in feature_list:
            # if re.search('ARG0',feature_value,re.I):
            #     print('*'+feature_value+'*')
            #     input('break')
            if feature_value.lower()  in [answer_feature, not_answer]:
                continue ## this is already taken care of
            elif column_number == 0:
                feature = 'word'
                if monocase:
                    value = feature_value.lower()
                else:
                    value = feature_value
                if value in word_dict:
                    word_dict[value]+= 1
                else:
                    word_dict[value] = 1
                column_number = 1 ## really just first column vs not first column
            elif feature_value.startswith('arg') and (not '=' in feature_value):
                continue  ## ignore other arguments
            elif not '=' in feature_value:
                print('bad feature value:',  feature_value)
                print('?',answer_feature)
                input('Pause1 -- check data')
            else:
                feature,value = feature_value.split('=')
                if convert_class_features and (feature.upper() in  word_class_dict):
                    feature = word_class_dict[feature.upper()]
                if monocase:
                    feature = feature.lower()
            if not feature in key_list:
                key_list.append(feature)
                key_frequency[feature]=1
            else:
                key_frequency[feature]+=1
            if number_string(value):
                pass
            elif not value in value_dictionary:
                value_dictionary[value] = current_number
                current_number += 1
                value_frequency[value] = 1
            else:
                value_frequency[value] += 1
    if use_OOV and (len(word_dict) > 0):
        value_dictionary['oov'] = current_number
        current_number += 1
        value_list = list(value_dictionary.keys())
        oovs = 0
        for value in value_list:
            if (value in word_dict) and (word_dict[value] < minimum_word_frequency):
                oovs += word_dict[value]
                value_dictionary.pop(value)
    if minimum_value_frequency > 0:
        for value in value_frequency.keys():
            if (value in value_dictionary) and (value_frequency[value] < minimum_value_frequency):
                value_dictionary.pop(value)
    if minimum_key_frequency > 0:
        new_key_list = []
        for key in key_list:
            if key_frequency[key] >= minimum_key_frequency:
                new_key_list.append(key)
        key_list = new_key_list
    if use_OOV:
        print(oovs, 'oovs')

    # Re-index the dict
    new_idx = 0
    _value_dictionary = dict()
    for k in value_dictionary.keys():
        _value_dictionary[k] = new_idx
        new_idx += 1
    value_dictionary = _value_dictionary

    return(key_list,value_dictionary)

def get_features_from_file(instream,key_list,value_dictionary,answer_feature,OOV,predict=False, convert_class_features=False):
    print('There are',len(key_list), 'keys')
##    input('pause again')
    not_feature = 'not_'+answer_feature
    pos_list = []
    out_dictionaries = []
    output_vectors =[]  ## evidence = list of lists of features
    output_features = [] ## answer = list of numbers
    words = [] ## a list of words (for predict onlly
    trace = True
    for line in instream:
        if monocase:
            line = line.lower()
        little_dictionary = {}
        little_list = []
        output_vectors.append(little_list)
        if re.match('^[ \t\r]*$',line):
            ## output_features.append(0)
            answer = 0
            if predict:
                words.append('')
                pos_list.append('')
            for num in range(len(key_list)):
                little_list.append(0)
        else:
            line = line.strip(os.linesep)
            key_values = line.split('\t')
            first_column = True
            answer = 0
            ## input('pause')
            print_on = False
            if 'arg1' in key_values:
                pass
                ## print_on = True
            for key_value in key_values:
                if print_on:
                    print(key_value)
                if (not predict) and (key_value.lower() == answer_feature.lower()):
                    ## 57 ##
                    ## print('ARG1')
                    answer = 1
                    ## input('pause')
                elif predict and (key_value.lower() == answer_feature.lower()):
                    continue ## ignore answer for purposes of running system (or delete it from input file)
                elif first_column:
                    feature = 'word'
                    word = key_value
                    if monocase:
                        value = key_value.lower()
                    else:
                        value = key_value
                    first_column = False
                    ## if feature in numeric_valued_features:
                    if value.upper() == 'RIGHT_NP_PP_NP':
                        print(feature,value)
                        # print(feature,value)
                        input('pause')
                    if number_string(value):
                        little_dictionary[feature]=float(value) ## for numeric values, use float of actual value
                    elif value in value_dictionary:
                        little_dictionary[feature]=value_dictionary[value]
                    elif OOV and (feature in baseline1_word_features):
                        little_dictionary[feature]=value_dictionary['oov'] ## handle OOV cases for word
                        # print('OOV',feature,value)
                        # input('pause')
                    else:
                        little_dictionary[feature]=0
                    if predict:
                        words.append(word)
                    ## output_features.append(0)  ## also not an arg1
                elif key_value.startswith('arg') and (not '=' in key_value):
                    continue  ## ignore other arguments
                elif not '=' in key_value:
                    print('bad feature value:',key_value)
                    input('Pause -- check data')
                else:
                    feature,value = key_value.split('=')
                    if convert_class_features and (feature.upper() in  word_class_dict):
                        feature = word_class_dict[feature.upper()]
                    if monocase:
                        value = value.lower()
                        feature = feature.lower()
                    if feature in ['POS','pos']:
                        pos_list.append(value)
                    ## if (feature in numeric_valued_features) or
                    if number_string(value) :
                        little_dictionary[feature]=float(value)
                    elif OOV and (feature in baseline1_word_features) and (not value in value_dictionary):
                        little_dictionary[feature]=value_dictionary['oov'] 
                    elif not value  in value_dictionary:
                        little_dictionary[feature]=0
                    else:
                        little_dictionary[feature]=value_dictionary[value]
            ## turn little_dictionary into little list
            for key in key_list:
                if key in little_dictionary:
                    value = little_dictionary[key]
                    little_list.append(value)
                else:
                    little_list.append(0)
        output_features.append(answer)
    try:
        output_vectors = np.array(output_vectors)
        output_features = np.array(output_features)
    except:
        print(1,output_vectors[0])
        print(2,output_features)
        raise(Exception('blah2'))
    if predict:
        return(words,output_vectors,pos_list)
    else:
        return(output_vectors,output_features,pos_list)



def train_system(feature_file,answer_feature,key_list,value_dictionary,\
                 model_file=False,OOV=False,convert_class_features= False,\
                 max_iter=30000, ada = False):
    global features
    global predictions
    with open(feature_file) as instream:
        features,predictions,pos_list = get_features_from_file(instream,key_list,value_dictionary,\
                                                               answer_feature,OOV,\
                                                               convert_class_features= convert_class_features)
    pipe = None
    if(ada == False):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter,penalty='none'))
    else:
        pipe = make_pipeline(StandardScaler(), AdaBoostClassifier(n_estimators=100, random_state=0))

    print('made pipe')
    ## ignoring pos_list here
    pipe.fit(features,predictions)
    MEclassifier = pipe
    print('fitted')
    if model_file != False:
        pickle.dump([MEclassifier,key_list,value_dictionary],open(model_file,'wb'))
    return(MEclassifier)

def run_classifer(classifer,key_list,value_dictionary,feature_file,answer_feature,outfile,OOV=False,convert_class_features= False):
    ## without scaling
    with open(feature_file) as instream:
        words,features,pos_list = get_features_from_file(instream,key_list,value_dictionary,answer_feature,OOV,predict=True,convert_class_features= convert_class_features)
        ## use pos_list for scoring NNPs
        output_features = classifer.predict(features)
        # print(output_features)
        output_features = list(output_features)
        print('how many:',output_features.count(1))
    with open(outfile,'w') as outstream:
        for index in range(len(output_features)):
            current_word = words[index]
            current_pos = pos_list[index]
            current_output_value = output_features[index]
            if current_output_value == 1:
                outstream.write(current_word+'\t'+current_pos+'\t'+answer_feature+'\n')
            elif current_word == '':
                outstream.write('\n')
            else:
                outstream.write(current_word+'\t'+current_pos+'\n')

def prefix_match(prefix,feature):
    if feature.startswith(prefix):
        the_rest = feature[len(prefix):]
        if the_rest.count('_') == 1:
            return(True)
        else:
            return(False)

def test_feature_set(input_features,feature_file,answer_feature,prefix_features):
    with open(feature_file) as instream:
        key_list,value_dict = key_list, value_dict = get_key_and_value_info (instream,answer_feature)
    for index in range(len(input_features)):
        input_features[index] = input_features[index].lower()
    for index in range(len(key_list)):
        key_list[index] = key_list[index].lower()
    input_features_only = []
    file_features_only = []
    for feature in key_list:
        if not feature in input_features:
            file_features_only.append(feature)
    for feature in input_features:
        if not feature in key_list:
            input_features_only.append(feature)
    file_only2 = []
    matched_prefixes =[]
    matched_prefix_features = []
    for feature in file_features_only:
        for prefix in prefix_features:
            if prefix_match(prefix,feature):
                if not prefix in matched_prefixes:
                    matched_prefixes.append(prefix)
                matched_prefix_features.append(feature)
                break
    for feat in file_features_only:
        if feat in matched_prefix_features:
            pass
        else:
            file_only2.append(feat)
    print('Matched prefixes',matched_prefixes)
    print('Input Features Only',input_features_only)
    print('file only',file_only2)

def some_prefix_match(key,prefixes):
    for prefix in prefixes:
        if prefix_match(prefix,key):
            return(True) ## exit loop and return True if any prefix matches
    ## if no prefix matches, return False
    return(False)

def edit_key_list(key_list,keep_features,keep_prefixes):
    keep_features2 =[]
    for index in range(len(keep_features)):
        keep_features2.append(keep_features[index].lower())
    out_list = []
    for key in key_list:
        key = key.lower()
        if (key in keep_features2):
            out_list.append(key)
        elif  keep_prefixes and (len(keep_prefixes)>0) and some_prefix_match(key,keep_prefixes):
            out_list.append(key)
        else:
            pass
    return(out_list)

def run_system(training_feature_file,predict_feature,model_file, system_feature_file,output_file,answer_key,
               edit_feature_list=False,features_to_keep=[],feature_prefixes_to_keep=[],OOV=False,
               convert_class_features=False,minimum_value_frequency=2,minimum_key_frequency=100,score_file=False,max_iter=30000, ada = False
):
    global key_list
    global value_dictionary
    with open(training_feature_file) as instream:
        key_list,value_dictionary = get_key_and_value_info(instream,predict_feature,\
                                                           use_OOV=OOV,\
                                                           convert_class_features= convert_class_features,\
                                                           minimum_key_frequency=minimum_key_frequency,\
                                                           minimum_value_frequency=minimum_value_frequency
        )
    if edit_feature_list:
        key_list = edit_key_list(key_list,features_to_keep,feature_prefixes_to_keep)
    try:
        classifer = train_system(training_feature_file,predict_feature,key_list,value_dictionary,model_file=model_file,OOV=OOV,convert_class_features= convert_class_features,max_iter=max_iter, ada=ada)
        run_classifer(classifer,key_list,value_dictionary,system_feature_file,predict_feature,output_file,OOV=OOV,convert_class_features= convert_class_features)
        score_file_with_NNP_adjustment(answer_key,output_file,predict_feature,score_file=score_file)
    except:
        print('Features are insufficiently predictive. Nothing is predicted.')

def get_feature_list_from_file(feature_file,predict_feature):
    with open(feature_file) as instream:
        feature_dictionary = get_feature_dictionary(instream,predict_feature)
        keys = list(feature_dictionary.keys())
        keys.sort()
        return(keys)

def train_and_run(folder, ada=False):
    run_system(os.path.join(folder, "train.features"),'arg1',os.path.join(folder, "classifier.model"),os.path.join(folder, "dev.features"),os.path.join(folder, "dev.output"),os.path.join(folder, "dev.data"),OOV=True,ada=ada)
