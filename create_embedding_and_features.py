## Version 4 -- update embedding features for sklearn

## Add features based on word similarity
## 1) use en_core_web_md  -- which uses tok_to_vec on a medium sized corpus (43 mb)
##          -- en_core_web_sm and en_core_web_lg also use tok_to_vec (on 12 and 741 mb)
##          -- en_core_web_trf  -- uses "roberta" based transformer on 438 mb
##     -- later add this difference as a parameter
## 2) For the training corpus create the following average vectors63
##     a) the average of all ARG1s
##     b) the average of all bigrams ending in ARG1
##     c) the average of all trigrams ending in ARG1
##     d and e -- the average of all bigrams and trigrams starting with ARG1
## 3) Create features, based on the values of 2 derived from the training corpus
##     a)  For each word, bigram and trigram in each sentence, rank the similarities with the ARG1 version
##     b)  For all corpora (training/dev/test), these rankings are based on comparisons with the training
##     c) these rankings are features of the form X or greater, e.g., 3rd includes 1st and 2nd

import re
import os
import spacy  ## not gensim yet
import numpy as np

spacy_nlp = spacy.load('en_core_web_md')  ## or other embedding model

sentence_break_features = {'word':'SENTENCE_BREAK','POS':'SENTENCE_BREAK','BIO':'SENTENCE_BREAK'}

look_behind = 2
look_ahead = 2

def all_zeros(vector):
    all_zeros = True
    for zero in vector:
        if zero != 0:
            return(False)
    return(True)

def stringify_similarity(score):
    ## similarity is a value between 0 and 1
    ## I will assume 5 decimal places
    return(str(round(score,5)))

def cosine_similarity_vector_and_array(vector,array):
    ## need to deal with OOV
    ## ** 57 **
    if (vector == 'SENTENCE_BREAK') or all_zeros(vector.vector) or all_zeros(array):
        return(0)
    array2 = vector.vector
    numerator = float(sum(array * array2))
    denominator = float((sum(array**2)*sum(array2**2))**(.5))
    try:
        if (denominator  > float('-inf')) and (denominator < float('inf')):
            output = numerator/denominator
            return(output)
        else:
            print('Warning Near 0 or infinity denominator for:',vector.vector,array)
            return(0)
    except:
        print(1,array)
        print(2,array2)
        print(2.5,vector)
        print('Does it match',vector=='SENTENCE_BREAK')
        input('pause')
        print('lengths:',len(array2),len(array))
        print('numerator',numerator)
        print('denominator',denominator)
        return(0)


def load_stemming_dictionary(infile):
    stem_dict = {}
    with open(infile) as instream:
            ## assume .csv file
            ## first field is lemma and remaining fields are possible base values
        for line in instream:
            line = line.strip(os.linesep)
            outlist = line.lower().split(',')
            word = outlist[0]
            if word in stem_dict:
                entry = stem_dict[word]
                for base in outlist[1:]:
                    if not base in entry:
                        entry[word].append(base)
            else:
                stem_dict[word] = outlist[1:]
    return(stem_dict)

stemming_dict = load_stemming_dictionary(os.path.join('tools/morph-base-mapping.csv'))

def get_stemmed_word(word):
    global stemming_dict
    lower_word = word.lower()
    if lower_word in stemming_dict:
        return(stemming_dict[lower_word][0])
    else:
        return(lower_word)

def normalize_partitive_class(feature):
    if feature == 'METONYM':
        return('MERONYM')  ## probably typo
    elif feature == 'INSTANCE':
        return('INSTANCE-OF-SET')
    elif feature.startswith('NOMADJ'):
        return('NOMADJ')
    elif feature.startswith('NOMLIKE'):
        return('NOM')
    elif feature == 'BODY-PART':
        return('PART-OF-BODY-FURNITURE-ETC')
    elif 'feature' in ['BOOK-CHAPTER','BORDER','DIVISION']:
        return('MERONYM')
    else:
        return(feature)

def get_pred_class_features(features_string):
    features = features_string.split('/')
    established_features = []
    additional_features = []
    for feature in features:
        feature = normalize_partitive_class(feature)
        if (feature in ['PARTITIVE-QUANT', 'PARTITIVE-PART','MERONYM','GROUP','SHARE']) \
           and (not feature in established_features):
            established_features.append(feature)
        else:
            additional_features.append(feature)
    return(established_features,additional_features)
## adjust code  ** 57 ***


def get_features_using_history(output,line_features,is_test_data,prediction):
    ## output is a list of digested sets of features for previous words
    ## line_features is an undigested list of features for the current word 
    ## print(output)
    ## column 0 = word
    ## column 1 = POS
    ## column 2 = chunk tag (ending in head)  
    ## column 3 = token number (used for distance)
    ## column 4 = sentence number
    ## column 5 = role label: Pred, Support, ARG0, ARG1, ARG2, ARG3
    ##                    -- use for distance/etc. between ARG1 and support or distance between
    ##                        ARG1 and Pred
    ## column 6 = Predicate class
    ## column 7 (training only) -- PRED, SUPPORT,ARG1, NOT-ARG (only last in NG), NOTHING
    pred_features = []
    feature_dict = {}
    feature_dict['word'] = line_features[0]
    feature_dict['POS'] = line_features[1]
    feature_dict['BIO'] = line_features[2]
    feature_dict['token_number'] = line_features[3]
    feature_dict['sentence_number'] = line_features[4]
    ## print(line_features[4])
    feature_dict['stemmed_word'] = get_stemmed_word(line_features[0])
    ## token_number/sentence_number used by other parts of program
    if len(line_features) > 5:
            relation_feature = line_features[5]
            ## if (relation_feature in ['PRED','SUPPORT']):
            feature_dict['relation_feature'] = relation_feature
    if len(line_features) > 6:
        established_pred_class_features,extra_features = get_pred_class_features(line_features[6])
        for feature in established_pred_class_features:
            feature_dict[feature]='True'
        for feature in extra_features:
            print(feature)
            ## fix here 57 
            ## input('pause')
            feature_dict[feature]='True'
        pred_features = established_pred_class_features+extra_features 
    for num in range(1,look_behind+1):
        word_feature = 'word_back_'+str(num)
        stemmed_word_feature = 'stemmed_word_back_'+str(num)
        POS_feature = 'POS_back_'+str(num)
        BIO_feature = 'BIO_back_'+str(num)
        REL_feature = 'REL_back_'+str(num)
        REL_stemmed = 'stemmed_REL_back_'+str(num)
        if len(output)>=num:
            old_feature_dict = output[-1*num]
            feature_dict[word_feature]=old_feature_dict['word']
            if old_feature_dict['word'] != 'SENTENCE_BREAK':
                feature_dict[stemmed_word_feature] = old_feature_dict['stemmed_word']
            feature_dict[POS_feature]=old_feature_dict['POS']
            feature_dict[BIO_feature]=old_feature_dict['BIO']
            if 'relation_feature' in old_feature_dict:
                    rel_feature = old_feature_dict['relation_feature']
                    if (rel_feature in ['PRED','SUPPORT']):
                            feature_dict[REL_feature] = rel_feature
        else:
            feature_dict[word_feature]='SENTENCE_BREAK'
            feature_dict[POS_feature]='SENTENCE_BREAK'
            if not is_test_data:
                feature_dict[BIO_feature]='SENTENCE_BREAK'
    # for num in range(1,look_ahead+1):    ## ** 57 only for sentence_break
    #     ## changed from 1 to look_head + 
    #     ## initialize look_ahead features so they default to sentence break
    #     ## they will be reset only if words follow
    #     word_feature = 'word_plus_'+str(num)
    #     stemmed_word_feature = 'stemmed_word_plus_'+str(num)
    #     POS_feature = 'POS_plus_'+str(num)
    #     feature_dict[word_feature]='SENTENCE_BREAK'
    #     feature_dict[stemmed_word_feature]='SENTENCE_BREAK'
    #     feature_dict[POS_feature]='SENTENCE_BREAK'
    return(feature_dict,pred_features)

def update_based_on_forward_context(output,line_features,is_test_data):
    for num in range(1, look_ahead+2):
        stemmed_word_feature = 'stemmed_word_plus_'+str(num)
        POS_feature = 'POS_plus_'+str(num)
        BIO_feature = 'BIO_plus_'+str(num)
        REL_feature = 'REL_plus_'+str(num)
        word_feature = 'word_plus_'+str(num)
        if len(output) < num:
            pass
        else:
        # feature_set = output[-1*num]
        #     feature_set[word_feature]='SENTENCE_BREAK'
        #     feature_set[stemmed_word_feature]='SENTENCE_BREAK'
        #     feature_set[POS_feature]='SENTENCE_BREAK'
        #     feature_set[BIO_feature]='SENTENCE_BREAK'
        # else:
            feature_set = output[-1*num]
            feature_set[word_feature]=line_features[0]
            feature_set[stemmed_word_feature]=get_stemmed_word(line_features[0])            
            feature_set[POS_feature]=line_features[1]
            feature_set[BIO_feature]=line_features[2]
            if len(line_features) > 5:
                relation_feature = line_features[5]
                if relation_feature in ['PRED','SUPPORT']:
                    feature_set[REL_feature] = relation_feature

def update_forward_context_sentence_break_features(current_sentence):
    ## sentence_break features to last few tokens of current_sentence
    for num in range(1,look_ahead+2):
        if len(current_sentence)>num:
            index = -1*num
            stemmed_word_feature = 'stemmed_word_plus_'+str(num)
            POS_feature = 'POS_plus_'+str(num)
            BIO_feature = 'BIO_plus_'+str(num)
            ## REL_feature = 'REL_plus_'+str(num) -- not sure
            word_feature = 'word_plus_'+str(num)
            feature_set = current_sentence[index]
            for feature in [stemmed_word_feature,POS_feature,BIO_feature,word_feature]:
                feature_set[feature] = 'SENTENCE_BREAK'


def get_feature_string(feature_dict,is_test_data,prediction):
    unused_features = ['sentence_number','token_number']
    if ('word' in feature_dict) and (feature_dict['word']=='SENTENCE_BREAK'):
        return(os.linesep)
    else:
        out_string = feature_dict['word']+'\t'
    for feature in feature_dict:
        value = feature_dict[feature]
        if type(value) == bool:
            print(feature,value)
            input('break')
        if (feature == 'word') or (feature in unused_features):
            pass
        elif (feature == 'relation_feature') and (value == prediction):
            pass
        else:
            if type(value) in [float, int]:
                value = str(value)
            out_string=out_string+feature+'='+value+'\t'
    out_string = out_string[:-1]
    if (not is_test_data) and ('relation_feature'  in feature_dict):
        value = feature_dict['relation_feature']
        if value == prediction:
            out_string = out_string+'\t'+value
    out_string = out_string+os.linesep
    return(out_string)

def add_interval_features(output):
    token_number = 0
    current_sentence = 0
    sentence_start = 0
    next_position = 0
    preposition_position = 'NONE'
    conjunction_position = 'NONE'
    current_conjunction = 'NONE'
    pred_position = 'NONE'
    support_position = 'NONE'
    current_preposition = 'NONE' 
    for featureset in output:
        if 'sentence_number' in featureset:
            current_sentence_number = int(featureset['sentence_number'])
            current_token_number = int(featureset['token_number'])
            if current_sentence_number != current_sentence:
                preposition_position = 'NONE'
                pred_position = 'NONE'
                support_position = 'NONE'
                conjunction_position = 'NONE'
                current_conjunction = 'NONE'
                sentence_start = next_position
                current_token_number = next_position
                current_sentence = current_sentence_number
            else:
                current_token_number = next_position
            if current_token_number != next_position:
                print(current_token_number,"does not equal",next_position)
                input('pause')
            if ('POS' in featureset) and (featureset['POS'] == 'CC') and (featureset['word'].lower() in ['or','and']):
                for back in range(1,4):
                    if len(previous_featuresets)>= back:
                        previous_feat =  previous_featuresets[-1*back]
                        if 'HEAD-NP' in previous_feat:
                            if back > 1:
                                feature_name = str(back)+'_or_Less_before_conj'
                            else:
                                feature_name = str(back)+'_before_conj'
                            previous_feat[feature_name]='True'     
            if  'relation_feature' in featureset:
                rel_feature = featureset['relation_feature']
                previous_featuresets = output[sentence_start:next_position]
                if rel_feature == 'PRED':
                    pred_position = current_token_number
                    for back in range(1,4):
                        if len(previous_featuresets) >= back:
                            previous_feat = previous_featuresets[-1*back]
                            if 'HEAD-NP' in previous_feat:
                                if back > 1:
                                    feature_name = str(back)+'_or_less_before_pred'
                                    feature_name2 = str(back)+'_or_less_before_pred_'+featureset['word'].lower()
                                else:
                                    feature_name = str(back)+'_before_pred'
                                    feature_name2 = str(back)+'_before_pred_'+featureset['word'].lower()
                                previous_feat[feature_name]='True'                                
                                previous_feat[feature_name2]='True'
                ## features regarding word that precede PRED words  ** 57 **
                elif rel_feature == 'SUPPORT':
                    support_position = current_token_number
                    for back in range(1,4):
                        if len(previous_featuresets)>= back:
                            previous_feat = previous_featuresets[-1*back]
                            if 'HEAD-NP' in previous_feat:
                                if back == 1:
                                    feature_name = str(back)+'_before_support'
                                    feature_name2 = str(back)+'_before_support_'+featureset['word'].lower()
                                else:
                                    feature_name = str(back)+'_or_less_before_support'
                                    feature_name2 = str(back)+'_or_less_before_support_'+featureset['word'].lower()
                                previous_feat[feature_name]='True'
                                previous_feat[feature_name2]='True'
                     ## features regarding words that precede Suppport words
            if ('POS' in featureset) and (featureset['POS'] == 'CC') and (featureset['word'].lower() in ['or','and']):
                conjunction_position = current_token_number
                current_conjunction = featureset['word'].lower()
            if  ('POS' in featureset) and (featureset['POS'] == 'IN'):
                preposition_position = current_token_number
                current_preposition = featureset['word'].lower()
           ## only add features to head np words
            if 'HEAD-NP' in featureset:
                ## features regarding words that follow prepositions
                if (conjunction_position != 'NONE') and (current_token_number > conjunction_position):
                    how_much = current_token_number - conjunction_position
                    if how_much <=3:
                        featureset['3_or_less_after_conj'] = 'TRUE'
                    if how_much <=2:
                        featureset['2_or_less_after_conj'] = 'TRUE'
                    if how_much ==1:
                        featureset['1_after_conj'] = 'TRUE'
                if (preposition_position != 'NONE') and (current_token_number > preposition_position):
                    how_much = current_token_number - preposition_position
                    preposition_string = 'after_'+current_preposition
                    ## print(preposition_string,how_much)
                    if how_much <= 3:
                        featureset['3_or_less_'+preposition_string]='TRUE'
                    if how_much <=2:
                        featureset['2_or_less_'+preposition_string]='TRUE'
                    if how_much == 1:
                        featureset['1_'+preposition_string]='TRUE'
                    if (how_much >=3):
                        preposition_position = 'NONE'
                        current_preposition == 'NONE'
               ## features regarding words that follow PRED (predicate)
                if (pred_position != 'NONE') and (current_token_number > pred_position):
                    how_much = current_token_number - pred_position
                    pred_string = 'after'+'_pred'
                    if how_much <= 3:
                        featureset['3_or_less_'+pred_string]='TRUE'
                    if how_much <=2:
                        featureset['2_or_less_'+pred_string]='TRUE'
                    if how_much == 1:
                        featureset['1_'+pred_string]='TRUE'
                    if (how_much >= 3):
                        pred_position = 'NONE'                        
           ## features regarding words that would follow SUPPORT word
                if (support_position != 'NONE') and (current_token_number > support_position):
                    how_much = current_token_number - support_position
                    support_string = 'after'+'_support'
                    if how_much <= 3:
                        featureset['3_or_less_'+support_string]='TRUE'
                    if how_much <=2:
                        featureset['2_or_less_'+support_string]='TRUE'
                    if how_much == 1:
                        featureset['1_'+support_string]='TRUE'
                    if how_much >=3:
                        support_position = 'NONE'
                        support_string = 'NONE'
        next_position += 1

def process_absolute_interval_features(this_sentence,conj_positions,prep_positions,support_positions,pred_positions,trace=False):
    if trace:
        print('SENTENCE',this_sentence)
        print('CONJS',conj_positions,'PREPS',prep_positions)
        print('SUPPORT',support_positions,'PRED',pred_positions)
        input('break')
    before = 0
    after = 0
    ## change names to reflect opposite before/after 
    ## 
    for feature_set in this_sentence:
        current_token_number = int(feature_set['token_number'])
        if (len(conj_positions)>0)  and (not current_token_number in conj_positions):
            for position in conj_positions:
                if (position < current_token_number):
                    if (before == 0) or ((current_token_number - position) < before):
                        before = current_token_number - position
                if (position > current_token_number):
                    if (after == 0) or ((position - current_token_number) < after):
                        after = position - current_token_number
            if before !=0:
                feature_set['after_conj'] = before
                before = 0
            if after !=0:
                feature_set['before_conj'] = after
                after = 0
        if (len(prep_positions)>0) and (not current_token_number in prep_positions):
            for position in prep_positions:
                if (position < current_token_number):
                    if (before == 0) or ((current_token_number - position) < before):
                        before = current_token_number - position
                if (position > current_token_number):
                    if (after == 0) or ((position - current_token_number) < after):
                        after = position - current_token_number
            if before !=0:
                feature_set['after_prep'] = before
                before = 0
            if after !=0:
                feature_set['before_prep'] = after
                after = 0 
        if (len(support_positions)>0) and (not current_token_number in support_positions):
            for position in support_positions:
                if (position < current_token_number):
                    if (before == 0) or ((current_token_number - position) < before):
                        before = current_token_number - position
                if (position > current_token_number):
                    if (after == 0) or ((position - current_token_number) < after):
                        after = position - current_token_number
            if before !=0:
                feature_set['after_support'] = before
                before = 0
            if after !=0:
                feature_set['before_support'] = after
                after = 0 
        if (len(pred_positions)>0) and (not current_token_number in pred_positions):
            for position in pred_positions:
                if (position < current_token_number):
                    if (before == 0) or ((current_token_number - position) < before):
                        before = current_token_number - position
                if (position > current_token_number):
                    if (after == 0) or ((position - current_token_number) < after):
                        after = position - current_token_number
            if before !=0:
                feature_set['after_pred'] = before
                before = 0
            if after !=0:
                feature_set['before_pred'] = after
                after = 0 

def add_absolute_interval_features (output):
    ## go through output looking for all
    ## conj positions, prep_positions, pred_positions, support_positions
    ## (for current versions of this task, there can only be 1 pred and 1 support position

    ## for each position that does not fill each role,
    ## calculate distance from that position forward and backward for each word
    current_sentence = 0
    sentence_start = 0
    next_position = 0
    this_sentence = []
    conj_positions = []
    prep_positions = []
    pred_positions = []
    support_positions = []
    current_sentence = 0
    for featureset in output:
        if not 'sentence_number' in featureset:
            continue ## skip blank lines that are between sentences
        current_token_number = int(featureset['token_number'])
        current_sentence_number = int(featureset['sentence_number'])
        if current_sentence != current_sentence_number:
            process_absolute_interval_features(this_sentence,conj_positions,prep_positions,support_positions,pred_positions)
            ## token_number = 0
            conj_positions = []
            prep_positions = []
            pred_positions = []
            support_positions = []
            current_sentence = current_sentence_number
            this_sentence = [featureset]
        else:
            this_sentence.append(featureset)
            ## current_token_number = 
        # if current_token_number != next_position:
        #         print(current_token_number,"does not equal",next_position)
        #         input('pause')
        if ('POS' in featureset) and (featureset['POS'] == 'CC') and (featureset['word'].lower() in ['or','and']):
            conj_positions.append(current_token_number)
        elif ('POS' in featureset) and (featureset['POS'] in ['IN','TO']):
            prep_positions.append(current_token_number)
        if  'relation_feature' in featureset:
            rel_feature = featureset['relation_feature']
            if rel_feature == 'PRED':
                pred_positions.append(current_token_number)
            elif rel_feature == 'SUPPORT':
                support_positions.append(current_token_number)
    process_absolute_interval_features(this_sentence,conj_positions,prep_positions,support_positions,pred_positions)
    ## this processes the final sentence, after exiting the loop

def remove_consec_duplicates(path_list):
    if len(path_list) <= 1:
        return(path_list)
    output = [path_list[0]]
    for item in path_list:
        if (item != output[-1]):
            output.append(item)
    return(output)

def remove_CONJP(path_list):
    output = []
    for item in path_list:
        if item != 'CONJP':
            output.append(item)
    return(output)
    
def stringize_path(path_list):
    ## print('in',path_list)
    path_list = remove_CONJP(path_list)
    ## path_list = remove_consec_duplicates(path_list)
    if len(path_list) == 0:
        return('sister')
    else:
        output = path_list[0]
        for item in path_list[1:]:
            output += ('_'+item)
    ## print('output',output)
    return(output)

def find_non_O(path,start):
    if (start == 0):
        while (start<len(path)) and (path[start]=='O'):
            start +=1
    else:
        while ((start > 0) and (path[start] =='O')):
            start -=1
    return(start)

def find_non_O_end(path,end):
    while((end <len(path)) and (path[end-1] == 'O')):
          end +=1
    return(end)

def all_O(path):
    every_O = True
    for item in path:
        if item != 'O':
            every_O = False
    return(every_O)

def modify_POS(path):
    # if 'POS' in path:
    #     print(path)
    output = []
    path.reverse()
    POS = False
    index = 0
    for item in path:
        if (index == 0) and (item == 'POS'):
            output.append(item)
        elif (not POS) and (item == 'POS'):
            POS = True
        elif POS and (item == 'I-NP'):
            pass
        elif POS and (item == 'B-NP'):
            POS = False
            output.append(item)
        else:
            output.append(item)
        index += 1
    output.reverse()
    # if 'POS' in path:
    #     print(output)
    #     input('pause')
    return(output)

def sub_path_sequence(path,start,end):
    ## print(path,start,end)
    end = end+1
    if path[start] =='O':
        start=find_non_O(path,start)
    if (start == 0) and all_O(path[start:end]):
        end2 = find_non_O_end(path,end)
        if path[end2-1] != 'O':
            end = end2
    out_path =[]
    new_path = path[start:end]
    # if 'O' in new_path:
    #     print(new_path)
        ## input('pause')
    if 'POS' in new_path:
        new_path = modify_POS(new_path)
    if len(new_path)>0:
        if new_path[0].startswith('I-'):
            first_symbol = new_path[0]
            first_symbol = 'B-'+first_symbol[2:]
            new_path[0]=first_symbol
        for link in new_path:
            if link[0] == 'B':
                out_path.append(link[2:])
    ## print('out',out_path)
    return(out_path)


def sub_path_sequence_old(path,start,end):
    if path[start] =='O':
        start=find_non_O(path,start)
    if (start == 0) and all_O(path[start:end]):
        end2 = find_non_O_end(path,end)
        if path[end2-1] != 'O':
            end = end2
    out_path =[]
    new_path = path[start:end]
    if 'POS' in new_path:
        new_path = modify_POS(new_path)
    if new_path[0].startswith('I-'):
        first_symbol = new_path[0]
        first_symbol = 'B-'+first_symbol[2:]
        new_path[0]=first_symbol
    for link in new_path:
        if link[0] == 'B':
            out_path.append(link[2:])
    # if len(out_path) == 0:
    #     print(path,start,end)
    #     print(path)
    #     print(start,end)
    return(out_path)

def generalize_POS(POS):
    if (POS[0] == 'N') or (POS in ['PRP','WP','PP$']):
        out = 'NOUN'
    elif (POS[0] == 'V') or (POS in ['MD']):
        out = 'VERB'
    elif (POS == 'TO'):
        out = 'TO'
    else:
        out = 'OTHER'
    return(out)

def sentence_blocked(sentence_featureset,start,end):
    import re
    for word_dict in sentence_featureset[start:end]:
        if (word_dict['POS'][0] == 'W'):
            return(True)
        elif re.search('SBAR',word_dict['BIO']):
            return(True)
        elif word_dict['POS'] in ['VB','VBD','VBZ','VBP','MD','COMMA']:
            return(True)
    return(False)

def type_sensitive_not (item):
    if (type(item) == bool) and (not item):
        return(True)
    else:
        return(False)
    
def type_sensitive_true (item):
    if (type(item) == bool) and (not item):
        return(False)
    else:
        return(True)
    
def remove_pred_path(feature_set):
    if 'pred_path' in feature_set:
        feature_set.pop('pred_path')

def add_sentence_features(sentence_featureset,prediction,only_one_path=True):
    ## maybe do not need prediction:
    ## modify remove pred path feature, if support is present
    ##      if only_one_path == True
    bio_sequence = []
    after_pred = False
    support_position = False
    pred_position = False
    index = 0
    pos_sequence = []
    sent_length = len(sentence_featureset)
    found_0_after_pred = False
    for indict in sentence_featureset:
        ### identify HEAD-NP
        current_BIO = indict['BIO']
        if (current_BIO == 'O') and (index > pred_position):
            found_0_after_pred = True
        if index == 0:
            previous_indict = False
            previous_BIO = False
        else:
            previous_indict = sentence_featureset[index-1]
            previous_BIO = previous_indict['BIO']
        index +=1
        if  (current_BIO == 'O') and previous_indict and ('HEAD-NP' in previous_indict):
            pass
        elif (previous_BIO in ['B-NP','I-NP']) and (not current_BIO in ['I-NP']):
            previous_indict['HEAD-NP'] = 'HEAD-NP'
        elif (index == sent_length) and (current_BIO in ['B-NP','I-NP']):
            indict['HEAD-NP'] = 'HEAD-NP'
    index = 0
    for indict in sentence_featureset:
        ## POS filter
        if indict['BIO'] == 'B-PP':
            word = indict['word'].upper()
            bio_sequence.append(indict['BIO']+'_'+word)
        else:
            bio_sequence.append(indict['BIO'])
        pos_sequence.append(generalize_POS(indict['POS']))
        ## remove all but the last POS later
        if 'relation_feature' in indict:
            value = indict['relation_feature']
            if value == 'PRED':
                pred_position = index
            elif value == 'SUPPORT':
                support_position = index
        index +=1
    index = 0
    for indict in sentence_featureset[:]:
        if indict['word'] == 'SENTENCE_BREAK':
            index+=1
        else:
            # print(index)
            # print(indict['word'])
            # print(indict['BIO'])
            support_path = False
            pred_path = False
            ## use this to constrain path arguments to head nouns
            ## adjacent noun arguments are a different case, e.g.,
            ## "IBM" in "IBM appointment'
            if 'HEAD-NP' in indict:
                head_NP = True
            else:
                head_NP = False            
            if (index == pred_position):
                pass
            elif index > pred_position:
                if not(sentence_blocked(sentence_featureset,pred_position,index)):
                    # print(0,bio_sequence[pred_position:index])
                    # print(indict['BIO'])
                    path1 = sub_path_sequence(bio_sequence,pred_position,index)
                    ## print(1,path1)
                    if (len(path1)>1) and (path1[-1] == 'NP') and (path1[-2]=='NP'):
                        path1 = []
                        pred_path = False
                    elif (len(path1)==0) or ((path1[-1] == 'NP') and (not head_NP)):
                        pass
                    else:
                        path1.append(pos_sequence[index])
                        pred_path = 'right_'+stringize_path(path1)
                    ## print(pred_path)
                if  (index > 0) and (sentence_featureset[index] ['POS']== 'POS'):
                    if ('pred_path' in sentence_featureset[index-1]):
                        sentence_featureset[index-1].pop('pred_path')
                        ## remove pred_path if immediately following possessive marker
            else:
                if not(sentence_blocked(sentence_featureset,index,pred_position)):
                    path1 = [pos_sequence[index]]
                    path2 =(sub_path_sequence(bio_sequence,index,pred_position))
                    if (len(path2) == 0) or ((path2[0] == 'NP') and (not head_NP)):
                        pass
                    else:
                        path1.extend(path2)
                        pred_path = 'left_'+stringize_path(path1)
                    # print(pred_path)
                    # input('pause 2')
            if type_sensitive_not (support_position):
                pass
            elif index == support_position:
                pass
            elif index == pred_position:
                pass
            elif index > support_position:
                if not(sentence_blocked(sentence_featureset,support_position,index)):
                    path1 = sub_path_sequence(bio_sequence,support_position,index)
                    pos_sequence[index]
                    if len(path1) <= 1:
                        ## pass
                        distance = index-support_position
                        support_path = 'right_'+str(distance)
                    elif (path1[-1] == 'NP') and (not head_NP):
                       pass
                    else:
                        path1.append(pos_sequence[index])
                        support_path = 'right_'+stringize_path(path1)
            else:
                if not(sentence_blocked(sentence_featureset,index,support_position)):
                    path1 = [pos_sequence[index]]
                    path2 = (sub_path_sequence(bio_sequence,index,support_position))
                    if (len(path2) == 0):
                        pass
                    elif (len(path2) == 1):
                        distance = support_position-index
                        support_path = 'left_'+str(distance)
                    elif (path2[0] == 'NP') and (not head_NP):
                        pass
                    else:
                        path1.extend(path2)
                        support_path = 'left_'+stringize_path(path1)
            if pred_path:
                indict['pred_path']=pred_path
                ## print('pred',pred_path)
            if support_path:
                indict['support_path'] = support_path
                ## print('support',support_path)
            ## input('pause')
            index +=1
    ## for same left paths, remove all but the last one (the one closest to pred/support)
    ## for same right paths, remove all but the first one (the one closest to pred/support)
    ## we do left paths first (deleting previous when going forward)
    ## theb we reverse the list and do the right paths
    previous_dict = False
    previous_pred = []
    previous_support = []
    for current_dict in sentence_featureset:
        if ('support_path' in current_dict):
            current_support_path = current_dict['support_path']
            if current_support_path.startswith('left') and  current_support_path in previous_support:
                current_dict.pop('support_path')
            else:
                previous_support.append(current_support_path)
        if ('pred_path' in current_dict):
            current_pred_path = current_dict['pred_path']
            if current_pred_path.startswith('left') and  current_pred_path in previous_pred:
                current_dict.pop('pred_path')
            else:
                previous_pred.append(current_pred_path)
    previous_pred = []
    previous_support = []
    sentence_featureset.reverse()
    for current_dict in sentence_featureset:
        if ('support_path' in current_dict):
            current_support_path = current_dict['support_path']
            if current_support_path.startswith('right') and  current_support_path in previous_support:
                current_dict.pop('support_path')
            else:
                previous_support.append(current_support_path)
        # if ('pred_path' in current_dict):
        #     current_pred_path = current_dict['pred_path']
        #     if current_pred_path.startswith('right') and  (current_pred_path in previous_pred):
        #         current_dict.pop('pred_path')
        #     else:
        #         previous_pred.append(current_pred_path)
    if type_sensitive_true(support_position):
        for feature_set in sentence_featureset:
            remove_pred_path(feature_set)
    sentence_featureset.reverse()  ## put them back in the right order

def add_sentence_features_old2(sentence_featset,prediction):
    ## derive sequence of chunk tags connecting
    ## pred and prediction
    ## collapse each sequence of the form
    ##  B-XP I-XP*
    ## to XP
    ## derive a sequence of chunk tags connecting
    ## support and prediction
    ## use
    ##
    ## if Pred occurs in a support path, assume "no support"
    ## if support occurs in a pred path, ssume "no pred"
    ## i.e., probably, only include closest path
    chunk_path = []
    sentence_number = 'not_checked'
    index = 0
    pred_exists = False
    prediction_exists = False
    support_exists = False
    arg_dict = 'False'
    prediction_index = 'False'
    for node_dict in sentence_featset:
        if (sentence_number == 'not_checked') and ('sentence_number' in node_dict):
            sentence_number = node_dict['sentence_number']
        chunk_path.append(node_dict['BIO'])
        if 'relation_feature' in node_dict:
            rfeature = node_dict['relation_feature']
            if rfeature=='PRED':
                pred_index = index
                pred_exists = True
            elif rfeature == 'SUPPORT':
                support_index = index
                support_exists = True
            elif rfeature == prediction:
                prediction_index = index
                prediction_exists = True
                arg_dict = sentence_featset[index]
        index +=1
##    if True: ## prediction_exists:
        if pred_exists and (prediction_index != 'False'):
            if pred_index < prediction_index:
                pred_path = 'right_'+stringize_path(sub_path_sequence(chunk_path,pred_index,prediction_index))
            else:
                pred_path = 'left_'+stringize_path(sub_path_sequence(chunk_path,prediction_index,pred_index))
            # print(pred_path)
            # print(pred_index,prediction_index)
            # print(chunk_path)
            # input('pause')
        else:
            pred_path = 'False'
        if support_exists:
            if support_index < prediction_index:
                support_path = 'left_'+stringize_path(sub_path_sequence(chunk_path,support_index,prediction_index))
            else:
                support_path = 'right_'+stringize_path(sub_path_sequence(chunk_path,prediction_index,support_index))
        else:
            support_path = 'False'
        print('arg')
        print(arg_dict)
        print(support_path)
        print(pred_path)
        ## input('pause')
        print('***')
        if arg_dict and (arg_dict != 'False') and support_path:
            print('*',arg_dict)
            arg_dict['support_path']=support_path
        if arg_dict and (arg_dict != 'False') and pred_path:
            print('*',arg_dict)
            arg_dict['pred_path']=pred_path
        # print('S',support_path)
        # print('P',pred_path)
        # input('pause')

def add_sentence_features_old(sentence_featset,prediction):
    chunk_path = []
    sentence_number = 'not_checked'
    for node_dict in sentence_featset:
        if (sentence_number == 'not_checked') and ('sentence_number' in node_dict):
            sentence_number = node_dict['sentence_number']
        if ('BIO' in node_dict) and (node_dict['BIO'][0] == 'B'):
            chunk_path.append(node_dict['BIO'][2:])
        if 'relation_feature' in node_dict:
            chunk_path.append(node_dict['relation_feature'])
            if node_dict['relation_feature']== prediction:
                arg_dict = node_dict
    if 'PRED' in chunk_path:
        pred = chunk_path.index('PRED')
    else:
        print('sentence_number:',sentence_number)
        print('Warning no PRED in sentence')
        return(False)
    if prediction in chunk_path:
        arg = chunk_path.index(prediction)
    else:
        print('warning: no',prediction,'in sentence',sentence_number)
        return(False)
    lower = min(pred,arg)+1
    higher = max(pred,arg)
    if higher>lower:
        pred_arg_path = stringize_path(chunk_path[lower:higher])
    elif lower == pred:
        pred_arg_path='pred_left'
    else:
        pred_arg_path='pred_right'
        # print('pred',pred)
        # print('arg',arg)
        # print(pred_arg_path)
        # input('pause')
    if 'SUPPORT' in chunk_path:
        support = chunk_path.index('SUPPORT')
        lower = min(support,arg)+1
        higher = max(support,arg)
        if higher>lower:
            support_path = stringize_path(chunk_path[lower:higher])
        elif lower == support:
            support_path ='SUPP_left'
        elif lower == arg:
            support_path ='SUPP_right'
        else:
            support_path = 'NO SUPPORT'
    else:
        support_path = 'NO SUPPORT'
        arg_dict['pred_path']=pred_arg_path
        arg_dict['support_path']=support_path

def get_all_n_grams(sentence_list):
    ## function not currently being used
    unigrams = sentence_list
    bigrams = []
    trigrams = []
    for n in range(1,len(sentence_list)):
        bigrams.append(' '.join(sentence_list[n-1:n+1]))
    for n in range(2, len(sentence_list)):
        trigrams.append(' '.join(sentence_list[n-2:n+1]))
    for index in range(len(unigrams)):
        unigrams[index]=spacy_nlp(unigrams[index])
    for index in range(len(bigrams)):
        bigrams[index]=spacy_nlp(bigrams[index])
    for index in range(len(trigrams)):
        trigrams[index]=spacy_nlp(trigrams[index])
    return(unigrams,bigrams,trigrams)


def get_word_list_from_sentence_features(sentence_feature_set):
    output = []
    for word_features in sentence_feature_set:
        output.append(word_features['word'])
    return(output)

def add_embedding_features(sentence_feature_set,embedding_vector_list,prediction):
    average_back_trigram,average_back_bigram,average_unigram,average_forward_bigram,average_forward_trigram,\
    average_slash_back_trigram,average_slash_back_bigram,average_slash_unigram,average_slash_forward_bigram,\
    average_slash_forward_trigram = \
      embedding_vector_list
    best_unigrams = []
    best_backward_bigram = []
    best_forward_bigram = []
    best_backward_trigram =[]
    best_forward_trigram= []
    best_slash_unigrams = []
    best_slash_backward_bigram = []
    best_slash_forward_bigram = []
    best_slash_backward_trigram =[]
    best_slash_forward_trigram= []
    index = 0
    ### *** add slash cases also ***
    words = get_word_list_from_sentence_features(sentence_feature_set)
    for index in range(len(sentence_feature_set)):
        word_features = sentence_feature_set[index]
        word = word_features['word']
        if word == 'SENTENCE_BREAK':
            continue  ## ignore sentence breaks for purposes of unigrams
        unigram = spacy_nlp(word)
        unigram_similarity =  cosine_similarity_vector_and_array(unigram,average_unigram)
        word_features['unigram_embed_similarity'] = stringify_similarity(unigram_similarity)
        best_unigrams.append([unigram_similarity,index])
        slash_unigram = spacy_nlp(index_complement_string(words,index,index+1))
        slash_unigram_similarity =  cosine_similarity_vector_and_array(slash_unigram,average_slash_unigram)
        word_features['slash_unigram_embed_similarity'] = stringify_similarity(slash_unigram_similarity)
        if index > 0:
            previous_word =sentence_feature_set[index-1]['word']
            if previous_word == 'SENTENCE_BREAK':
                continue ## ignore bigrams for first words in the sentence
            bigram_words = ' '.join([previous_word,word])
            bigram = spacy_nlp(bigram_words)
            slash_bigram = spacy_nlp(index_complement_string(words,index-1,index+1)) 
            similarity1 = cosine_similarity_vector_and_array(bigram,average_forward_bigram)
            similarity2 = cosine_similarity_vector_and_array(bigram,average_back_bigram)
            similarity3 = cosine_similarity_vector_and_array(slash_bigram,average_slash_forward_bigram)
            similarity4 = cosine_similarity_vector_and_array(bigram,average_slash_back_bigram)
            ## generalize to PREDICTION=ARG?
            word_features['forward_bigram_embed_similarity'] = stringify_similarity(similarity1 )
            word_features['back_bigram_embed_similarity'] = stringify_similarity(similarity2 )
            word_features['forward_bigram_embed_slash_similarity'] = stringify_similarity(similarity3)
            word_features['back_bigram_embed_slash_similarity'] = stringify_similarity(similarity4 )
            best_forward_bigram.append([similarity1,index-1])
            best_backward_bigram.append([similarity2,index])
            best_slash_forward_bigram.append([similarity3,index-1])
            best_slash_backward_bigram.append([similarity4,index])
        if index > 1:
            previous_word2 =sentence_feature_set[index-2]['word']
            if previous_word2 == 'SENTENCE_BREAK':
                continue ## ignore trigrams beginning with sentence breaks
            previous_word =sentence_feature_set[index-1]['word']
            trigram_words = ' '.join([previous_word2,previous_word,word])
            trigram = spacy_nlp(trigram_words)
            slash_trigram = spacy_nlp(index_complement_string(words,index-2,index+1)) ## 57
            similarity1 = cosine_similarity_vector_and_array(bigram,average_back_trigram)
            similarity2 = cosine_similarity_vector_and_array(bigram,average_forward_trigram)
            similarity3 = cosine_similarity_vector_and_array(bigram,average_slash_back_trigram)
            similarity4 = cosine_similarity_vector_and_array(bigram,average_slash_forward_trigram)
            ## generalize to PREDICTION=ARG?
            word_features['forward_trigram_embed_similarity'] = stringify_similarity(similarity1 )
            word_features['back_trigram_embed_similarity'] = stringify_similarity(similarity2)
            word_features['forward_trigram_embed_slash_similarity'] = stringify_similarity(similarity1 )
            word_features['back_trigram_embed_slash_similarity'] = stringify_similarity(similarity2)
            best_forward_trigram.append([similarity1,index-2])
            best_backward_trigram.append([similarity2,index])
            best_slash_forward_trigram.append([similarity3,index-2])
            best_slash_backward_trigram.append([similarity4,index])
    best_unigrams.sort(reverse=True, key=lambda pair: pair[0])
    best_backward_bigram.sort(reverse=True, key=lambda pair: pair[0])
    best_forward_bigram.sort(reverse=True, key=lambda pair: pair[0])
    best_backward_trigram.sort(reverse=True, key=lambda pair: pair[0])
    best_forward_trigram.sort(reverse=True, key=lambda pair: pair[0])
    best_slash_unigrams.sort(reverse=True, key=lambda pair: pair[0])
    best_slash_backward_bigram.sort(reverse=True, key=lambda pair: pair[0])
    best_slash_forward_bigram.sort(reverse=True, key=lambda pair: pair[0])
    best_slash_backward_trigram.sort(reverse=True, key=lambda pair: pair[0])
    best_slash_forward_trigram.sort(reverse=True, key=lambda pair: pair[0])
    ## Above sorts these by similarity
    ## the second element is an index into sentence_feature_set -- identifying where to add the features
    ## only look at first 5 in each sorted list
    for rank in range(5):
        try:
            if rank > 0:
                prefix = 'top_'+str(rank+1)+'_'
            else:
                prefix = 'best_'
            if  len(best_unigrams) > rank:
                next_unigram = best_unigrams[rank]
                feature_set = sentence_feature_set[next_unigram[1]]
                feature_set[prefix+'embed_unigram'] = 'True'
            if len(best_backward_bigram) > rank:
                next_back_bigram = best_backward_bigram[rank]
                feature_set = sentence_feature_set[next_back_bigram[1]]
                feature_set[prefix+'embed_back_bigram']='True'
            if  len(best_backward_trigram) > rank:
                next_backward_trigram = best_backward_bigram[rank]
                feature_set = sentence_feature_set[next_backward_trigram[1]]
                feature_set[prefix+'embed_back_trigram']='True'
            if  len(best_forward_bigram) > rank:
                next_forward_bigram = best_forward_bigram[rank]
                feature_set = sentence_feature_set[next_forward_bigram[1]]
                feature_set[prefix+'embed_forward_bigram']='True'
            if  len(best_forward_trigram) > rank:
                next_forward_trigram = best_forward_trigram[rank]
                feature_set = sentence_feature_set[next_forward_trigram[1]]
                feature_set[prefix+'embed_forward_trigram']='True'
### *** 57
            if  len(best_slash_unigrams) > rank:
                next_unigram = best_slash_unigrams[rank]
                feature_set = sentence_feature_set[next_unigram[1]]
                feature_set[prefix+'embed_slash_unigram'] = 'True'
            if len(best_slash_backward_bigram) > rank:
                next_back_bigram = best_slash_backward_bigram[rank]
                feature_set = sentence_feature_set[next_back_bigram[1]]
                feature_set[prefix+'embed_slash_back_bigram']='True'
            if  len(best_slash_backward_trigram) > rank:
                next_backward_trigram = best_slash_backward_bigram[rank]
                feature_set = sentence_feature_set[next_backward_trigram[1]]
                feature_set[prefix+'embed_slash_back_trigram']='True'
            if  len(best_slash_forward_bigram) > rank:
                next_forward_bigram = best_slash_forward_bigram[rank]
                feature_set = sentence_feature_set[next_forward_bigram[1]]
                feature_set[prefix+'embed_slash_forward_bigram']='True'
            if  len(best_slash_forward_trigram) > rank:
                next_forward_trigram = best_slash_forward_trigram[rank]
                feature_set = sentence_feature_set[next_forward_trigram[1]]
                feature_set[prefix+'embed_slash_forward_trigram']='True'
        except:
            print('rank',rank)
            print('length',len(best_forward_trigram))
            raise(Exception)

def get_sentences_from_file(infile):
    sentences = []
    sentence = []
    for line in open(infile):
        line = line.strip(os.linesep)
        if '\t' in line:
            columns = line.split('\t')
            sentence.append(columns)
        else:
            sentences.append(sentence)
            sentence = []
    return(sentences)

def get_first_items(inlist):
    output = []
    for item in inlist:
        output.append(item[0])
    return(output)

def get_string_from_tuples(inlist):
    items = get_first_items(inlist)
    return(" ".join(items))

def get_embedding_vectors_from_sentence(sentence,prediction):
    index = 0
    no_prediction = True
    for tuple in sentence:
        if (len(tuple)>5) and (tuple[5] == prediction):
            no_prediction = False
            break
        else:
            index +=1
    ## index points to the prediction (the middle of the N-grams)
    ## always do unigram -- that is how the data is set up
    if no_prediction:
        return(False,False,False,False,False)
    unigram = spacy_nlp (sentence[index][0])
    back_bigram = False
    back_trigram = False
    forward_bigram = False
    forward_trigram = False
    if index >=1:
        back_bigram = spacy_nlp(get_string_from_tuples(sentence[(index-1):(index+1)]))
        ## also do back bigram
    if index >=2:
        ## also do back trigram
       back_trigram = spacy_nlp(get_string_from_tuples(sentence[(index-2):(index+1)]))
    if (len(sentence)-index) >=1:
        ## do forward bigram
        forward_bigram = spacy_nlp(get_string_from_tuples(sentence[index:index+2]))
    if (len(sentence)-index) >= 2:
        ## do forward trigram
        forward_trigram = spacy_nlp(get_string_from_tuples(sentence[index:index+3]))
    return(back_trigram,back_bigram,unigram,forward_bigram,forward_trigram)    

def index_complement_string(word_list,start_index,end_index):
    sequence = word_list[:start_index] + word_list[end_index:]
    for index in range(len(sequence)):
        sequence[index]=sequence[index][0] ## use words, not other features
    return(' '.join(sequence))

def  get_slash_embedding_vectors_from_sentence(sentence,prediction):
    index = 0
    for tuple in sentence:
        if (len(tuple)>5) and (tuple[5] == prediction):
            break
        else:
            index +=1
    ## index points to the prediction (the middle of the N-grams)
    ## always do unigram -- that is how the data is set up
    slash_unigram = spacy_nlp (index_complement_string(sentence,index,index+1))
    slash_back_bigram = False
    slash_back_trigram = False
    slash_forward_bigram = False
    slash_forward_trigram = False
    if index >=1:
        slash_back_bigram = spacy_nlp(index_complement_string(sentence,index-1,index+1))
        ## also do back bigram
    if index >=2:
        ## also do back trigram
       slash_back_trigram = spacy_nlp(index_complement_string(sentence,index-2,index+1))
    if (len(sentence)-index) >=1:
        ## do forward bigram
        slash_forward_bigram = spacy_nlp(index_complement_string(sentence,index,index+2))
    if (len(sentence)-index) >= 2:
        ## do forward trigram
        slash_forward_trigram = spacy_nlp(index_complement_string(sentence,index,index+3))
    return(slash_back_trigram,slash_back_bigram,slash_unigram,slash_forward_bigram,slash_forward_trigram)    

    
def vector_average_old(vector_list):
    length = len(vector_list)
    new_array = np.array(len(vector_list[0].vector))
    for vector in vector_list:
        new_array = new_array+vector.vector
    new_array = new_array/length
    print(new_array)
    return(new_array)

def vector_average(vector_list):
    ## adjusted from test file (vector --> vector.vector)
    length = len(vector_list[0].vector) 
    output_list = [0]*len(vector_list[0].vector)
    ## print(output_list)
    for vector in vector_list:
        for index in range(len(vector.vector)):
            output_list[index] +=(vector.vector[index]/length)
    new_array = np.array(output_list)
    return(new_array)
    
def make_prediction_embeddings(infile,prediction):
    ##
    global spacy_nlp
    backward_trigram_vectors = []
    backward_bigram_vectors = []
    unigram_vectors = []
    forward_bigram_vectors = []
    forward_trigram_vectors = []
    backward_trigram_slash_vectors = []
    backward_bigram_slash_vectors = []
    unigram_slash_vectors = []
    forward_bigram_slash_vectors = []
    forward_trigram_slash_vectors = []
    for sentence in get_sentences_from_file(infile):
        back_tri,back_bi,unigram,forward_bi,forward_tri = get_embedding_vectors_from_sentence(sentence,prediction)
        back_slash_tri,back_slash_bi,unigram_slash,forward_slash_bi,forward_slash_tri =\
            get_slash_embedding_vectors_from_sentence(sentence,prediction)
        ## 57 
        if back_tri:
            backward_trigram_vectors.append(back_tri)
        if back_bi:
            backward_bigram_vectors.append(back_bi)
        if unigram:
            unigram_vectors.append(unigram)
        if forward_bi:
            forward_bigram_vectors.append(forward_bi)
        if forward_tri:
            forward_trigram_vectors.append(forward_tri)
        if back_slash_tri:
            backward_trigram_slash_vectors.append(back_slash_tri)
        if back_slash_bi:
            backward_bigram_slash_vectors.append(back_slash_bi)
        if unigram_slash:
            unigram_slash_vectors.append(unigram_slash)
        if forward_slash_bi:
            forward_bigram_slash_vectors.append(forward_slash_bi)
        if forward_slash_tri:
            forward_trigram_slash_vectors.append(forward_slash_tri)
    average_back_trigram = vector_average(backward_trigram_vectors)
    average_back_bigram =  vector_average(backward_bigram_vectors)
    average_unigram = vector_average(unigram_vectors)
    average_forward_bigram =  vector_average(forward_bigram_vectors)
    average_forward_trigram =  vector_average(forward_trigram_vectors)
    ## slash vectors
    average_slash_back_trigram = vector_average(backward_trigram_slash_vectors)
    average_slash_back_bigram =  vector_average(backward_bigram_slash_vectors)
    average_slash_unigram = vector_average(unigram_slash_vectors)
    average_slash_forward_bigram =  vector_average(forward_bigram_slash_vectors)
    average_slash_forward_trigram =  vector_average(forward_trigram_slash_vectors)
    return([average_back_trigram,average_back_bigram,average_unigram,average_forward_bigram,average_forward_trigram, \
            average_slash_back_trigram,average_slash_back_bigram,average_slash_unigram,average_slash_forward_bigram,\
            average_slash_forward_trigram])

def  save_embedding_info(embedding_vector_list,embedding_file):
    #
    average_back_trigram,average_back_bigram,average_unigram,average_forward_bigram,average_forward_trigram, \
            average_slash_back_trigram,average_slash_back_bigram,average_slash_unigram,average_slash_forward_bigram,\
            average_slash_forward_trigram  = embedding_vector_list
    with open(embedding_file,'w') as outstream:
        for label,vector in  [['average_unigram',average_unigram],['average_back_bigram',average_back_bigram],\
                                      ['average_forward_bigram',average_forward_bigram],['average_back_trigram',average_back_trigram],\
                                      ['average_forward_trigram',average_forward_trigram],\
                              ['average_slash_back_trigram',average_slash_back_trigram],\
                              ['average_slash_back_bigram',average_slash_back_bigram],\
                              ['average_slash_unigram',average_slash_unigram],\
                              ['average_slash_forward_bigram',average_slash_forward_bigram],\
                              ['average_slash_forward_trigram',average_slash_forward_trigram]]:
            outstream.write(label)
            for number in vector:
                outstream.write('\t'+str(number))
            outstream.write(os.linesep)

def load_embedding_file(embedding_file):
    output = {}
    with open(embedding_file) as instream:
        for line in instream:
            line = line.strip(os.linesep)
            line_list = line.split('\t')
            line_label = line_list[0]
            for index in range(1,len(line_list)):
                line_list[index] = float(line_list[index])
            new_array = np.array(line_list[1:])
            output[line_label] = new_array
    return([output['average_back_trigram'],output['average_back_bigram'],\
                output['average_unigram'],output['average_forward_bigram'],\
                output['average_forward_trigram'],output['average_slash_back_trigram'],output['average_slash_back_bigram'],\
                output['average_slash_unigram'],output['average_slash_forward_bigram'],\
                output['average_slash_forward_trigram']
    ])

def noun_group_B_NP_fix(line_features,previous_features):
    if (len(previous_features) > 0) and ('BIO' in previous_features[-1])\
         and (previous_features[-1]['BIO']=='B-NP'):
        if len(line_features)<3:
            print('bad line_featuresin noun_group_B_NP_fix',line_features) ## 57
        elif (line_features[2] == 'B-NP'):
            line_features[2] = 'I-NP'

def modify_BIO_closed_class(output):
    last_BIO = False
    for wdict in output:
        if (not 'POS' in wdict):
            continue
        elif (wdict['POS'] == 'POS'):
            wdict['BIO'] = 'POS'
            last_BIO = 'POS'
        elif (last_BIO == 'POS') and (wdict['BIO']=='I-NP'):
            wdict['BIO']=='B-NP'
            last_BIO = False
        if (wdict['POS'] == 'WDT') and (wdict['word'].lower() == 'that'):
            wdict['BIO'] = 'WDT'
            
def make_feature_file(infile,outfile,is_test_data,prediction,embedding_file,use_embedding_file=False,noun_group_edit=False,partitive=False):
    ## noun_group_edit assumes that instances of 2 B-NP in a row are errors and the second B-NP should actually be an I-NP
    ## 57 *** use partitive key word
    print("Making embedding info")
    if not(is_test_data) and ((not(use_embedding_file))  or (not  os.path.isfile(embedding_file))):
        embedding_vector_list = make_prediction_embeddings(infile,prediction)
        save_embedding_info(embedding_vector_list,embedding_file)  ##
    else:
        embedding_vector_list =load_embedding_file(embedding_file)

    print("Making feature set")

    with open(infile) as instream:
        buf = instream.readlines()
    output = []
    ## if PRD is first, make AFTER_PATH, else make FORWARD_PATH
    ## accumulate instances of B-XP
    ## ends when it includes both a PRD and a prediction (e.g., ARG1)
    ## add feature to every NP head (final noun in NG)
    ## PATH of all CHUNK HEADS
    ## with SUPPORT
    start = 0
    current = 0
    predicate_features = []
    current_sentence = []
    print(len(buf))
    for line in buf:
        if current % 100 == 0:
            print(current)
        current += 1
        line = line.strip(os.linesep)
        line_features = line.split('\t')
        if noun_group_edit and (len(output)>0):
            noun_group_B_NP_fix(line_features,output)
        if (len(line_features) >= 3):
            ## current_features,predicate_features_out =get_features_using_history(output,line_features,is_test_data,prediction)
            current_features,predicate_features_out =get_features_using_history(current_sentence,line_features,is_test_data,prediction)
            output.append(current_features)
            current_sentence.append(current_features)
            update_based_on_forward_context(current_sentence[:-1],line_features,is_test_data)
            ## remove current line (which is at the end of line_features
            if len(predicate_features_out)> 0:
                for pred_feature in predicate_features_out:
                    if not pred_feature in predicate_features:
                        predicate_features.append(pred_feature)
        else:
            output.append(sentence_break_features.copy())
            update_forward_context_sentence_break_features(current_sentence)
            for little_feature_set in current_sentence:
                for predicate_feature in predicate_features:
                    if predicate_feature in little_feature_set:
                        pass
                    else:
                        little_feature_set[predicate_feature]='True'
            predicate_features = []
            current_sentence = []
            # if (len (output) > 0) and (output[-1]['BIO'] in ['B-NP','I-NP']):
            #     output[-1]['HEAD-NP'] = 'HEAD-NP'
            #     print(output[-1])
            #     input('pause2')
            modify_BIO_closed_class(output[start:current])
            add_sentence_features(output[start:current],prediction)
            add_embedding_features(output[start:current], embedding_vector_list,prediction)
            start = current
    add_interval_features(output)
    add_absolute_interval_features (output)
    with open(outfile,'w') as outstream:
        for features in output:
            outstream.write(get_feature_string(features,is_test_data,prediction))

def answer_feature_exchange(line,old_feature,new_feature):
    import re
    import os
    old_feature_match = re.compile('\t'+old_feature+'[\n\r]+')
    new_feature_match = re.compile('\trelation_feature='+new_feature)
    old_match = old_feature_match.search(line)
    if old_match:
        ## place the relational feature 5th
        if line.count('\t') > 5:
            line_list = line.split('\t')[:-1]
            new_line_list = line_list[:5]+['relation_feature='+old_feature]+line_list[5:]
            line = '\t'.join(new_line_list)+'\n'
        else:
            line = line[:old_match.start()]+'\trelation_feature='+old_feature+'\n'
    new_match = new_feature_match.search(line)
    if new_match:
        line = line[:new_match.start()]+line[new_match.end():]
        line = line.strip(os.linesep)
        line = line+'\t'+new_feature+'\n'
    return(line)

def convert_feature_file_for_other_arg(infile,old_feature,new_feature,outfile):
    with open(infile) as instream,open(outfile,'w') as outstream:
        for line in instream:
            line = answer_feature_exchange(line,old_feature,new_feature)
            outstream.write(line)

def make_percent_training_feature_file(noun_group_edit=False):
    input_file = os.path.join("percentage", "train.data")
    output_file = os.path.join("percentage", "train.features")
    embedding_file = os.path.join("percentage", "train.embedding")
    make_feature_file(input_file,output_file,False,'ARG1',embedding_file,use_embedding_file=False,noun_group_edit=True)

def make_percent_dev_feature_file(noun_group_edit=False):
    input_file = os.path.join("percentage", "dev.data")
    output_file = os.path.join("percentage", "dev.features")
    embedding_file = os.path.join("percentage", "train.embedding")
    make_feature_file(input_file,output_file,False,'ARG1',embedding_file,use_embedding_file=True, noun_group_edit=True)

def make_percent_test_feature_file(noun_group_edit=False):
    input_file = os.path.join("percentage", "test.data")
    output_file = os.path.join("percentage", "test.features")
    embedding_file = os.path.join("percentage", "train.embedding")
    make_feature_file(input_file,output_file,False,'ARG1',embedding_file,use_embedding_file=True, noun_group_edit=True)

def make_part_training_feature_file(noun_group_edit=False):
    input_file = os.path.join("part", "train.data")
    output_file = os.path.join("part", "train.features")
    embedding_file = os.path.join("part", "train.embedding")
    make_feature_file(input_file,output_file,False,'ARG1',embedding_file,use_embedding_file=False,noun_group_edit=True, partitive=True)

def make_part_dev_feature_file(noun_group_edit=False):
    input_file = os.path.join("part", "dev.data")
    output_file = os.path.join("part", "dev.features")
    embedding_file = os.path.join("part", "train.embedding")
    make_feature_file(input_file,output_file,False,'ARG1',embedding_file,use_embedding_file=True, noun_group_edit=True, partitive=True)

def make_part_test_feature_file(noun_group_edit=False):
    input_file = os.path.join("part", "test.data")
    output_file = os.path.join("part", "test.features")
    embedding_file = os.path.join("part", "train.embedding")
    make_feature_file(input_file,output_file,False,'ARG1',embedding_file,use_embedding_file=True, noun_group_edit=True, partitive=True)

def make_all_training_feature_file(noun_group_edit=False):
    input_file = os.path.join("all_nombank", "train.data")
    output_file = os.path.join("all_nombank", "train.features")
    embedding_file = os.path.join("all_nombank", "train.embedding")
    make_feature_file(input_file,output_file,False,'ARG1',embedding_file,use_embedding_file=False,noun_group_edit=True, partitive=True)

def make_all_dev_feature_file(noun_group_edit=False):
    input_file = os.path.join("all_nombank", "train.data")
    output_file = os.path.join("all_nombank", "train.features")
    embedding_file = os.path.join("all_nombank", "train.embedding")
    make_feature_file(input_file,output_file,False,'ARG1',embedding_file,use_embedding_file=True,noun_group_edit=True, partitive=True)

def conversion_for_all_arg1_to_arg0():
    convert_feature_file_for_other_arg('all-training.features','ARG1','ARG0','all-training0.features')
    convert_feature_file_for_other_arg('all-dev.features','ARG1','ARG0','all-dev0.features')
