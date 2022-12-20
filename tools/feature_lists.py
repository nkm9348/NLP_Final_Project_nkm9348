baseline1_features = ['token_number','sentence_number','word','pos','bio','relation_feature','pos_back_1','pos_back_2','pos_plus_1','pos_plus_2',
                      'word_back_1','word_back_2','word_plus_1','word_plus_2',
                      'bio_back_1','bio_back_2','bio_plus_1','bio_plus_2','rel_back_1','rel_back_2','rel_plus_1','rel_plus_2','head-np']

baseline1_word_features= ['word','word_back_1','word_back_2','word_plus_1','word_plus_2']

baseline2a_features = baseline1_features + ['stemmed_word','stemmed_word_back_1','stemmed_word_back_2',
                                            ## 'stemmed_rel_back_1', 'stemmed_rel_back_2',  ## not used (for %)
                                            'stemmed_word_plus_1','stemmed_word_plus_2',
                                            ## 'stemmed_rel_plus1','stemmed_rel_plus_2',  ## not used (for %)
                                           '1_before_support','2_or_less_before_support','3_or_less_before_support',
                                           '1_after_support','2_or_less_after_support','3_or_less_after_support',
                                           '1_before_conj','2_or_less_before_conj','3_or_less_before_conj',
                                           '1_after_conj','2_or_less_after_conj','3_or_less_after_conj',
                                           '1_before_pred','2_or_less_before_pred','3_or_less_before_pred',
                                           '1_after_pred','2_or_less_after_pred','3_or_less_after_pred']

three_word_window_features = ['stemmed_word_back_1','stemmed_word_back_2',
                                            ## 'stemmed_rel_back_1', 'stemmed_rel_back_2',  ## not used (for %)
                                            'stemmed_word_plus_1','stemmed_word_plus_2',
                                            ## 'stemmed_rel_plus1','stemmed_rel_plus_2',  ## not used (for %)
                                           '1_before_support','2_or_less_before_support','3_or_less_before_support',
                                           '1_after_support','2_or_less_after_support','3_or_less_after_support',
                                           '1_before_conj','2_or_less_before_conj','3_or_less_before_conj',
                                           '1_after_conj','2_or_less_after_conj','3_or_less_after_conj',
                                           '1_before_pred','2_or_less_before_pred','3_or_less_before_pred',
                                           '1_after_pred','2_or_less_after_pred','3_or_less_after_pred']

baseline2b_features = baseline2a_features + ['pred_path','support_path']

system3_features = baseline2b_features + ['top_5_embed_unigram','top_4_embed_unigram',
                                         'top_3_embed_unigram','top_2_embed_unigram','best_embed_unigram',
                                         'top_5_embed_back_bigram','top_4_embed_back_bigram',
                                         'top_3_embed_back_bigram','top_2_embed_back_bigram',
                                         'best_embed_back_bigram',  'top_5_embed_back_trigram',
                                         'top_4_embed_back_trigram','top_3_embed_back_trigram',
                                         'top_2_embed_back_trigram', 'best_embed_back_trigram',
                                         'top_5_embed_forward_bigram','top_4_embed_forward_bigram',
                                         'top_3_embed_forward_bigram','top_2_embed_forward_bigram',
                                         'best_embed_forward_bigram',  'top_5_embed_forward_trigram',
                                         'top_4_embed_forward_trigram','top_3_embed_forward_trigram',
                                         'top_2_embed_forward_trigram', 'best_embed_forward_trigram']

# def  run_with_feature_set(feature_set):
#     ...

baseline2a_word_prefixes = ['1_after','1_before',
                            '1_before_pred','1_after_pred',
                            '1_before_support', '1_after_support',
                            '2_or_less_before', '2_or_less_after',
                            '2_or_less_before_pred', '2_or_less_after_pred',
                            '2_or_less_before_support', '2_or_less_after_support'
                            '3_or_less_before', '3_or_less_after',
                            '3_or_less_before_pred', '3_or_less_after_pred',
                            '3_or_less_before_support', '3_or_less_after_support']


ngram_similarity_features_old = ['unigram_embed_similarity', 'slash_unigram_embed_similarity',
                                         'forward_bigram_ARG1_embed_similarity', 'back_bigram_ARG1_embed_similarity',
                                         'forward_bigram_ARG1_embed_slash_similarity', 'back_bigram_ARG1_embed_slash_similarity',
                                         'forward_trigram_ARG1_embed_similarity', 'back_trigram_ARG1_embed_similarity']

ngram_similarity_features = ['unigram_embed_similarity', 'slash_unigram_embed_similarity',
                                         'forward_bigram_embed_similarity', 'back_bigram_embed_similarity',
                                         'forward_bigram_embed_slash_similarity', 'back_bigram_embed_slash_similarity',
                                         'forward_trigram_embed_similarity', 'back_trigram_embed_similarity']

system3b_features = ngram_similarity_features

token_distance_features = ['before_pred', 'after_pred', 'before_support', 'after_support','before_prep', \
                           'after_prep', 'before_conj', 'after_conj']

system5a_features = system3_features + system3b_features
system5b_features = baseline2b_features + system3b_features

## for the next few features, I will need to create new embedding files

ARG1_embed_slash_features =  ['unigram_ARG1_embed_slash_similarity','back_bigram_ARG1_embed_slash_similarity','back_trigram_ARG1_embed_slash_similarity',  'forward_bigram_ARG1_embed_slash_similarity','forward_trigram_ARG1_embed_slash_similarity']

embed_slash_features =  ['unigram_embed_slash_similarity','back_bigram_embed_slash_similarity','back_trigram_embed_slash_similarity',  'forward_bigram_embed_slash_similarity','forward_trigram_embed_slash_similarity']


embed_sequence_features_old = ['embed_sequence_similarity_pred_to_arg1','embed_sequence_similarity_support_to_arg1']

embed_sequence_features = ['embed_sequence_similarity_pred_to_arg','embed_sequence_similarity_support_to_arg']

system5c_features = system5b_features + ARG1_embed_slash_features ## thus assumes that system5a_features are not better than system_5a_features

system5d_features = system5c_features +embed_sequence_features  

numeric_valued_features = system3b_features + ARG1_embed_slash_features + embed_sequence_features + token_distance_features

system5e_features = system5a_features + token_distance_features + system3b_features

def list_minus(big_list,little_list):
    out = []
    for item in big_list:
        if not item in little_list:
            out.append(item)
    return(out)

numeric_valued_features_only_5 = list_minus(system5e_features,['top_5_embed_unigram','top_4_embed_unigram',
                                         'top_3_embed_unigram','top_2_embed_unigram','best_embed_unigram',
                                         'top_5_embed_back_bigram','top_4_embed_back_bigram',
                                         'top_3_embed_back_bigram','top_2_embed_back_bigram',
                                         'best_embed_back_bigram',  'top_5_embed_back_trigram',
                                         'top_4_embed_back_trigram','top_3_embed_back_trigram',
                                         'top_2_embed_back_trigram', 'best_embed_back_trigram',
                                         'top_5_embed_forward_bigram','top_4_embed_forward_bigram',
                                         'top_3_embed_forward_bigram','top_2_embed_forward_bigram',
                                         'best_embed_forward_bigram',  'top_5_embed_forward_trigram',
                                         'top_4_embed_forward_trigram','top_3_embed_forward_trigram',
                                         'top_2_embed_forward_trigram', 'best_embed_forward_trigram'])

## remove baseline2a_word_prefixes prefixes

#### components defined ******

## baseline2a_word_prefixes  -- specific before/after counts
## token_distance_features -- new before/after counts
## embed_sequence_features -- new embed_features

shared_base = baseline1_features + baseline1_word_features

path_features = ['pred_path','support_path']

embed_features = ['top_5_embed_unigram','top_4_embed_unigram',
                                         'top_3_embed_unigram','top_2_embed_unigram','best_embed_unigram',
                                         'top_5_embed_back_bigram','top_4_embed_back_bigram',
                                         'top_3_embed_back_bigram','top_2_embed_back_bigram',
                                         'best_embed_back_bigram',  'top_5_embed_back_trigram',
                                         'top_4_embed_back_trigram','top_3_embed_back_trigram',
                                         'top_2_embed_back_trigram', 'best_embed_back_trigram',
                                         'top_5_embed_forward_bigram','top_4_embed_forward_bigram',
                                         'top_3_embed_forward_bigram','top_2_embed_forward_bigram',
                                         'best_embed_forward_bigram',  'top_5_embed_forward_trigram',
                                         'top_4_embed_forward_trigram','top_3_embed_forward_trigram',
                                         'top_2_embed_forward_trigram', 'best_embed_forward_trigram',
                                          'top_5_embed_slash_unigram','top_4_embed_slash_unigram',
                                         'top_3_embed_slash_unigram','top_2_embed_slash_unigram','best_embed_slash_unigram',
                                         'top_5_embed_slash_back_bigram','top_4_embed_slash_back_bigram',
                                         'top_3_embed_slash_back_bigram','top_2_embed_slash_back_bigram',
                                         'best_embed_slash_back_bigram',  'top_5_embed_slash_back_trigram',
                                         'top_4_embed_slash_back_trigram','top_3_embed_slash_back_trigram',
                                         'top_2_embed_slash_back_trigram', 'best_embed_slash_back_trigram',
                                         'top_5_embed_slash_forward_bigram','top_4_embed_slash_forward_bigram',
                                         'top_3_embed_slash_forward_bigram','top_2_embed_slash_forward_bigram',
                                         'best_embed_slash_forward_bigram',  'top_5_embed_slash_forward_trigram',
                                         'top_4_embed_slash_forward_trigram','top_3_embed_slash_forward_trigram',
                                         'top_2_embed_slash_forward_trigram', 'best_embed_slash_forward_trigram',
                                         'unigram_embed_similarity', 'slash_unigram_embed_similarity',
                                         'forward_bigram_ARG1_embed_similarity', 'back_bigram_ARG1_embed_similarity',
                                         'forward_bigram_ARG1_embed_slash_similarity', 'back_bigram_ARG1_embed_slash_similarity',
                                         'forward_trigram_ARG1_embed_similarity', 'back_trigram_ARG1_embed_similarity',
                  'forward_bigram_embed_similarity', 'back_bigram_embed_similarity',
                                         'forward_bigram_embed_slash_similarity', 'back_bigram_embed_slash_similarity',
                                         'forward_trigram_embed_similarity', 'back_trigram_embed_similarity']

## used???
embed_features_minus_prefix = ['unigram_embed_similarity', 'slash_unigram_embed_similarity',
                                         'forward_bigram_ARG1_embed_similarity', 'back_bigram_ARG1_embed_similarity',
                                         'forward_bigram_ARG1_embed_slash_similarity', 'back_bigram_ARG1_embed_slash_similarity',
                                         'forward_trigram_ARG1_embed_similarity', 'back_trigram_ARG1_embed_similarity']

all_unprefixed_features = baseline1_features+ baseline1_word_features + ['stemmed_word'] \
    + three_word_window_features + path_features + embed_features + token_distance_features + \
    embed_sequence_features + ngram_similarity_features

### best so far as of Jun 7 
all_unprefixed_features_minus_token_distance = baseline1_features+ baseline1_word_features + ['stemmed_word'] \
    + three_word_window_features + path_features + embed_features_minus_prefix\
    + embed_sequence_features + ngram_similarity_features

edited_token_distance_features = ['before_pred', 'after_pred', 'before_support', 'after_support',
                                  ## 'before_prep', 'after_prep', \
                                  ## 'before_conj', 'after_conj'\
]

all_unprefixed_features_minus_after_prep = baseline1_features+ baseline1_word_features + ['stemmed_word'] \
    + three_word_window_features + path_features + embed_features + \
    embed_sequence_features + ngram_similarity_features +edited_token_distance_features

embed_features = ['top_5_embed_unigram','top_4_embed_unigram',
                                         'top_3_embed_unigram','top_2_embed_unigram','best_embed_unigram',
                                         'top_5_embed_back_bigram','top_4_embed_back_bigram',
                                         'top_3_embed_back_bigram','top_2_embed_back_bigram',
                                         'best_embed_back_bigram',  'top_5_embed_back_trigram',
                                         'top_4_embed_back_trigram','top_3_embed_back_trigram',
                                         'top_2_embed_back_trigram', 'best_embed_back_trigram',
                                         'top_5_embed_forward_bigram','top_4_embed_forward_bigram',
                                         'top_3_embed_forward_bigram','top_2_embed_forward_bigram',
                                         'best_embed_forward_bigram',  'top_5_embed_forward_trigram',
                                         'top_4_embed_forward_trigram','top_3_embed_forward_trigram',
                                         'top_2_embed_forward_trigram', 'best_embed_forward_trigram',
                                          'top_5_embed_slash_unigram','top_4_embed_slash_unigram',
                                         'top_3_embed_slash_unigram','top_2_embed_slash_unigram','best_embed_slash_unigram',
                                         'top_5_embed_slash_back_bigram','top_4_embed_slash_back_bigram',
                                         'top_3_embed_slash_back_bigram','top_2_embed_slash_back_bigram',
                                         'best_embed_slash_back_bigram',  'top_5_embed_slash_back_trigram',
                                         'top_4_embed_slash_back_trigram','top_3_embed_slash_back_trigram',
                                         'top_2_embed_slash_back_trigram', 'best_embed_slash_back_trigram',
                                         'top_5_embed_slash_forward_bigram','top_4_embed_slash_forward_bigram',
                                         'top_3_embed_slash_forward_bigram','top_2_embed_slash_forward_bigram',
                                         'best_embed_slash_forward_bigram',  'top_5_embed_slash_forward_trigram',
                                         'top_4_embed_slash_forward_trigram','top_3_embed_slash_forward_trigram',
                                         'top_2_embed_slash_forward_trigram', 'best_embed_slash_forward_trigram']


all_unprefixed_features_minus_token_distance = baseline1_features+ baseline1_word_features + ['stemmed_word'] \
    + three_word_window_features + path_features + embed_features + \
    embed_sequence_features + ngram_similarity_features

all_unprefixed_features = baseline1_features+ baseline1_word_features + ['stemmed_word'] \
    + three_word_window_features + path_features + embed_features + token_distance_features + \
    embed_sequence_features + ngram_similarity_features

all_unprefixed_features_no_embed = baseline1_features+ baseline1_word_features + ['stemmed_word'] \
    + three_word_window_features + path_features + token_distance_features + \
    ngram_similarity_features

all_unprefixed_features_no_n_gram_sim = baseline1_features+ baseline1_word_features + ['stemmed_word'] \
    + three_word_window_features + path_features + token_distance_features + embed_features

all_unprefixed_features_no_embed_or_sim = baseline1_features+ baseline1_word_features + ['stemmed_word'] \
    + three_word_window_features + path_features + token_distance_features

baseline_jun9 = baseline1_features+ baseline1_word_features + ['stemmed_word'] \
    + three_word_window_features + path_features

baseline_jun9_2 = baseline1_features+ baseline1_word_features + ['stemmed_word'] \
    + three_word_window_features + path_features + embed_features \
    + embed_sequence_features + ngram_similarity_features

partitive_classes = ['ENVIRONMENT', 'PARTITIVE-QUANT', 'GROUP', 'NOM', 'PARTITIVE-PART', \
                     'MERONYM', 'PART-OF-BODY-FURNITURE-ETC', 'SHARE', 'BOOK-CHAPTER', 'BORDER', \
                     'DIVISION', 'WORK-OF-ART', 'CONTAINER', 'INSTANCE-OF-SET', 'NOMADJ']

baseline_jun16_1 = baseline_jun9_2 + partitive_classes

##  baseline1_features -- most basic
## baseline2a_features --> + stemming + true/false close to support, conj, pred
## baseline2b_features --> + ['pred_path','support_path']
## system3_features --> + basic embed features
##### system5a_features --> system3_features+ ngram_similarity_features
###### system5b_features = same except no basic embed features
###### system5c and 5d -- more variation re embed features

## 0) alll    
## 1) baseline2a_word_prefixes -- specific words with after/before pred pews, support
## 2) features_to_keep=baseline_jun9 -- baseline1_features+ baseline1_word_features + ['stemmed_word'] \
##     + three_word_window_features + path_features
## 3_ features_to_keep=baseline_jun9_2 + ngram_similarity_features
## 4)  features_to_keep=baseline_jun16_1  + partitive_classes

sets_to_test =[['base1',baseline1_features],
               ['base2',baseline2a_features],  ## no gain
               ['base3',baseline2b_features],
               ['base4',baseline2b_features+ partitive_classes], ## no gain
               ['base5',baseline2b_features+ partitive_classes+embed_features+embed_sequence_features+ngram_similarity_features]## no gain
]

sets_to_test2 = [['base1',baseline1_features],
                 ['base2a',baseline1_features+['pred_path','support_path']],
                 ['base5a',baseline1_features+['pred_path','support_path']+partitive_classes]
                 ]

sets_to_test3 = [['path_only',['pred_path','support_path']],
                 ['path_and_word_class',['pred_path','support_path']+partitive_classes]]

sets_to_test4 = \
                [['base1',baseline1_features],
                 ['base2',baseline1_features +partitive_classes],
                 ['base3', baseline1_features +partitive_classes+embed_features+embed_sequence_features+ngram_similarity_features]
                 ]

def remove_duplicates(inlist):
    return (list(set(inlist)))

sets_to_test5 = \
    [['base4',baseline2b_features],
     ['base5',system3_features],
     ['base6',remove_duplicates(system3_features+embed_features)]]
     


## need to add alltype_classes ## 7th column of all_nombank.clean.train
## sorted,split and uniqued

alltype_classes = ['GOOD', 'ABILITY', 'FACE', 'ACCESS', 'ACCUMULATE', 'ACCUSE', 'ACHIEVE', 'ACKNOWLEDGE', 'ACQUIRE', 'ACT', 'ACTIVITY', 'ACTREL', 'ADAPT', 'ADD', 'ADDITIONAL', 'ADHERE', 'ADMIRE', 'ADMIT', 'ADOPT', 'ADVANTAGE', 'ADVERTISE', 'AFFECT', 'AFFECTION', 'AFFIX', 'AGREE', 'AGRICULTURE', 'AIM', 'ALERT', 'ALIGN', 'ALLOCATE', 'ALLOW', 'ALLY', 'ALTER', 'AMASS', 'AMENABLE', 'AMUSE', 'ANNIVERSARY', 'ANNOUNCE', 'ANNOY', 'ANTI', 'APPEARANCE', 'APPLAUD', 'APPLY', 'APPROVE', 'ARG', 'ARGUE', 'HORSE', 'ASK', 'ASKING', 'ASPECTUAL', 'ASSENT', 'ASSERT', 'ASSESS', 'ASSIGN', 'ASSOCIATE', 'EASE', 'HAND', 'WORK', 'ATHLETICS', 'ATTACH', 'ATTAIN', 'ATTEMPT', 'ATTEND', 'ATTEST', 'ATTRACT', 'ATTRIBUTE', 'AUCTION', 'AUGMENT', 'BACK', 'BANISH', 'BASEBALL', 'BASIC', 'BATTLE', 'BE', 'BOTHER', 'RENTER', 'AFLUSTER', 'ATTRACTIVE', 'CONTRITE', 'DEFEATED', 'BE DIFFERENT', 'DULL', 'ILL', 'CERTAIN', 'SHELTER', 'SEND', 'INVOLVED', 'LEFT', 'LOUDLY', 'THAN', 'IMPORTANCE', 'PARALLEL', 'PART', 'PREPARED', 'RELATED', 'SHORT', 'STRONGLY', 'SUBSEQUENT', 'SUCCESSFUL', 'VERY', 'WORTHY', 'BECOME', 'BEFOUL', 'BEFRIEND', 'BEG', 'BEGIN', 'BEHAVIOR', 'BELIEVE', 'BELONG', 'BID', 'BILL', 'BITCH', 'BLEND', 'BLESS', 'BLOCK', 'BODY', 'BOND', 'BOOK', 'BORDER', 'BREAK', 'BRING', 'BROADCAST', 'BUILD', 'BUSINESS', 'BUY', 'BUYING', 'CALCULATE', 'CALL', 'CALLING', 'CALM', 'CARESS', 'CARRY', 'CAST', 'CATCH', 'CAUSE', 'CEASE', 'CF', 'CHALLENGE', 'CHANGE', 'CHARGE', 'CHASTISE', 'CHECK', 'CHOOSE', 'CLAP', 'CLOSE', 'COHORT', 'COLLECT', 'COMBINE', 'COME', 'COMMERCE', 'COMMIT', 'COMMUNICATE', 'COMPARE', 'COMPATIBLE', 'COMPETE', 'COMPETITION', 'COMPLAIN', 'COMPLAINT', 'COMPLEX', 'COMPRISE', 'CONCEAL', 'CONFINE', 'CONFIRM', 'CONSTRUCT', 'CONSULTANT', 'CONSUME', 'CONTAINER', 'CONTEMPLATE', 'CONTINUE', 'CONTROL', 'CONVINCE', 'COOPERATE', 'COPY', 'COUNT', 'COURT', 'COVER', 'CREATE', 'CREDIT', 'CRISSCROSS', 'CRITICIZE', 'CUT', 'DANCE', 'DEAL', 'DECEIVE', 'DECIDE', 'DECIMAL', 'DECLINE', 'DEDUCE', 'DEFAULT', 'DEFECT', 'DEFEND', 'DEFREL', 'DEMAND', 'DESCRIBE', 'DESIGN', 'DESPERATE', 'DESPISE', 'DESS', 'DESTROY', 'DETER', 'DETERMINE', 'DIFFERENT', 'DIG', 'DIRECTED', 'DISAGREE', 'DISAPPOINT', 'DISCOURSE', 'DISCOVER', 'DISCOVERY', 'DISCRETELY', 'DISEASE', 'DISPLAY', 'DISRUPT', 'DISSEMINATE', 'DISSENT', 'DISTRIBUTE', 'DIVIDE', 'DIVISION', 'DO', 'DOUBT', 'DREAM', 'DRIVE', 'DURATION', 'EARN', 'EAT', 'ELECT', 'EMIT', 'EMOTE', 'EMPHASIZE', 'ENACT', 'END', 'ENGAGE', 'ENHANCE', 'ENJOY', 'ENSURE', 'ENTER', 'ENUMERATE', 'ENVIRONMENT', 'EQUAL', 'EQUIVALENT', 'ERUPTION', 'ESTABLISH', 'ESTEEM', 'EVENT', 'EX', 'EXAGGERATE', 'EXAMINE', 'EXCHANGE', 'EXCLUDE', 'EXIT', 'EXPECT', 'EXPEL', 'EXPERIENCE', 'EXPLAIN', 'EXPLORE', 'EXPOSE', 'EXPRESS', 'EXTEND', 'EXTRACT', 'FAIL', 'FAIR', 'FALL', 'FEAR', 'FIELD', 'FIGHT', 'FILL', 'FINANCIAL', 'FIND', 'FIRE', 'FLAG', 'FLY', 'FOCUS', 'FOLLOW', 'SERVICES', 'FORCE', 'FREE', 'FRIGHTEN', 'FUNDRAISING', 'GET', 'COMMODITY', 'BETTER', 'BIGGER', 'PREDECESSOR', 'MONEY', 'RID', 'SMALLER', 'TEMPORARILY', 'TOGETHER', '', 'GIVE', 'GO', 'GOV', 'GOVERNMENT', 'GRASP', 'GROUP', 'GROW', 'GUESS', 'HALLMARK', 'HANDLE', 'HANG', 'HAPPENING', 'RELATION', 'RESPONSIBILITY', 'DATE', 'MONOPOLY', 'PARTY', 'SENSATION', 'SUSPICION', 'TENDENCY', 'EFFECT', 'FIRST', 'POWER', 'SWAY', 'HELP', 'HESITATE', 'HINDER', 'HIT', 'HOLD', 'HOLE', 'HONOR', 'HOPE', 'HOUSE', 'HOUSING', 'HUG', 'IDIOM', 'IGNORE', 'ILLEGAL', 'IMAGINE', 'IMPEL', 'IMPELLED', 'IMPLEMENT', 'IMPROVE', 'INTEREST', 'ADDITION', 'CONSTRAST', 'EXISTANCE', 'PLACE', 'STORE', 'TIME', 'TOUCH', 'X', 'INCLUDE', 'INCREASE', 'INCUR', 'INEXPERIENCED', 'INGEST', 'INITIATE', 'INJURE', 'INSTANCE', 'INTANS', 'INTEND', 'INTENTIONAL', 'INTERNATIONALIZE', 'INTRANS', 'INTRODUCE', 'INTUIT', 'INVESTIGATE', 'INVITE', 'ISSUE', 'JOB', 'JOIN', 'JUDGE', 'JUDGEMENT', 'JUSTIFICATION', 'JUSTIFIED', 'JUSTIFY', 'KEEP', 'KILL', 'LABEL', 'LAST', 'LAUGH', 'LAW', 'LAWSUIT', 'LAY', 'LEAVE', 'LEDGER', 'LEG', 'LEGAL', 'LEND', 'LENGTHEN', 'LET', 'LEVEL', 'LIE', 'LIFT', 'LIGHT', 'CRIMINAL', 'LION', 'MAGAZINE', 'NEW', 'LISTEN', 'LIVE', 'LOCATION', 'LOOK', 'LOOKS', 'LOSE', 'LOWER', 'MAKE', 'MAKES', 'MANNER', 'MANUFACTURED', 'MARK', 'MARKETPLACE', 'MATCH', 'MATHEMATICS', 'MEANING', 'MEASURE', 'MEDICAL', 'MEDICINE', 'MEET', 'MEMBER', 'MERONYM', 'METONYM', 'MILITARY', 'MINOR', 'MIX', 'MODE', 'MODIFIED', 'MONOMANIA', 'MOTION', 'MOVE', 'MULTIPLY', 'NAG', 'NAME', 'NEED', 'NEGATIVE', 'LONGER', 'NOM', 'NOMADJ', 'NOMADJLIKE', 'NOMADLIKE', 'NOMADV', 'NOMADVLIKE', 'NOMING', 'NOMLIKE', 'NON', 'OBEY', 'SINK', 'SUCCEED', 'NOUN', 'OBJECT', 'OBSCURE', 'OBSERVE', 'OCCUR', 'OFFICIAL', 'MARKET', 'TARGET', 'TOP', 'OPEN', 'OPERATE', 'OPINION', 'OPPOSE', 'QUESTION', 'PAINTING', 'PAPERWORK', 'PARTICIPATE', 'PARTITIVE', 'PASS', 'PASTIME', 'PATH', 'PAY', 'PAYMENT', 'PERFORM', 'PERMIT', 'PERSON', 'PERSONAL', 'PERSUADE', 'PHRASAL', 'PHYSICAL', 'PHYSICALLY', 'PICK', 'PILLAGE', 'PISS', 'PLAN', 'PLAY', 'POINT', 'PORTRAY', 'POSITION', 'POSSESS', 'POSSESSION', 'POSSIBLE', 'PRACTICE', 'PRAISE', 'PREPARE', 'PREPOSITIONAL', 'PRETEND', 'PREVENT', 'PROGNOSTICATE', 'PROHIBIT', 'PROMISE', 'PROOF', 'PROPOSED', 'PROSPER', 'PROTECT', 'PROTOTYPE', 'PROVE', 'PROVIDE', 'PROVOKE', 'PUBLISH', 'PULL', 'PURCHASE', 'PURSUE', 'PUSH', 'PUSHING', 'PUT', 'QUALITATIVE', 'QUANTIFIABLE', 'QUANTIFY', 'QUIT', 'QUOTE', 'RAISE', 'RANDOMIZE', 'RE', 'REACH', 'READ', 'RECALCULATE', 'RECIP', 'RECIPRICAL', 'RECOMMEND', 'RECORD', 'RECOVER', 'REDUCE', 'REFRAIN', 'REFUSE', 'REGISTER', 'REGRET', 'REJECT', 'RELEASE', 'RELIEVE', 'RELIGIOUS', 'RELY', 'REMAIN', 'REMOVE', 'REORGANIZE', 'REPAIR', 'REPEL', 'REPLACE', 'REPORT', 'REPUDIATE', 'REQUEST', 'REQUIRE', 'REQUIRES', 'RESENT', 'RESIDE', 'RESIST', 'RESOLVE', 'RESTRAIN', 'RESULTED', 'REVENUE', 'REVIEW', 'REVISE', 'RISE', 'RISK', 'ROT', 'ROTATION', 'RUN', 'SAVE', 'SAY', 'SCALE', 'SEARCH', 'SECURE', 'SEEM LIKE', 'SEIZE', 'SELECT', 'SELL', 'SENSE', 'SEPARATE', 'SERVE', 'SET', 'SHAKE', 'SHARE', 'SHIELD', 'SHINE', 'SHIP', 'SHOW', 'SHUT', 'SIGNIFY', 'SILENT', 'SIMILAR', 'SIN', 'SING', 'SITE', 'SLICE', 'SLIDE', 'SMEAR', 'SOCIAL', 'WEIRD', 'SOUND', 'SPEAK', 'SPECIALIZE', 'SPEED', 'SPOON', 'SPORTING', 'STAGNATE', 'STAND', 'START', 'STATE', 'STAY', 'STERBEN', 'STICK', 'STOCK', 'STOP', 'STRAY', 'STRETCH', 'STRIKE', 'STROKE', 'STRUGGLE', 'STUDENT', 'STUDY', 'SUBJECT', 'SUBMIT', 'SUBPART', 'SUBSTITUTE', 'SUBTRACT', 'SUCK', 'SUDDEN', 'SUFFER', 'SUGGEST', 'SUMMON', 'SUPERLATIVE', 'SUPERVISE', 'SUPPORT', 'SURPASS', 'SURPRISE', 'SURVEY', 'SURVIVE', 'SWARM', 'SWEAR', 'SWEAT', 'SYNCHRONIZE', 'TAKE', 'TALK', 'TEACH', 'TELL', 'TEMPT', 'TERMINATE', 'THANK', 'THERE', 'THING', 'THINK', 'THOUGHT', 'THREATEN', 'THROW', 'TITLE', 'ABANDON', 'ACCOMMODATE', 'ACCUSTOM', 'ADDRESS', 'ADVICE', 'AMOUNT', 'APPROACH', 'ARRANGE', 'ASSAULT', 'ASSIST', 'ASSUME', 'ATTACK', 'CONCERNED', 'BRANCH', 'CONDUCT', 'CONNECT', 'CONSIDER', 'CONTAMINATE', 'CULTIVATE', 'CURE', 'DECLARE', 'DECREASE', 'DELIVER', 'DEPRESS', 'DEPRIVE', 'DIRECT', 'DISCUSS', 'DISFIGURE', 'DRAW', 'DROP', 'EDIT', 'ENTRUST', 'EVALUATE', 'EXERCISE', 'FEEL', 'FERTILIZE', 'FORBID', 'FORM', 'FORMALLY', 'GAIN', 'HAPPEN', 'HARM', 'HUNT', 'IMPOSE', 'INTERPRET', 'INTERRUPT', 'INTRUDE', 'INVOLVE', 'ISOLATE', 'MEDDLE', 'MESS', 'MISS', 'MONITOR', 'MURDER', 'NEGOTIATE', 'OBTAIN', 'OPPRESS', 'ORDER', 'PACK', 'PRESENT', 'PRESERVE', 'PROCEED', 'PRONOUNCE', 'PROPOSE', 'PURIFY', 'REAP', 'REGARD', 'RENDER', 'RESTORE', 'REVOKE', 'RUB', 'SCATTER', 'SICKEN', 'SIGN', 'SPEND', 'STRENGTHEN', 'SURROUND', 'SUSPEND', 'TEST', 'TRANSLATE', 'TRANSMIT', 'TREAT', 'UNDERGO', 'UNITE', 'URGE', 'USE', 'VIEW', 'WEAKEN', 'WIN', 'WRITE', 'TUNE', 'TOLERATE', 'TRADE', 'TRANSACTION', 'TRANSFER', 'TRANSFORM', 'TRANSITIVE', 'TRAVEL', 'TREMBLE', 'TRY', 'TURN', 'TWO', 'TYPE', 'UNBALANCED', 'UNDERSTAND', 'UNFILL', 'UP', 'UPDATE', 'UPHEAVAL', 'UPPER', 'UPSET', 'UTTER', 'VERSION', 'VICTIM', 'VIOLENCE', 'VISION', 'VISIT', 'WAGER', 'WAGES', 'WAIT', 'WAIVE', 'WALK', 'WANT', 'WASH', 'WATCH', 'WATER', 'WEAR', 'WIPE', 'WISH', 'WITHDRAW', 'WITTY', 'WORLD', 'WORRY', 'WORSEN', 'WRECK']

## note that there were 2104 classes after initial normalization
## then we chose the first non-closed-class word instead
## reducing the number to 778


sets_for_all_test = [['path_only',['pred_path','support_path']],
                     ['path+alltype_classes',['pred_path','support_path']+alltype_classes],
                     ['Plus_baseline1',['pred_path','support_path']+alltype_classes+baseline1_features],
                     ['Plus_embed',['pred_path','support_path']+alltype_classes+baseline1_features+embed_features+embed_sequence_features+ngram_similarity_features]]


sets_for_all_test2 = [['baseline1',baseline1_features]]

def get_all_features_from_feature_file(infile):
    ## purpose is to get all features of the form X=Y
    ## we are only collecting the X part (feature), not eht Y part (value)
    import os
    set_of_features = set()
    with open(infile) as instream:
        for line in instream:
            line = line.strip(os.linesep)
            line_list = line.split('\t')
            for fv in line_list:
                if '=' in fv:
                    feat,value = fv.split('=')
                    set_of_features.add(feat)
        set_of_features = list(set_of_features)
        set_of_features.sort()
    return(set_of_features)

# def get_part_features():
#     import os
#     all_part_features = get_all_features_from_feature_file(os.path.join("part","train.features"))
#     return(all_part_features)

# all_part_features = get_part_features()

# def sort_part_features():
#     import re
#     before_after = []
#     BIO_POS_word = []
#     rel_support_distance = []
#     path_features = []
#     embed_feats =[]
#     embed_slash = []
#     embed_not_slash = []
#     rest = []
#     for feature in all_part_features:
#         if re.search('^[0-9]_((before)|(after))',feature,re.I):
#             before_after.append(feature)
#         elif re.search('^[0-9]_or_((less)|(more))_((before)|(after))',feature,re.I):
#             before_after.append(feature)
#         elif re.search('^((before)|(after))',feature):
#             before_after.append(feature)
#         elif re.search('^((BIO)|(POS))',feature,re.I):
#             BIO_POS_word.append(feature)
#         elif re.search('^(stemmed)?_?word',feature,re.I):
#             BIO_POS_word.append(feature)
#         elif re.search('_path$',feature,re.I):
#             path_features.append(feature)
#         elif re.search('^(rel_)|(support_)',feature,re.I):
#             rel_support_distance.append(feature)
#         elif re.search('_embed',feature,re.I):
#             embed_feats.append(feature)
#             if re.search('slash',feature,re.I):
#                 embed_slash.append(feature)
#             else:
#                 embed_not_slash.append(feature)
#         else:
#             rest.append(feature)
#     return(before_after,BIO_POS_word,rel_support_distance,path_features,embed_feats,rest,embed_slash,embed_not_slash)

# before_after,BIO_POS_word,rel_support_distance,path_features,embed_feats,pred_classes,embed_slash,embed_not_slash = sort_part_features()

# new_test_set =[['basic',BIO_POS_word+['word']],
#               ['plus_ngram',BIO_POS_word+before_after+['word']],
#               ['plus_ngram2',BIO_POS_word+before_after+rel_support_distance+['word']],
#               ['plus_paths',BIO_POS_word+before_after+rel_support_distance+path_features+['word']],
#                ['plus_pred_classes', BIO_POS_word+before_after+rel_support_distance+path_features+pred_classes+['word']],
#                ['plus_embed',BIO_POS_word+before_after+rel_support_distance+path_features+pred_classes+embed_feats+['word']],
#                ['all',BIO_POS_word+before_after+rel_support_distance+path_features+pred_classes+embed_feats+['word']],
#                ['all_but_word',BIO_POS_word+before_after+rel_support_distance+path_features+pred_classes+embed_feats],
#                ['all_but_ngram2',BIO_POS_word+before_after+path_features+pred_classes+embed_feats+['word']],
#                ['all_but_ngram2_and_embed',BIO_POS_word+before_after+path_features+pred_classes+['word']],
#                 ['all_but_ngram2_but_only_not_slash_embed',BIO_POS_word+before_after+path_features+pred_classes+['word']+embed_not_slash],
#                 ['all_but_ngram2_but_only_slash_embed',BIO_POS_word+before_after+path_features+pred_classes+['word']+embed_slash]
# ]

# new_test = [['all_but_ngram2_but_only_slash_embed',BIO_POS_word+before_after+path_features+pred_classes+['word']+embed_slash]]

# new_test_set2 =[['basic',BIO_POS_word+['word']],
#               ['plus_ngram',BIO_POS_word+before_after+['word']],
#                ['plus_pred_classes', BIO_POS_word+before_after+pred_classes+['word']],
#                 ['plus_paths', BIO_POS_word+before_after+path_features+pred_classes+['word']],
#                ['plus_embed',BIO_POS_word+before_after+path_features+pred_classes+embed_feats+['word']],
#                ['all',BIO_POS_word+before_after+path_features+pred_classes+embed_feats+['word']],
#                ['all_but_word',BIO_POS_word+before_after+path_features+pred_classes+embed_feats],
#                ['all_but_embed',BIO_POS_word+before_after+path_features+pred_classes+['word']],
#                 ['all_but_slash_embed',BIO_POS_word+before_after+path_features+pred_classes+['word']+embed_not_slash],
#                 ['all_but_not_slash_embed',BIO_POS_word+before_after+path_features+pred_classes+['word']+embed_slash],
#                 ['all_but_path',BIO_POS_word+before_after+pred_classes+embed_feats+['word']]]

# extra_test =   [['plus_pred_classes', BIO_POS_word+before_after+pred_classes+['word']],
#                 ['plus_paths', BIO_POS_word+before_after+path_features+pred_classes+['word']]
#     ]

# simple_tests = [['word',['word']],['basic_ngram_including_word',BIO_POS_word+before_after+['word']], ['word_and_pred_only',BIO_POS_word+pred_classes+['word']],['embed_only',embed_feats],['path_only',path_features],['path_and_word',path_features+['word']]]

