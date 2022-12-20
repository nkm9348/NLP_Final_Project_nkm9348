import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from tools.arg_scorer import score_file_with_NNP_adjustment
from tree import *

# # Params for percentage task
# train_filename = "percentage/train.data"
# train_parse_filename = "gold_parses/%_nombank.clean.train.gold_parse"
# train_baseline_filename = "percentage/train.features"
# dev_filename = "percentage/dev.data"
# dev_parse_filename = "gold_parses/%_nombank.clean.dev.gold_parse"
# dev_baseline_filename = "percentage/dev.features"
# dev_out_filename = "percentage/dev.out"

# Params for partitive task
train_filename = "part/train.data"
train_parse_filename = "gold_parses/partitive_group_nombank.clean.train.gold_parse"
train_baseline_filename = "part/train.features"
dev_filename = "part/dev.data"
dev_parse_filename = "gold_parses/partitive_group_nombank.clean.dev.gold_parse"
dev_baseline_filename = "part/dev.features"
dev_out_filename = "part/dev.out"


# Stemmer

def load_stemming_dictionary(infile):
    stem_dict = dict()
    with open(infile) as instream:
        for line in instream:
            line = line.strip(os.linesep)
            outlist = line.lower().split(',')
            word = outlist[0]
            if word not in stem_dict:
                stem_dict[word] = outlist[1]
    return stem_dict

stemming_dict: dict[str, str] = load_stemming_dictionary(os.path.join('tools/morph-base-mapping.csv'))

def get_stemmed_word(word):
    global stemming_dict
    lower_word = word.lower()
    if lower_word in stemming_dict:
        return stemming_dict[lower_word]
    else:
        return lower_word

# Partitive normalizer

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


def get_path_features(filename, parse_filename):
    _paths = []

    with open(filename) as f:
        data = [x.strip() for x in f.readlines()]

    data_idx = 0

    with open(parse_filename) as f:
        buf = [x.strip() for x in f.readlines()]

    sentence_cnt = 0
    parse_file_cnt = 0
    no_dest_node = 0
    while parse_file_cnt < len(buf):
        # Ignore the header
        while parse_file_cnt < len(buf) and len(buf[parse_file_cnt]) > 0:
            parse_file_cnt += 1
        while parse_file_cnt < len(buf) and len(buf[parse_file_cnt]) == 0:
            parse_file_cnt += 1

        # Read the parse tree
        parse_tree = ""
        while parse_file_cnt < len(buf) and len(buf[parse_file_cnt]) > 0:
            parse_tree += buf[parse_file_cnt]
            parse_file_cnt += 1
        parse_tree = parse_tree[1:-1]

        while parse_file_cnt < len(buf) and len(buf[parse_file_cnt]) == 0:
            parse_file_cnt += 1

        tree = build_tree(parse_tree)
        leaves = find_leaves(tree, [])

        sentence_words = []
        while len(data[data_idx]) > 0:
            sentence_words.append(data[data_idx].split("\t"))
            data_idx += 1

        while data_idx < len(data) and len(data[data_idx]) == 0:
            data_idx += 1

        assert len(leaves) == len(sentence_words)

        # POS of parent and grandparent
        tmp = [dict() for _ in range(len(sentence_words))]
        for i in range(len(sentence_words)):
            tmp[i]["parent_pos"] = ""
            tmp[i]["grandparent_pos"] = ""
            line = sentence_words[i]
            parent = leaves[int(line[3])].parent
            if parent is not None:
                tmp[i]["parent_pos"] = parent.pos
                gran = parent.parent
                if gran is not None:
                    tmp[i]["grandparent_pos"] = gran.pos


        # Find paths from each word to pred
        dest_node = None
        for line in sentence_words:
            if len(line) > 5 and line[5] == "PRED":
                dest_node = leaves[int(line[3])]
                break

        if dest_node is None:
            no_dest_node += 1
        for i in range(len(sentence_words)):
            line = sentence_words[i]
            if dest_node is None:
                tmp[i]["pred_path"] = ""
            else:
                tmp[i]["pred_path"] = find_path_between_nodes(leaves[int(line[3])], dest_node, tree)

        # Find paths from each word to supprort
        dest_node = None
        for line in sentence_words:
            if len(line) > 5 and line[5] == "SUPPORT":
                dest_node = leaves[int(line[3])]
                break

        for i in range(len(sentence_words)):
            line = sentence_words[i]
            if dest_node is None:
                tmp[i]["support_path"] = ""
            else:
                tmp[i]["support_path"] = find_path_between_nodes(leaves[int(line[3])], dest_node, tree)

        sentence_cnt += 1

        _paths += tmp

    print(f"{no_dest_node} sentences with no predicate")

    paths = pd.DataFrame(_paths)

    return paths

def get_other_baseline_features(feature_filename, use_cache=False):
    if use_cache and os.path.exists(f"{feature_filename}.pkl"):
        return pd.read_pickle(f"{feature_filename}.pkl")
    useless_feature_list = set(["POS", "BIO", "stemmed_word", "word_back_1", "stemmed_word_back_1", "POS_back_1", "BIO_back_1", "word_back_2", "stemmed_word_back_2", "POS_back_2", "BIO_back_2", "word_plus_1", "stemmed_word_plus_1", "POS_plus_1", "BIO_plus_1", "word_plus_2", "stemmed_word_plus_2", "POS_plus_2", "BIO_plus_2", "word_plus_3", "stemmed_word_plus_3", "POS_plus_3", "BIO_plus_3", "REL_plus_1", "REL_plus_2", "REL_plus_3", "REL_back_1", "REL_back_2", "pred_path", "support_path"])
    with open(feature_filename) as f:
        buf = [x.strip() for x in f.readlines()]
    buf = [x for x in buf if len(x) > 0]

    _features = []
    _feature_freq = dict()
    for line in buf:
        feature_dict = dict()
        for x in line.split("\t")[1:]:
            _x = x.split("=")
            if len(_x) != 2:
                continue
            k = _x[0]
            v = _x[1]
            if k in useless_feature_list:
                continue
            else:
                feature_dict[k] = v
                if k not in _feature_freq:
                    _feature_freq[k] = 0
                _feature_freq[k] += 1
        _features.append(feature_dict)

    print("Constructing DataFrame")
    features = pd.DataFrame(_features, columns=[k for k, v in _feature_freq.items() if v >= 50])  # Delete useless features
    print(features.shape)

    for column in features.columns:
        value_count = features[column].value_counts()
        if value_count.shape[0] == 1:
            features[column] = (~features[column].isna()).astype(int)
        elif column == "relation_feature":
            features[column] = features[column].fillna("")
        else:
            features[column] = features[column].fillna(0).astype(np.float64)

    features.to_pickle(f"{feature_filename}.pkl")

    return features


def get_data(filename, parse_filename, baseline_filename):
    _data = []

    print("Reading file")

    with open(filename) as infile:
        buf = infile.read().splitlines()

    prefix = ["prev2_", "prev_", "", "next_", "next2_", "next3_"]

    print("Generating basic fields")

    sentence_words = [["SENTENCE_BREAK"] * 5] * 2
    for _line in buf:
        if len(_line) == 0:
            sentence_words += [["SENTENCE_BREAK"] * 5] * 3
            stem_words = [get_stemmed_word(x[0]) for x in sentence_words]
            found_pred = "False"
            for i in range(0, len(sentence_words) - 5):
                tmp = {"arg1": 0}
                for j in range(6):
                    tmp[prefix[j] + "word"] = sentence_words[i + j][0]
                    tmp[prefix[j] + "stem"] = stem_words[i + j]
                    tmp[prefix[j] + "bio"] = sentence_words[i + j][2]
                    tmp[prefix[j] + "pos"] = sentence_words[i + j][1]
                    if len(sentence_words[i + j]) > 5 and sentence_words[i + j][5] != "ARG1":
                            tmp[prefix[j] + "rel"] = sentence_words[i + j][5]

                line_features = sentence_words[i + 2]
                tmp["token_number"] = line_features[3]
                tmp["sentence_number"] = line_features[4]
                if len(line_features) > 5:
                    if line_features[5] == "PRED":
                        found_pred = "True"
                    if line_features[5] == "ARG1":
                        tmp["arg1"] = 1
                tmp["right_to_pred"] = found_pred

                _data.append(tmp)

            sentence_words = [["SENTENCE_BREAK"] * 5] * 2
        else:
            sentence_words.append(_line.split("\t"))

    str_data = pd.DataFrame(_data)

    print("Generating baseline features")
    numeric_data = get_other_baseline_features(baseline_filename, use_cache=True)
    relation_feature = numeric_data["relation_feature"]
    del numeric_data["relation_feature"]

    print("Generating path features")
    path_features = get_path_features(filename, parse_filename)
    assert path_features.shape[0] == str_data.shape[0]

    str_data = pd.concat([str_data, relation_feature, path_features], axis=1)

    # Sort data columns in alphabetical order
    # This is extremely important!!!
    str_data = str_data[sorted(str_data.columns)]
    numeric_data = numeric_data[sorted(numeric_data.columns)]

    return str_data, numeric_data

str_data, numeric_data = get_data(train_filename, train_parse_filename, train_baseline_filename)

y = str_data["arg1"]
del str_data["arg1"]

enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2)
scaler = StandardScaler(with_mean=False)

X = scaler.fit_transform(enc.fit_transform(str_data))
X = hstack((X, csr_matrix(numeric_data.values)))
print(X.shape)
print(y.shape)

# clf = AdaBoostClassifier(random_state=0, n_estimators=100)
clf = LogisticRegression(random_state=0, max_iter=100)
clf.fit(X, y)


def apply_to_test_data(in_filename, in_parse_filename, in_baseline_filename, out_filename):
    test_str_data, test_numeric_data = get_data(in_filename, in_parse_filename, in_baseline_filename)

    del test_str_data["arg1"]

    # Align columns of dev set and training set
    test_numeric_data = pd.concat([test_numeric_data, pd.DataFrame(index=range(test_numeric_data.shape[0]), columns=list(set(numeric_data.columns).difference(set(test_numeric_data.columns)))).fillna(0)], axis=1)
    test_str_data = test_str_data[str_data.columns]
    test_numeric_data = test_numeric_data[numeric_data.columns]

    test_X = scaler.transform(enc.transform(test_str_data))
    test_X = hstack((test_X, csr_matrix(test_numeric_data.values)))
    print(test_X.shape)

    test_y = clf.predict(test_X)

    with open(in_filename) as f:
        buf = f.read().splitlines()
    with open(out_filename, "w") as outf:
        i = 0
        res = []
        for _line in buf:
            if len(_line) == 0:
                res.append("\n")
            else:
                res.append("\t".join(_line.split("\t")[:5] + (["ARG1"] if test_y[i] > 0 else [])) + "\n")

                i += 1

        outf.writelines(res)


print("Applying to dev set")

apply_to_test_data(dev_filename, dev_parse_filename, dev_baseline_filename, dev_out_filename)

score_file_with_NNP_adjustment(dev_filename, dev_out_filename, "arg1")

