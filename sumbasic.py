from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import operator
import sys
import os

# 1. orig: The original version, including the non-redundancy update of the word scores.
# 2. best-avg: A version of the system that picks the sentence that has the highest average probability
# in Step 2, skipping Step 3.
# 3. simplified: A simplied version of the system that holds the word scores constant and does not
# incorporate the non-redundancy update.
WORDS_NUM = 100


def load_cluster(path):
    dirname = os.path.dirname(path)
    filename_regx = os.path.basename(path)
    files_all = os.listdir(dirname)

    cluster = []
    for file in files_all:
        if file.startswith(filename_regx.split('*')[0]):
            cluster.append(dirname + '/' + file)
    return cluster


def get_tokens(sents, lemmatize=True, rm_stopwords=True):
    tokens = word_tokenize(''.join(i for i in  sents if ord(i) < 128))
    tokens = [t.lower() for t in tokens]
    wordnet_lemmatizer = WordNetLemmatizer()
    if lemmatize: tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    if rm_stopwords: tokens = [t for t in tokens if t not in stopwords.words('english')]
    return tokens


def get_probabilities(cluster):
    word_prob_dict = {}
    token_count = 0
    for path in cluster:
        with open(path) as f:
            tokens = get_tokens(f.read().replace('\n', ' '))
            token_count += len(tokens)
            for token in tokens:
                word_prob_dict.update({token: word_prob_dict.get(token, 0) + 1.0})
    for word_prob in word_prob_dict:
        word_prob_dict[word_prob] /= float(token_count)
    return word_prob_dict


def get_sentences(cluster):
    sentences = []
    for path in cluster:
        with open(path) as f:
            article = f.read().replace('\n',' ')
            sentences.extend(sent_tokenize(''.join(i for i in  article if ord(i) < 128)))
    return sentences


def get_sentence_score(sentence, word_prob_dict):
    if len(sentence.split(' ')) == 0:
        return 0
    score = 0.0
    num_tokens = 0
    tokens = get_tokens(sentence)
    for token in tokens:
        if token in word_prob_dict:
            score += word_prob_dict[token]
            num_tokens += 1
    return score / num_tokens


def update_prob_dict(max_sentence, word_prob_dict):
    tokens = get_tokens(max_sentence)
    for word in tokens:
        word_prob_dict[word] **= 2


#
def get_max_sentence(sentences, word_prob_dict, simplified=False):
    max_sentence = ""
    max_score = float('-inf')
    for sentence in sentences:
        score = get_sentence_score(sentence, word_prob_dict)
        if score > max_score:
            max_sentence = sentence
            max_score = score
    if not simplified: update_prob_dict(max_sentence, word_prob_dict)
    return max_sentence


# get the highest-frequency-word from the dictionary
def get_hpw(word_prob_dict):
    return max(word_prob_dict.iteritems(), key=operator.itemgetter(1))[0]


# compute rouge-1 score for a given summary with a reference summary
def get_rouge_score(s_ref, s):
    tokens_ref = get_tokens(s_ref)
    tokens = get_tokens(s)
    count = 0.0
    for token in tokens:
        if token in tokens_ref:
            count += 1.0

    return count / len(s)


def get_sentence_with_hpw(sentences, word_prob_dict):
    subset_sent = []
    hpw = get_hpw(word_prob_dict)
    for s in sentences:
        if hpw in get_tokens(s):
            subset_sent.append(s)
    return subset_sent


# orig method
def orig(cluster):
    summary = ""
    sentences = get_sentences(cluster)
    word_prob_dict = get_probabilities(cluster)
    #print word_prob_dict
    word_count = 0
    while word_count < WORDS_NUM:
        sents_candidate = get_sentence_with_hpw(sentences, word_prob_dict)
        sentence = get_max_sentence(sents_candidate, word_prob_dict)
        if word_count + len(sentence.split(' ')) < WORDS_NUM:
            summary += sentence + ' '
            # Remove the sentence added to the summary so that it won't get selected again.
            #sentences.remove(sentence)
            word_count += len(sentence.split(' '))
        else:
            break
    return summary


# best_avg method
def best_avg(cluster):
    summary = ""
    sentences = get_sentences(cluster)
    word_prob_dict = get_probabilities(cluster)
    word_count = 0
    while word_count < WORDS_NUM:
        sentence = get_max_sentence(sentences, word_prob_dict)
        if word_count + len(sentence.split(' ')) < WORDS_NUM:
            summary += sentence + ' '
            # Remove the sentence added to the summary so that it won't get selected again.
            #sentences.remove(sentence)
            word_count += len(sentence.split(' '))
        else:
            break
    return summary


def simplified(cluster):
    summary = ""
    sentences = get_sentences(cluster)
    word_prob_dict = get_probabilities(cluster)
    word_count = 0
    while word_count < WORDS_NUM:
        sents_candidate = get_sentence_with_hpw(sentences, word_prob_dict)
        sentence = get_max_sentence(sents_candidate, word_prob_dict, True)
        if word_count + len(sentence.split(' ')) < WORDS_NUM:
            summary += sentence + ' '
            # Remove the sentence added to the summary so that it won't get selected again.
            sentences.remove(sentence)
            word_count += len(sentence.split(' '))
        else:
            break
    return summary


# article selection component for the leading method
# the article with the maximum length would be selected
def select_file(files):
    len_files = []
    for file in files:
        with open(file) as f:
            len_files.append((len(f.read()), file))
    return max(len_files)


# leading method
def leading(cluster):
    #_, leading_file = select_file(cluster)
    word_count = 0
    summary = ""
    sentence_by_article = []
    for article in cluster:
        sentence_by_article.append(get_sentences([article, ]))
    while word_count < WORDS_NUM:
        for sents_by_article in sentence_by_article:
            sentence = sents_by_article[0]
            sents_by_article.remove(sentence)
            s = sentence.split(' ')
            if word_count + len(s) < WORDS_NUM:
                summary += sentence + ' '
                word_count += len(s)
            else:
                return summary
    return summary


if __name__ == '__main__':
    method = sys.argv[1]
    cluster = load_cluster(sys.argv[2])

    # s_leading =  leading(cluster)
    # s_orig = orig(cluster)
    # s_best_avg = best_avg(cluster)
    # s_simplified = simplified(cluster)
    #
    # print s_leading
    # for s in [s_orig, s_best_avg, s_simplified]:
    #     print "Rouge score:" + str(get_rouge_score(s_leading, s))
    #     print s

    if method == 'orig':
        print orig(cluster)
    elif method == 'best-avg':
        print best_avg(cluster)
    elif method == 'simplified':
        print simplified(cluster)
    else:
        print "UNKNOWN METHOD"
