from nltk.corpus import stopwords  # for removing stop word
from nltk.tokenize import word_tokenize  # tokenize sentence
import numpy as np
import nltk
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer

# from sklearn.datasets import make_classification
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score
# from matplotlib import pyplot
# from nltk.stem import PorterStemmer


nltk.download('stopwords')
nltk.download('punkt')


class Naive_bayes_Algo():
    data = pd.read_csv('EmotionPhrases.csv', header=None).as_matrix()
    x = data[1:, 1]
    y = data[1:, 0]
    #    print('X value =',x,'Y value =',y)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.1, train_size=.9, random_state=0)

    # ===================================================================
    # Text Cleaning Method
    # ===================================================================

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'i\'m', 'i am', text)
        text = re.sub(r'he\'s', 'he is', text)
        text = re.sub(r'she\'s', 'she is', text)
        text = re.sub(r'it\'s', 'it is', text)
        text = re.sub(r'that\'s', 'that is', text)
        text = re.sub(r'what\'s', 'what is', text)
        text = re.sub(r'who\'s', 'who is', text)
        text = re.sub(r'how\'s', 'how is', text)
        text = re.sub(r'hows', 'how is', text)
        text = re.sub(r'where\'s', 'where is', text)
        text = re.sub(r'\'re', ' are', text)

        text = re.sub(r'isn\'t', 'is not', text)
        text = re.sub(r'isnt', 'is not', text)
        text = re.sub(r'aint', 'is not', text)
        text = re.sub(r'ain\'t', 'is not', text)
        text = re.sub(r'aren\'t', 'are not', text)
        text = re.sub(r'arent', 'are not', text)
        text = re.sub(r'wasn\'t', 'was not', text)
        text = re.sub(r'wasnt', 'was not', text)
        text = re.sub(r'weren\'t', 'were not', text)

        text = re.sub(r'don\'t', 'do not', text)
        text = re.sub(r'dont', 'do not', text)
        text = re.sub(r'doesn\'t', 'does not', text)
        text = re.sub(r'doesnt', 'does not', text)
        text = re.sub(r'didn\'t', 'did not', text)
        text = re.sub(r'didnt', 'did not', text)

        text = re.sub(r'\'ve', ' have', text)
        text = re.sub(r'haven\'t', 'have not', text)
        text = re.sub(r'hasn\'t', 'has not', text)
        text = re.sub(r'hadn\'t', 'had not', text)

        text = re.sub(r'mustn\'t', 'must not', text)

        text = re.sub(r'can\'t', 'can not', text)
        text = re.sub(r'cannot', 'can not', text)
        text = re.sub(r'cant', 'can not', text)
        text = re.sub(r'couldn\'t', 'could not', text)

        text = re.sub(r'shan\'t', 'shall not', text)
        text = re.sub(r'shant', 'shall not', text)
        text = re.sub(r'shouldn\'t', 'should not', text)

        text = re.sub(r'won\'t', 'will not', text)
        text = re.sub(r'wont', 'will not', text)
        text = re.sub(r'wouldn\'t', 'would not', text)
        text = re.sub(r'wouldnt', 'would not', text)

        text = re.sub(r'\'ll', ' will', text)
        text = re.sub(r'\'d', ' would', text)

        text = re.sub(r'gonna', 'going to', text)
        text = re.sub(r'wanna', 'want to', text)

        text = re.sub(r'let\'s', 'let us', text)
        text = re.sub(r'lets', 'let us', text)

        text = re.sub(r'[-`=~!@#$%&*()_+{}\|;:\'",.<>/?]', ' ', text)
        text = re.sub(r'[[]', ' ', text)
        text = re.sub(r'[]]', ' ', text)
        return text

    def class_probability(self, data, class_size):
        class_probability_value = {}  # for storing class probability value
        class_value = {}
        for i in data:  # calculate total number of appearence of a class
            if i in class_value.keys():
                class_value[i] = class_value[i] + 1
            else:
                class_value[i] = 1
        for i in class_value:  # calculate probabilty
            class_probability_value[i] = class_value[i] / class_size
        return class_probability_value, class_value

    def word_count_under_class(self, data):
        class_unique_word = {}
        unique_word = {}
        class_word = {}
        unique_word_count = 0
        class_unique_word_count = {}
        for i in data:
            if i[1] in class_unique_word.keys():
                for j in i[0]:
                    if j in class_unique_word[i[1]].keys():
                        class_unique_word[i[1]][j] = class_unique_word[i[1]][j] + 1
                        unique_word[j] = unique_word[j] + 1
                    else:
                        class_unique_word[i[1]][j] = 1
                        if j in unique_word.keys():
                            unique_word[j] = unique_word[j] + 1
                        else:
                            unique_word[j] = 1
                            unique_word_count = unique_word_count + 1
                        class_unique_word_count[i[1]] = class_unique_word_count[i[1]] + 1
                    class_word[i[1]] = class_word[i[1]] + 1
            else:
                class_unique_word_count[i[1]] = 0
                class_word[i[1]] = 0
                class_unique_word[i[1]] = {}
                for j in i[0]:
                    if j in class_unique_word[i[1]].keys():
                        class_unique_word[i[1]][j] = class_unique_word[i[1]][j] + 1
                        unique_word[j] = unique_word[j] + 1
                    else:
                        class_unique_word[i[1]][j] = 1
                        if j in unique_word.keys():
                            unique_word[j] = unique_word[j] + 1
                        else:
                            unique_word[j] = 1
                            unique_word_count = unique_word_count + 1
                        class_unique_word_count[i[1]] = class_unique_word_count[i[1]] + 1
                class_word[i[1]] = 1
        return class_word, class_unique_word, unique_word, unique_word_count, class_unique_word_count

    def preprocess_data(self, data):

        stop_word = set(stopwords.words('english')).union('?', '.', ',')
        process_data = []
        lemmatizer = WordNetLemmatizer()
        for i in data:
            i[0] = self.clean_text(i[0])
            word = word_tokenize(i[0].lower())

            filtered_sentence = [lemmatizer.lemmatize(w) for w in word if w not in stop_word and not w.isdigit()]
#            print("Filter sentence",filtered_sentence)
            # =============================================================================
            #       bigram
            # =============================================================================
            bigram = []
            for j in filtered_sentence:
                bigram.append(j)
            word_data = ' '.join(filtered_sentence)
            nltk_tokens = nltk.word_tokenize(word_data)

            a = list(nltk.bigrams(nltk_tokens))
            for j in a:
                bigram.append(' '.join(j))

            process_data.append([bigram, i[1]])

        return process_data

    def train_data(self):
        train_data = [[self.x[i], self.y[i]] for i in range(len(self.xtrain))]
        data_new = self.preprocess_data(train_data)

        class_word, class_unique_word, unique_word, unique_word_count, class_unique_word_count = self.word_count_under_class(
            data_new)
        data_2 = np.array(self.data)
        class_probability_value, class_value = self.class_probability(data_2[:, 0], len(self.y))

        global info
        info = {}
        info['class_word'] = class_word
        info['class_unique_word'] = class_unique_word
        info['unique_word'] = unique_word
        info['unique_word_count'] = unique_word_count
        info['class_unique_word_count'] = class_unique_word_count
        info['class_probability_value'] = class_probability_value
        info['class_value'] = class_value
        return info
