import nltk
from nltk.corpus import stopwords  # for removing stop word
from nltk.tokenize import word_tokenize  # tokenize sentence
from data_train import Naive_bayes_Algo
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import re
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle


# from nltk.stem import PorterStemmer


def clean_text(text):
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


def test(data, class_list, class_unique_word, class_word, unique_word, class_probability_value):
    result = []
    stop_word = set(stopwords.words('english')).union('?', '.', ',')
    lemmatizer = WordNetLemmatizer()
    for i in data:
        i = clean_text(i)
        word_main = word_tokenize(i.lower())
        print("Word token",word_main)
        sen = ' '.join(word_main)
        print('sen',sen)
        
        tagged_sentence = nltk.tag.pos_tag(sen.split())
        
        print("tagged sentance",tagged_sentence)
        
        edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS' and tag != 'NN']
        print("edited_sentence",edited_sentence)
        
        filtered_sentence = [lemmatizer.lemmatize(w) for w in edited_sentence if w not in stop_word and not w.isdigit()]
        
        print("Filtered_sentence",filtered_sentence)
        # ================= Remove noun ===============================================
#        sen = ' '.join(filtered_sentence)
#        tagged_sentence = nltk.tag.pos_tag(sen.split())
#        [('I', 'PRP'), ('am', 'VBP'), ('named', 'VBN'), ('John', 'NNP'), ('Doe', 'NNP')]
#        sentence = "I am named John Doe"
#        tagged_sentence = nltk.tag.pos_tag(sentence.split())
        
#        edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS' and tag != 'NN']
#        print(' '.join(edited_sentence))
#        print(' '.join(edited_sentence))

#        print(sen)
#        print("edit sentence RAkib",edited_sentence)
#        print("Filter sentence RAkib",filtered_sentence)
#        print("tag sentence lll",tagged_sentence)
        # =============================================================================
        #       bigram
        # =============================================================================
        bigram = []
        bigram = filtered_sentence
        word_data = ' '.join(filtered_sentence)
        nltk_tokens = nltk.word_tokenize(word_data)

        a = list(nltk.bigrams(nltk_tokens))
        for j in a:
            bigram.append(' '.join(j))

        ans = {}
        for j in class_unique_word:
            total_current_class_word = class_word[j]
            ans[j] = class_probability_value[j]

            for k in bigram:
                if k in class_unique_word[j].keys():
                    ans[j] = ans[j] * ((class_unique_word[j][k] + .01) / (total_current_class_word))
                #                    print('#######'+str(class_unique_word[j][k]))
                else:
                    if k in unique_word.keys():
                        ans[j] = ans[j] * ((.01) / (total_current_class_word))

        #                print(ans[j],j)
        global class_value_check
        class_value_check = 0
        value = 0
        class_name = []
        for class_result in ans:
            if ans[class_result] >= value:
                value = ans[class_result]
                class_name.append(class_result)
                if value == class_probability_value[class_result]:
                    class_value_check = 1
                else:
                    class_value_check = 0

        result.append(class_name[-1])
    return result


def test_result(data):
    global class_word
    global class_unique_word
    global unique_word
    global class_probability_value
    global class_list

    class_word = data_info['class_word']
    class_unique_word = data_info['class_unique_word']
    unique_word = data_info['unique_word']
    class_probability_value = data_info['class_probability_value']
    class_list = [i for i in class_word.keys()]

    result = test(data, class_list, class_unique_word, class_word, unique_word, class_probability_value)
    #    print (data)
    #    print (result)
    return result


global data_info

n = Naive_bayes_Algo()
print('LOADING...............................')
data_info = n.train_data()
print('TRAINING COMPLETE')
result = test_result(n.xtest)
real_result = list(n.ytest)
l = len(real_result)
correct = 0
t = 0
for i in range(l):
    if real_result[i] == result[i]:
        correct += 1
print('Error :' + str(100 - (correct / l) * 100))
print('Correct Answer :' + str((correct / l) * 100))
print('F1 Score: ' + str(f1_score(result, real_result, average='macro')))

confusion_matrix = {}
for i in class_list:
    confusion_matrix[i] = {}
for i in class_list:
    for j in confusion_matrix:
        confusion_matrix[j][i] = 0
# print(confusion_matrix)
#
for i in range(l):
    if result[i] in confusion_matrix[real_result[i]].keys():
        confusion_matrix[real_result[i]][result[i]] += 1

print('              #fear  #anger  #sadness  #disgust  #shame  #guilt  #joy')

print('#fear         ' + str(confusion_matrix['fear']['fear']) + '        ' + str(
    confusion_matrix['fear']['anger']) + '        ' + str(confusion_matrix['fear']['sadness']) + '        ' + str(
    confusion_matrix['fear']['disgust']) + '        ' + str(confusion_matrix['fear']['shame']) + '        ' + str(
    confusion_matrix['fear']['guilt']) + '        ' + str(confusion_matrix['fear']['joy']))
print('#anger        ' + str(confusion_matrix['anger']['fear']) + '        ' + str(
    confusion_matrix['anger']['anger']) + '        ' + str(confusion_matrix['anger']['sadness']) + '        ' + str(
    confusion_matrix['anger']['disgust']) + '        ' + str(confusion_matrix['anger']['shame']) + '        ' + str(
    confusion_matrix['anger']['guilt']) + '        ' + str(confusion_matrix['anger']['joy']))
print('#sadness      ' + str(confusion_matrix['sadness']['fear']) + '        ' + str(
    confusion_matrix['sadness']['anger']) + '        ' + str(confusion_matrix['sadness']['sadness']) + '        ' + str(
    confusion_matrix['sadness']['disgust']) + '        ' + str(confusion_matrix['sadness']['shame']) + '        ' + str(
    confusion_matrix['sadness']['guilt']) + '        ' + str(confusion_matrix['sadness']['joy']))
print('#disgust      ' + str(confusion_matrix['disgust']['fear']) + '        ' + str(
    confusion_matrix['disgust']['anger']) + '        ' + str(confusion_matrix['disgust']['sadness']) + '        ' + str(
    confusion_matrix['disgust']['disgust']) + '        ' + str(confusion_matrix['disgust']['shame']) + '        ' + str(
    confusion_matrix['disgust']['guilt']) + '        ' + str(confusion_matrix['disgust']['joy']))
print('#shame        ' + str(confusion_matrix['shame']['fear']) + '        ' + str(
    confusion_matrix['shame']['anger']) + '        ' + str(confusion_matrix['shame']['sadness']) + '        ' + str(
    confusion_matrix['shame']['disgust']) + '        ' + str(confusion_matrix['shame']['shame']) + '        ' + str(
    confusion_matrix['shame']['guilt']) + '        ' + str(confusion_matrix['shame']['joy']))
print('#guilt        ' + str(confusion_matrix['guilt']['fear']) + '        ' + str(
    confusion_matrix['guilt']['anger']) + '        ' + str(confusion_matrix['guilt']['sadness']) + '        ' + str(
    confusion_matrix['guilt']['disgust']) + '        ' + str(confusion_matrix['guilt']['shame']) + '        ' + str(
    confusion_matrix['guilt']['guilt']) + '        ' + str(confusion_matrix['guilt']['joy']))
print('#joy          ' + str(confusion_matrix['joy']['fear']) + '        ' + str(
    confusion_matrix['joy']['anger']) + '        ' + str(confusion_matrix['joy']['sadness']) + '        ' + str(
    confusion_matrix['joy']['disgust']) + '        ' + str(confusion_matrix['joy']['shame']) + '        ' + str(
    confusion_matrix['joy']['guilt']) + '        ' + str(confusion_matrix['joy']['joy']))

# =============================================================================
#       ROC
# =============================================================================
class_name = ['fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt', 'joy']

real_result_binary = []
predict_result_binary = []
for i in real_result:
    array_value = []
    for j in class_name:
        if i == j:
            array_value.append(1)
        else:
            array_value.append(0)
    real_result_binary.append(array_value)

for i in result:
    array_value = []
    for j in class_name:
        if i == j:
            array_value.append(1)
        else:
            array_value.append(0)
    predict_result_binary.append(array_value)

import numpy as np

real_result_binary = np.array(real_result_binary)
predict_result_binary = np.array(predict_result_binary)

fpr = dict()
tpr = dict()
roc_auc = dict()
class_size = len(class_name)
for i in range(class_size):
    fpr[class_name[i]], tpr[class_name[i]], _ = roc_curve(real_result_binary[:, i], predict_result_binary[:, i])

    roc_auc[class_name[i]] = auc(fpr[class_name[i]], tpr[class_name[i]])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(real_result_binary.ravel(), predict_result_binary.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in class_name]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in class_name:
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= class_size

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'yellow', 'blue', 'green', 'purple'])
lw = 2
for i, color in zip(class_name, colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

single_data = input('USER: ')
while single_data != 'exit':
    result = test_result([single_data])
    if class_value_check == 0:
        print('Emotion :' + result[0])
    else:
        print('Emotion : Cannot detect perfectly for lack of emotion related keyword')

    single_data = input('USER: ')
