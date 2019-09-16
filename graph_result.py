import tkinter as tk
from tkinter import *
from tkinter import ttk

import nltk
from nltk.corpus import stopwords  # for removing stop word
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize  # tokenize sentence
from sklearn.metrics import f1_score

from data_train import Naive_bayes_Algo
# nltk.download('wordnet')


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
        word = word_tokenize(i.lower())
        filtered_sentence = [lemmatizer.lemmatize(w) for w in word if w not in stop_word and not w.isdigit()]
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
    return result


def value_accuracy():
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

    # Result pop up
    def popup_bonus():
        value_error = str(100 - (correct / l) * 100)
        value_correct = str((correct / l) * 100)
        value_f1_score = str(f1_score(result, real_result, average='macro'))
        win = tk.Toplevel()
        win.wm_title("Accuracy Result")

        win.title("Emotion Detector Result Accuracy")
        win.geometry("450x300+150+150")

        # Label text
        pop_label = tk.Label(win, text="Result Accuracy", font=('Helvetica', 18, 'bold'))
        pop_label.grid(row=0, column=1, padx=30)

        # Text Field
        result_box = tk.Text(win, width=40, height=10)
        result_box.grid(row=1, column=1, padx=60)
        result_box.insert("end-1c", "Error = "+value_error+"\n")
        result_box.insert("end-1c", "Correct = "+value_correct+"\n")
        result_box.insert("end-1c", "F1 Score = "+value_f1_score+"\n")

        b = ttk.Button(win, text="Okay", command=win.destroy)
        b.grid(row=2, column=1, pady=20)

    popup_bonus()
