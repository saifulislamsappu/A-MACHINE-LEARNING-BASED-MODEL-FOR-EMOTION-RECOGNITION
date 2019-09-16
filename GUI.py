import tkinter as tk
from tkinter import *

import nltk
from nltk.corpus import stopwords  # for removing stop word
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize  # tokenize sentence

from data_train import Naive_bayes_Algo
from graph import graph_min
from graph_result import value_accuracy

nltk.download('wordnet')

root = tk.Tk()
root.title("Emotion Detector")
root.geometry("850x400+150+150")

# Label text
entry_label = tk.Label(root, text="INPUT TEXT: ", font=('Helvetica', 16, 'bold'))
entry_label.grid(row=0, column=0, padx=30)

# Entry Field
user_entry = tk.Entry(root, width=50)
user_entry.grid(row=0, column=4, padx=30, pady=10, ipady=8)

# Text Field
text_box = tk.Text(root, width=55, height=10)
text_box.grid(row=1, column=4)
text_box.insert("end-1c", "Emotion Will Appear Here.")


# Clear Button
def clear_data():
    user_entry.delete(0, END)


cl_btn = tk.Button(root, text='Clear', command=clear_data)
cl_btn.grid(row=0, column=5)


# ROC Graph
def graph():
    graph_min()


cl_btn = tk.Button(root, text='Roc Graph', command=graph)
cl_btn.grid(row=20, column=0, padx=30, pady=10)


btn = tk.Button(root, text="Accuracy", width=8, command=value_accuracy)
btn.grid(row=20, column=4, pady=20)


# Exit Button
btn = tk.Button(root, text="Quit", width=8, command=root.quit)
btn.grid(row=20, column=5, pady=20)


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


def test(data, class_unique_word, class_word, unique_word, class_probability_value):
    result = []
    stop_word = set(stopwords.words('english')).union('?', '.', ',')
    lemmatizer = WordNetLemmatizer()
    for i in data:
        i = clean_text(i)
        word = word_tokenize(i.lower())
        filtered_sentence = [lemmatizer.lemmatize(w) for w in word if w not in stop_word and not w.isdigit()]
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
                    ans[j] = ans[j] * ((class_unique_word[j][k] + .01) / total_current_class_word)
                else:
                    if k in unique_word.keys():
                        ans[j] = ans[j] * (.01 / total_current_class_word)

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

    result = test(data, class_unique_word, class_word, unique_word, class_probability_value)
    return result


def guess_number(event=None):
    guess = user_entry.get()
    result = test_result([guess])
    if class_value_check == 0:
        text_box.delete(1.0, "end-1c")  # Clears the text box of data
        text_box.insert("end-1c", str(result[0]))  # adds text to text box

    else:
        text_box.delete(1.0, "end-1c")
        text_box.insert("end-1c", "Cannot detect perfectly for lack of emotion related keyword")
        user_entry.delete(0, "end")


n = Naive_bayes_Algo()
data_info = n.train_data()
result = test_result(n.xtest)

user_entry.bind("<Return>", guess_number)
root.mainloop()