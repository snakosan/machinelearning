import json
import pprint
import os
from nltk.tokenize import wordpunct_tokenize

dir = './files'
def count_df(text, use_nltk=True):
    freq = {}
    if use_nltk:
        full_list = wordpunct_tokenize(text)
    else:
        full_list = []
        for x in text.split('\n'):
            full_list.extend(x.split(' '))

    for word in full_list:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq

def index_dir(dir):
    tf_json = {}
    for file_name in os.listdir(dir):
        if os.path.isfile(os.path.join(dir,file_name)):
            text_file = os.path.join(dir, file_name)
            with open(text_file) as f:
                frq = count_df(f.read())
                tf_json[file_name] = frq
    return tf_json

tf_json = index_dir(dir)
with open('index.json', 'w') as f:
    json.dump(tf_json,f)
