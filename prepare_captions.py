import json
import pickle
import nltk.tokenize

# File to prepare a dictionary which store image id as the key and its captions as the values
with open('captions_val2014.json') as json_data:
    d = json.load(json_data)

d = d['annotations']
res = {}
for item in d:
    if item['image_id'] not in res:
        res[item['image_id']] = []
    res[item['image_id']].append(str(item['caption']))

new_res = {}
for item in res:
    new_res[item] = []
    for item2 in res[item]:
        new_res[item].append(nltk.tokenize.word_tokenize(item2.lower()))
pickle.dump(new_res, open('val2014_image_id_2_cap.p', "wb"))

