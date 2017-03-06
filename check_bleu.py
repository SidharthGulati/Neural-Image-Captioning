import nltk.tokenize
import nltk
import pickle

# This file is to check the BLEU score of a given image id.

# pickle file which saves a dictionary which has key as image id and its generated
# captions as the values.It also stores the bleu score for that image_id
caption_res = pickle.load(open("val2014_results_bleu.p",
                               "rb"))
id_2_cap = pickle.load(open("val2014_image_id_2_cap.p", "rb"))

# The image id for which you want to see the generated captions
image_id = 1340

res = []
temp = caption_res[image_id]
for k in range(len(temp) - 1):
    res.append(nltk.tokenize.word_tokenize(temp[k].lower()))

print("The labels for the image: ", id_2_cap[image_id])

for k in range(len(res)):
    BLEUscore = nltk.translate.bleu_score.sentence_bleu(id_2_cap[image_id], res[k])
    print("The bleu score: ", temp[k], BLEUscore)
