# import libiaries
import numpy
import json
import re
import pandas as pd
import multiprocessing
import numpy as np
from nltk.corpus import stopwords
nltk.download('stopwords')

def process_yelp_line(line):
	'''read yelp review and rating'''

	# conver the text line to a json object
	json_object = json.loads(line)

	# read and tokenize the text
	text = json_object['text']

	# read the label and convert to an integer
	label = int(json_object['stars'])

	# return the text and the label
	if text:
		return text, label
	else:
		return None

def normalize(texts,labels):
	'''tokenize and normalize'''

	stop_words = set(stopwords.words('english')) 

	# convert to dataframe
	data = pd.DataFrame({'text': texts, 'label': labels})

	# keep only numbers, letters and space
	data['text'] = data.apply(lambda t: re.sub(r'[^0-9A-Za-z ]', '', str(t['text'])), axis=1)

	# remove stopwords and convert to lower case
	data['text'] = data.apply(lambda r: ' '.join(w.lower() for w in r['text'].split() if w.lower() not in stop_words),axis=1)

	# discard NA reviews
	data = data.dropna()

	# assign binary labels - 1 for rating>=3: good rating
	data['binary_label'] = data.apply(lambda r: 0 if r['label'] < 4 else 1, axis=1)
	
	# store the dataframe
	data.to_csv('review_cleaned.csv')

	return data

if __name__ == "__main__":
	# read the first 500,000 yelp reviews
	lines = open('yelp_academic_dataset_review.json', encoding="utf8").readlines()[:500000]
	
	# distribute the processing across the machine cpus
	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	result = pool.map(process_yelp_line, lines)
	result = list(filter(None, result))

	# "unzip" the (tokens, label) tuples to a list of lists of tokens, and a list of labels
	texts, labels = zip(*result)

	# data cleaning
	cleaned_data = normalize(texts,labels)

	print("Number of documents is %s"%len(cleaned_data))
	print("Number of labels is %s"%len(set(labels)))
	print("Average word length of reviews is %s"%(np.mean([len(text.split(' ')) for text in texts])))
	print("Label distribution:\n", cleaned_data.groupby('label').count().reset_index().rename(columns={'review':'# of reviews'}))
