# import libiaries
import pandas as pd
import numpy as np
import json
import pickle
from nltk.corpus import stopwords
nltk.download('stopwords')


if __name__ == "__main__":
	# read svm model and tfidf tranformer
	model_svm = pickle.load(open('yelp_svm.sav', 'rb'))

	with open('tfidf_vec.pkl', 'rb') as f:
		feature_transformer = pickle.load(f)

	review = ['I love this restaurant! It is sooooo good!',
			'The delivery never came. I had to call them to cancel the order.',
			'The place is nice with large space and nice decoration',
			'The food is okay but too pricy',
			'It is easy to park nearby, furnished recently, but too many people in the gym']

	label = [1, 0, 1, 0, 0]

	# tokenize and normalize the documents
	stop_words = set(stopwords.words('english')) 

	# convert to dataframe
	data = pd.DataFrame({'text': review, 'label': label})

	# keep only numbers, letters and space
	data['text'] = data.apply(lambda t: re.sub(r'[^0-9A-Za-z ]', '', str(t['text'])), axis=1)

	# remove stopwords and convert to lower case
	data['text'] = data.apply(lambda r: ' '.join(w.lower() for w in r['text'].split() if w.lower() not in stop_words),axis=1)

	# discard NA reviews
	data = data.dropna()

	# get tfidf features in a sparse matrix
	fe = feature_transformer.transform(data['text'].tolist())

	# predict
	confidence_score = model_svm.decision_function(fe)
	y_pred = model_svm.predict(fe)

	result = {}
	for i in range(len(data)):
		result[str(i) + '__label: ' + str(label[i])] = {'predicted label': int(y_pred[i]),
														'confidence score': confidence_score[i]}
	# save results
	with open('svm_predcition.json', 'w') as f:
		json.dump(result, f)

	print(json.dumps(result, indent=2))
