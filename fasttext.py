# import libiaries
import pandas as pd
import numpy as 
from sklearn.model_selection import train_test_split
import fasttext

if __name__ == "__main__":
	# read data
	data = pd.read_csv('review_cleaned.csv').drop('Unnamed: 0', axis=1)

	# modeling
	# fasttext requires data to be in the format of: __label__1 text
	data['new_label'] = data.apply(lambda t: '__label__' + str(t['binary_label']) + ' ' + str(t['text']), axis=1)

	# train test split
	X_train, X_test, y_train, y_test = train_test_split(data['new_label'], data['binary_label'], test_size=0.3, random_state=66)

	# save train and test data
	X_train.to_csv('fasttext_train.txt',index=False, header=False)
	X_test.to_csv('fasttext_test.txt',index=False, header=False)

	# fasttext model - default
	ft_model = fasttext.train_supervised('fasttext_train.txt')

	# calculate evaluation metrics
	result = ft_model.test('fasttext_test.txt')
	precision = result[1]
	recall = result[2]
	print("Precision: %0.4f"%precision)
	print("Recall: %0.4f"%recall)
	print("F1 score: %0.4f"%(2*precision*recall/(precision+recall)))

	# fasttext model - setting 1
	ft_model = fasttext.train_supervised('fasttext_train.txt',wordNgrams=2)
	result = ft_model.test('fasttext_test.txt')
	precision = result[1]
	recall = result[2]
	print("Precision: %0.4f"%precision)
	print("Recall: %0.4f"%recall)
	print("F1 score: %0.4f"%(2*precision*recall/(precision+recall)))

	# fasttext model - setting 2
	ft_model = fasttext.train_supervised('fasttext_train.txt',lr=0.8, wordNgrams=2, loss='softmax')
	result = ft_model.test('fasttext_test.txt')
	precision = result[1]
	recall = result[2]
	print("Precision: %0.4f"%precision)
	print("Recall: %0.4f"%recall)
	print("F1 score: %0.4f"%(2*precision*recall/(precision+recall)))

	# fasttext model - setting 3
	ft_model = fasttext.train_supervised('fasttext_train.txt',lr=0.05, wordNgrams=2, loss='softmax')
	result = ft_model.test('fasttext_test.txt')
	precision = result[1]
	recall = result[2]
	print("Precision: %0.4f"%precision)
	print("Recall: %0.4f"%recall)
	print("F1 score: %0.4f"%(2*precision*recall/(precision+recall)))