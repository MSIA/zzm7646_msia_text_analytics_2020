# import libiaries
import numpy
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == "__main__":
	# read data
	data = pd.read_csv('review_cleaned.csv').drop('Unnamed: 0', axis=1)

	# unigram:  fit tfidf
	tfidf = TfidfVectorizer(sublinear_tf=True, min_df=500, norm='l2',
		max_features=500, encoding='UTF-8', ngram_range=(1,1), stop_words='english')

	# get tfidf features in a sparse matrix
	fe1 = tfidf.fit_transform(data['text'].tolist())

	# turn into a dataframe of features
	fe_df1 = pd.DataFrame.sparse.from_spmatrix(fe1)

	# 1gram+2gram
	tfidf2 = TfidfVectorizer(sublinear_tf=True, min_df=500, norm='l2',
		max_features=500, encoding='UTF-8', ngram_range=(1,2), stop_words='english')
	fe2 = tfidf2.fit_transform(data['text'].tolist())
	fe_df2 = pd.DataFrame.sparse.from_spmatrix(fe2)

	# train test split
	X_train1, X_test1, y_train1, y_test1 = train_test_split(fe_df1, data['binary_label'], test_size=0.3, random_state=66)
	X_train2, X_test2, y_train2, y_test2 = train_test_split(fe_df2, data['binary_label'], test_size=0.3, random_state=66)
	
	# unigram - default
	svm_unigram = LinearSVC(random_state=66) # fit svm
	svm_unigram.fit(X_train1, y_train1)
	y_pred_unigram = svm_unigram.predict(X_test1) # predict

	# evaluation metrics
	print("Accuracy: %0.4f"%accuracy_score(y_test1, y_pred_unigram))
	print("Precision: %0.4f"%precision_score(y_test1, y_pred_unigram))
	print("Recall: %0.4f"%recall_score(y_test1, y_pred_unigram))
	print("F1 score: %0.4f"%f1_score(y_test1, y_pred_unigram))
	print("Micro-averaged F1 score: %0.4f"%f1_score(y_test1, y_pred_unigram, average='micro'))

	# unigram - penalty
	svm_unigram2 = LinearSVC(random_state=66, loss='hinge', C=10)
	svm_unigram2.fit(X_train1, y_train1)
	y_pred_unigram2 = svm_unigram2.predict(X_test1)

	print("Accuracy: %0.4f"%accuracy_score(y_test1, y_pred_unigram2))
	print("Precision: %0.4f"%precision_score(y_test1, y_pred_unigram2))
	print("Recall: %0.4f"%recall_score(y_test1, y_pred_unigram2))
	print("F1 score: %0.4f"%f1_score(y_test1, y_pred_unigram2))
	print("Micro-averaged F1 score: %0.4f"%f1_score(y_test1, y_pred_unigram2, average='micro'))

	# 1gram+2gram - default 
	svm_2gram1 = LinearSVC(random_state=66)
	svm_2gram1.fit(X_train2, y_train2)
	y_pred_2gram1 = svm_2gram1.predict(X_test2)

	print("Accuracy: %0.4f"%accuracy_score(y_test2, y_pred_2gram1))
	print("Precision: %0.4f"%precision_score(y_test2, y_pred_2gram1))
	print("Recall: %0.4f"%recall_score(y_test2, y_pred_2gram1))
	print("F1 score: %0.4f"%f1_score(y_test2, y_pred_2gram1))
	print("Micro-averaged F1 score: %0.4f"%f1_score(y_test2, y_pred_2gram1, average='micro'))

	# 1gram+2gram - regularization and stopping criterion
	svm_2gram2 = LinearSVC(random_state=66, loss='hinge', C=10)
	svm_2gram2.fit(X_train2, y_train2)
	y_pred_2gram2 = svm_2gram2.predict(X_test2)

	print("Accuracy: %0.4f"%accuracy_score(y_test2, y_pred_2gram2))
	print("Precision: %0.4f"%precision_score(y_test2, y_pred_2gram2))
	print("Recall: %0.4f"%recall_score(y_test2, y_pred_2gram2))
	print("F1 score: %0.4f"%f1_score(y_test2, y_pred_2gram2))
	print("Micro-averaged F1 score: %0.4f"%f1_score(y_test2, y_pred_2gram2, average='micro'))

	# save best performing svm model
	with open('tfidf_vec.pkl', 'wb') as f:
	    pickle.dump(tfidf, f)

	pickle.dump(svm_unigram2, open('yelp_svm.sav', 'wb'))
