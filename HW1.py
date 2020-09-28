# import libiaries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
import spacy
import os
import re
import time
import multiprocessing
import glob

##################################### Functions #####################################

# nltk word tokenization
def nltk_word_tokenization(data):
    tokenized_word = []
    for d in data:
        words = word_tokenize(d)
        for word in words:
            tokenized_word.append(word)
    return tokenized_word

# nltk sentence tokenization
def nltk_sentence_tokenization(data):
    tokenized_sent = []
    for d in data:
        sents = sent_tokenize(d)
        for sent in sents:
            tokenized_sent.append(sent)
    return tokenized_sent

# nltk Stemming
def nltk_stemming(tokenized_word):
    stemmed_word = []
    ps = PorterStemmer()
    for w in tokenized_word:
        stemmed_word.append(ps.stem(w))
    return stemmed_word

# nltk POS tagging 
def nltk_pos_tagging(tokenized_word):
    pos_tags=nltk.pos_tag(tokenized_word)
    return pos_tags

# spacy sentence tokenization
def spacy_sentence_tokenization(data):
    nlp = spacy.load('en_core_web_sm', disable=["parser", "tagger", "ner"])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    tokenized_sent = []
    for d in data:
        doc = nlp(d)
        sents = [sent.string.strip() for sent in doc.sents]
        for sent in sents:
            tokenized_sent.append(sent)
    return tokenized_sent    
            
# spacy word tokenization
def spacy_word_tokenization(data):
    nlp = spacy.load('en_core_web_sm', disable=["parser", "tagger", "ner"])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    tokenized_word = []
    for d in data:
        doc = nlp(d)
        words = [token.text for token in doc]
        for word in words:
            tokenized_word.append(word)
    return tokenized_word

# spacy pos tagging
def spacy_pos_tagging(data):
    nlp = spacy.load('en_core_web_sm',disable=["parser", "ner"])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    for d in data:
        doc = nlp(d)          
        pos_tags = [[token.text,token.pos_]for token in doc]
    return pos_tags


# nltk parallelization
def nltk_parallelization(data):
    word = nltk_word_tokenization(data)
    sent = nltk_sentence_tokenization(data)
    tag = nltk_pos_tagging(word)
    return (word,sent,tag)

# spacy parallelization
def spacy_parallelization(data):
    nlp = spacy.load('en_core_web_sm',disable=["parser", "ner"])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    tokenized_word = []
    tokenized_sent = []
    pos_tag = []
    for doc in nlp.pipe(data, n_threads = -1):       
        pos_tags = [[token.text,token.pos_]for token in doc]
        sents = [sent.string.strip() for sent in doc.sents]
        words = [token.text for token in doc]
        for word in words:
            tokenized_word.append(word)
        for sent in sents:
            tokenized_sent.append(sent)
        for tag in pos_tags:
            pos_tag.append(tag)           
    return (tokenized_word,tokenized_sent,pos_tag)

############################################# Main ##########################################
if __name__ == '__main__':

	#####################################################################################
    ##################################### Problem 1 #####################################
    #####################################################################################

	# get path
	path = os.getcwd() + '/20_newsgroups/sci.space/*'
	files = glob.glob(path)

	# read txt files into a nested list
	sci_space = []
	for i in files:
	    with open(i, 'r', encoding="utf8", errors="ignore") as f:
	        data = f.read()
	        sci_space.append(data)

	######################### Using `nltk` w/o  parallelization #########################
	# running time for applyung tokenization, stemming and POS tagging using nltk
	start = time.time()
	result_token_word = nltk_word_tokenization(sci_space)
	print ("Word tokenization using nltk takes: %s"%(time.time()-start))

	start = time.time()
	result_tokwn_sent = nltk_sentence_tokenization(sci_space)
	print ("Sentence tokenization using nltk takes: %s"%(time.time()-start))

	start = time.time()
	result_stem = nltk_stemming(result_token_word)
	print ("Word stemming using nltk takes: %s"%(time.time()-start))

	start = time.time()
	result_pos_tag = nltk_pos_tagging(result_token_word)
	print ("POS tagging using nltk takes: %s"%(time.time()-start))

	######################## Using `spacy` w/o  parallelization #########################
	# running time for applyung tokenization, stemming and POS tagging using spacy
	start = time.time()
	result_token_word = spacy_word_tokenization(sci_space)
	print ("Word tokenization using spacy takes: %s"%(time.time()-start))

	start = time.time()
	result_tokwn_sent = spacy_sentence_tokenization(sci_space)
	print ("Sentence tokenization using spacy takes: %s"%(time.time()-start))

	start = time.time()
	result_pos_tag = spacy_pos_tagging(sci_space)
	print ("POS tagging using spacy takes: %s"%(time.time()-start))

	################ Compare parallelization between `nltk` and `spacy` #################
	# nltk
	count = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(count)
	start = time.time()
	result = pool.map(nltk_parallelization, sci_space)
	print ("Parallelization using nltk takes: %s"%(time.time()-start))

	# spacy
	start = time.time()
    result = spacy_parallelization(sci_space)
    print ("Spacy using threadding takes: %s"%(time.time()-start))

	count = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(count)
	start = time.time()
	result = pool.map(spacy_parallelization, sci_space)
	print ("Parallelization using spacy takes: %s"%(time.time()-start))

	#####################################################################################
	##################################### Problem 2 #####################################
	#####################################################################################

	# emails
	tokens = nltk_sentence_tokenization(sci_space)
	emails = []
	for sentence in tokens:
	    email = re.findall(r'[\w\.-]+@[\w-]+\.[\w\.-]+', sentence)
	    for e in email:
	        emails.append(e)
	print(emails)


	'''
	Dates pattern: 
	    1. May 12, 1996 
	    2. 05/12/1996
	    3. 1996/5/12
	    4. 1996May12
	    5. 12 May 1996
	'''

	dates = []
	for sentence in tokens:
	    date = re.findall(r'[JFMAJSOND][a-z]\w+[\s][\d]{1,2}[,][\s][\d]{2,4}|[\d]{1,2}/[\d]{1,2}/[\d]{2,4}|[\d]{1,2}/[\d]{1,2}/[\d]{2,4}|[\d]{4}[JFMASOND][a-z]\w+[\d]{2}|[\d]{1,2}[\s][JFMASOND][a-z]\w+[\s][\d]{2,4}',sentence)
	    for d in date:
	        dates.append(d)
	print(dates)
