# import libiaries
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import pandas as pd
import gensim
from gensim.models import Word2Vec

# Functions
def preprocess(df):
    '''tokenization, normalization (e.g. convert to lowercase, remove non-alphanumeric chars, numbers, etc.）'''
    text_norm = []
    text_norm_string = []
    
    for i in range(0,len(df)): 
        row = df.iloc[[i]]
        strings = row.to_string(header = False)       
        
        # tokenize 
        token = word_tokenize(strings)
        
        # only keep alphabet and get lower case
        alpha = [word.lower() for word in token if word.isalpha()]
        
        # nomolized taxt list 
        text_norm.append(alpha)
        text_norm_string.append(" ".join(alpha))
    
    # Return to text file
    with open('yelp_text_norm.txt', 'w') as f:
        for l in text_norm_string:
            f.write("%s\n" % l)
    
    return(text_norm)

# find top k closest neighbors of target words under given models 
def find_neighbor(model, words, k):
    dic = {}
    for word in words:
        top5 = model.wv.most_similar(word)[:k]
        dic[word] = top5
    
    model_df = pd.DataFrame(dic)
    
    return model_df  


if __name__ == "__main__":
    # Read Yelp review data
    review = pd.read_json('yelp_academic_dataset_review.json',lines=True)

    # Preprocess: data is too large, choose only 100k records of data
    nomalized_text = preprocess(review.head(100000))

    #word2vec models with varying parameters
    model_cbow1 = gensim.models.Word2Vec(nomalized_text, window=5, workers=5, sg=0)
    model_cbow2 = gensim.models.Word2Vec(nomalized_text, window=50, workers=5, sg=0)
    model_sg = gensim.models.Word2Vec(nomalized_text, window=50, workers=5, sg=1)

    # find top k=5 closest neighbors of 10 target words under given models 
    words_10 = ["expensive", "summer", "hotpot", "husband","hotel", "salad", "chicago", "coffee", "gay","wonderful"]

    model1_similar = find_neighbor(model_cbow1, words_10, k=5)
    model2_similar = find_neighbor(model_cbow2, words_10, k=5)
    model3_similar = find_neighbor(model_sg, words_10, k=5)  

    model1_similar.to_csv("model1_top5.csv")
    model2_similar.to_csv("model2_top5.csv")
    model3_similar.to_csv("model3_top5.csv")
