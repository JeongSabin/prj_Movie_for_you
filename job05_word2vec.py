import pandas as pd
from gensim.models import Word2Vec

review_word = pd.read_csv('./crawling_data/cleaned_review_one.csv')
review_word.info()

cleaned_token_reviews = list(review_word['reviews'])
print(cleaned_token_reviews)

cleaned_tokens = []
for sentence in cleaned_token_reviews:
    token = sentence.split()
    cleaned_tokens.append(token)
print(cleaned_tokens[0])

embedding_model = Word2Vec(cleaned_tokens, vector_size=100,         # 형태소들로 이루어진 리스트, 차원축소 ( 차원의저주 방지 )
                           window=4, min_count=20,                  # window = 문맥을 단어, min_count = 20번 이상 등장하는 단어들만 벡터라이징
                           workers=8, epochs=100, sg=1)             # workers = 코어를 몇개 쓰는지, sg = 알고리즘을 어떤 걸 쓸것인지
embedding_model.save('./models/Word2vec_2017_2020_movies.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key))