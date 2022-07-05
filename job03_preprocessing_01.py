import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('./crawling_data/reviews_2021_2.csv')
df.info()
df.dropna()
df.reset_index()
df.info()

#형태소 분리
okt = Okt()

df_stopwords = pd.read_csv('./crawling_data/stopwords.csv')
#list로 만들기
stopwords = list(df_stopwords['stopword'])

# token = okt.pos(df.reviews[0], stem = True)
# print(token)
# exit()

cleaned_sentences = []
#reviews 컬럼에 있는 값들을 하나씩 받아서
for review in df.reviews:
    review = re.sub('[^가-힣 ]', ' ', review) #가부터 힣까지 그리고 띄어쓰기 빼고 모두 다 띄어쓰기로 대체한다 => 한글과 띄어쓰기만 남기기
    token = okt.pos(review, stem = True)
    #morphs() 형태소로 잘라주는 라이브러리
    #pos는 형태소의 품사까지 알려줌, 주어진 텍스트를 형태소 단위로 나누고 나눠진 각 형태소를 품사와 함께 리스트화

    #동사, 명사, 부사만 남기고 모두 버리기
    #df_token의 class가 명사이거나 |(또는)
    df_token = pd.DataFrame(token, columns=['word', 'class'])
    df_token = df_token[(df_token['class'] == 'Noun') |
                        (df_token['class'] == 'Verb') |
                        (df_token['class'] == 'Adjective')]

    #불용어 제거
    #words에는 한글인 명사, 동사, 수식어만 남고 한글자, stopwords제거한 문장으로 남음
    words = []
    for word in df_token.word:
        if len(word) > 1:
            if word not in stopwords:
              words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)

# print(cleaned_sentences)
# exit()

df['cleaned_sentences'] = cleaned_sentences
df = df[['title', 'cleaned_sentences']]
df.dropna(inplace=True)
df.to_csv('./crawling_data/cleaned_review_2021.csv', index = False)
df.info()