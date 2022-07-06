import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from gensim.models import Word2Vec

def getRecommendation(cosin_sim):
    simScore = list(enumerate(cosin_sim[-1]))
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True)
    simScore = simScore[:11]
    movieIdx = [i[0] for i in simScore]
    recMovieList = df_reviews.iloc[movieIdx, 0]
    return recMovieList


df_reviews = pd.read_csv('crawling_data/reviews_2017_2022.csv')
Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
with open('models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

# 영화 제목 / index 를 이용
# movie_idx = df_reviews[df_reviews['titles'] == '모가디슈 (Escape from Mogadishu)'].index[0]
# cosine_sim = linear_kernel(Tfidf_matrix[movie_idx], Tfidf_matrix)
# recommendation = getRecommendation(cosine_sim)
# print(recommendation)

# keyword 이용
embedding_model = Word2Vec.load('./models/Word2vec_2017_2022_movies.model')
keyword = '살인'
sim_word = embedding_model.wv.most_similar(keyword, topn=10)
words = [keyword]
for word, _ in sim_word:
    words.append(word)

# 문장 이용
# sentence = []
# cnt = 10
# for word in words:
#     sentence = sentence + [word] * cnt
#     cnt -= 1
# sentence = ' '.join(sentence)
# print(sentence)
# sentence_vec = Tfidf.transform([sentence]) # 리스트로 줄 것
# cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
# recommendation = getRecommendation(cosine_sim)
# print(recommendation)
#문장 이용
sentence = '''아침에 눈뜨자마자 시풀시풀 거리다가 발등 찍고, 집에 들어오자마자 시풀시풀 거리다가 무릎 찍는 기정을 보고 엄마는 딴 게 팔자가 아니라고, 심뽀가 팔자라고, 심뽀 좀 곱게 쓰라고. 나이 들면 세련되고 발칙하게 '섹스앤더시티'를 찍으며 살 줄 알았는데, 매일 길바닥에 서너 시간씩 버려가면서 출퇴근하느라고 서울 것들보다 빠르게 늙어 간다. 밤이면 발바닥은 찢어질 것 같고, 어깨엔 누가 올라타 앉은 것 같고. 지하철 차창에 비친 얼굴을 보면 저 여자는 누군가 싶고. 나, 이렇게 저무는 건가. 그 전에. 마지막으로. 아무나. 사랑해보겠습니다. 아무나, 한 번만, 뜨겁게, 사랑해보겠습니다.

그동안 인생에 오점을 남기지 않기 위해, 처음부터 마지막 종착지가 될 남자를 찾느라, 간보고 짱보고... 그래서 지나온 인생은 아무것도 없이 그저 지겨운 시간들뿐이었습니다. 이제, 막판이니, 아무나, 정말 아무나, 사랑해보겠습니다. 들이대 보겠습니다. 태훈의 연인. 경선의 고등학교 동창.'''
sentence_vec = Tfidf.transform([sentence])
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
print(recommendation)