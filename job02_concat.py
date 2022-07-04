import pandas as pd

df = pd.read_csv('./crawling_data/reviews_2021_1_page.csv')
df.info()

for i in range(1, 18):
    df_temp = pd.read_csv(f'./crawling_data/reviews_2021_{i}_page.csv')
    df_temp.dropna(inplace=True)
    df_temp.drop_duplicates(inplace=True)
    df = pd.concat([df, df_temp], ignore_index=True)
df.drop_duplicates(inplace=True)
df.info()
my_year = 2021
df.to_csv('./crawling_data/reviews_{}.csv'.format(my_year))