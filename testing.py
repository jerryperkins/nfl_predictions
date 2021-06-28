import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# General update on 6/27/21: Turns out my first attempt at cleaning the data was terrible and I spent 10hrs this week doing a more thorough and effective job at it. We now have almost 40% more useable data :)

df = pd.read_csv('data/nfl_data2.csv') # I did most of my data cleaning in SQL and then loaded the csv
print(df.head())
print(df.info())

df = df.drop(columns=['weather_humidity', 'stadium', 'stadium_neutral', 'schedule_season'])
df = df.dropna()

df = df[df['over_under_line'] != " "]

df['over_under_line'] = pd.to_numeric(df['over_under_line'])
print(df.info())
# temp = df['over_under_line']==" "
# print(df.loc[temp])


conditions = [
    (df['score_home'] + df['score_away'] < df['over_under_line']),
    (df['score_home'] + df['score_away'] > df['over_under_line']),
    (df['score_home'] + df['score_away'] == df['over_under_line'])
    ]

values = [0,1,2]

df['over_yes'] = np.select(conditions, values) # adding a column that tells me if the game was over under or a push. This will be something I will try and predict

print(df.info())
print(df['over_yes'])
print(df['over_yes'].value_counts())
print(df[['team_home', 'score_home', 'score_away', 'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line','over_yes']])



# # conditions = [
# #     ((df['team_home'] == df['team_favorite_id']) & (df['score_home'] - df['score_away'] > np.abs(df['spread_favorite'])))
# #     ]
# # values = [1]

conditions = [
    ((df['team_home'] == df['team_favorite_id']) & (df['score_home'] - df['score_away'] > np.abs(df['spread_favorite']))),
    ((df['team_home'] == df['team_favorite_id']) & (df['score_home'] - df['score_away'] < np.abs(df['spread_favorite']))),
    ((df['team_home'] == df['team_favorite_id']) & (df['score_home'] - df['score_away'] == np.abs(df['spread_favorite']))),
    ((df['team_away'] == df['team_favorite_id']) & (df['score_away'] - df['score_home'] > np.abs(df['spread_favorite']))),
    ((df['team_away'] == df['team_favorite_id']) & (df['score_away'] - df['score_home'] < np.abs(df['spread_favorite']))),
    ((df['team_away'] == df['team_favorite_id']) & (df['score_away'] - df['score_home'] == np.abs(df['spread_favorite']))),
    (df['team_favorite_id'] == 'PICK')
    ]
values = [1,0,2,1,0,2,3]

df['fave_wins'] = np.select(conditions,values) # adding a column that tells me if the favorite beat the spread, lost to the spread or pushed. I will also be predicting this
print(df.info())
print(df[['team_home', 'score_home', 'score_away', 'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line','fave_wins']])

print(df.head())
print(df['team_home'].value_counts())
temp = df['team_home']=='DAL'
temp2 = df['team_away']=='DAL'
dallas = df.loc[temp | temp2]
print(dallas['over_yes'].value_counts())

# I may also add another column that will identify how many road games a team has played in X amount of days. The data has no nulls, the data types all make sense and I should be ready to start working on the data next week. I will likely drop more columns but want to make sure I don't need them to calculate other columns first.

# this is where I wrote out my logic to create the fave_wins columns
# score_home = 26
# score_away = 7
# team_favorite_id = 'CHI'
# team_home = 'CHI'
# team_away = 'MIN'
# spread_favorite = -6
# if team_home == team_favorite_id and score_home - score_away > np.abs(spread_favorite):
#     print("fave win")
# elif team_home == team_favorite_id and score_home - score_away < np.abs(spread_favorite):
#     print('dog win')
# elif team_home == team_favorite_id and score_home - score_away == np.abs(spread_favorite):
#     print('push')
# elif team_away == team_favorite_id and score_away - score_home > np.abs(spread_favorite):
#     print("fave win")
# elif team_away == team_favorite_id and score_away - score_home < np.abs(spread_favorite):
#     print('dog win')
# else:
#     print('push')

# schedule_home = df.groupby(df['team_home'])[['score_home']].sum()
# print(schedule_home)

# schedule_away = df.groupby(df['team_away'])[['score_away']].sum()
# print(schedule_away)

