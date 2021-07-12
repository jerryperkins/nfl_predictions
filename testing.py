import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


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
print(dallas['over_yes'].value_counts())

over_under_df = df[['spread_favorite', 'over_under_line', 'weather_temperature', 'weather_wind_mph', 'over_yes']]
print(over_under_df)

X = over_under_df.loc[:, over_under_df.columns[(over_under_df.columns != 'over_yes')]]
y = over_under_df['over_yes'].values

print(X)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)

params={'n_estimators': [25,50,75,100,125,150],
            'min_samples_split': [2,3,4,5,6,8,10],
            'max_depth': [4,5,6,7,8]}

# gs = GridSearchCV(RandomForestClassifier(), param_grid = params)
# gs.fit(X_train, y_train)
# print("grid search", gs.best_params_) # grid search {'max_depth': 6, 'min_samples_split': 5, 'n_estimators': 150}
# print(gs.score)

rf_class = RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True, max_depth=5, min_samples_split=5)
rf_class.fit(X_train, y_train)
score = rf_class.score(X_train, y_train)
print("X_train rf_class", score) 
score = rf_class.score(X_test, y_test)
print("X_test rf_class", score) 
predictions = rf_class.predict(X_test) 
print("rf_class X_test predictions", predictions) #0.5064935064935064

features = rf_class.feature_importances_

features_df = pd.DataFrame({'feature':X.columns, 'importance': rf_class.feature_importances_})
features_df.sort_values(by='importance', inplace = True)
print("rf_class feature importances")
print(features_df)

# # One vs Rest

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

ovr = OneVsRestClassifier(LinearSVC(random_state=3, multi_class='ovr'))
ovr.fit(X_train, y_train)
print('ovr training accuracy:', ovr.score(X_train, y_train)) 
print('ovr testing accuracy', ovr.score(X_test, y_test))

# # knn classifier

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=3, stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# params={'n_neighbors': [3,5,7,9,11],
#             'leaf_size': [20,25,30,35,40]}

# gs = GridSearchCV(KNeighborsClassifier(), param_grid = params)
# gs.fit(X_train, y_train)
# print("grid search", gs.best_params_) 
# print(gs.score)

knn_class = KNeighborsClassifier(n_neighbors=9, leaf_size=20)
knn_class.fit(X_train, y_train)
score = knn_class.score(X_train, y_train)
print("X_train knn_class", score) 
score = knn_class.score(X_test, y_test)
print("X_test knn_class", score) #0.5020508613617719




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

