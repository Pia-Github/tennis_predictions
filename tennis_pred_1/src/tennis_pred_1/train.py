#!/usr/bin/env python
# coding: utf-8

# # ATP Tennis - How long will it take?
# 
# Background: https://www.kaggle.com/datasets/guillemservera/tennis
# 
# ***Question***  
# Can we predict how long an ATP Tennis match is going to take?  
# 
# 
# 
# ## Read data

# In[1]:


import pandas as pd
import numpy as np


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import h2o
from h2o.automl import H2OAutoML
import matplotlib.pyplot as pyplot


# In[2]:


tennis_df = pd.read_csv("ATP_tennis_group_project.csv")
tennis_df = tennis_df.drop(columns=["Unnamed: 0"])


# In[3]:


pd.set_option('display.max_columns', None)
tennis_df.head()


# In[4]:


tennis_df.shape


# ## Data Cleaning

# ***tourney_id*** --> keep, indicates specifics about a day  
# ***tourney_name*** --> keep  
# ***surface*** --> keep  
# ***draw_size***: Number of players in the draw; all participants in the tournament  
# ***tourney_level*** --> keep  
# ***tourney_date***: Maybe split by season? Or month or something  
# ***winner_id***: drop, we can use name instead  
# ***winner_seed*** indicates world ranking (if empty, ranking not high) --> is this before or after match?  
# ***winner_entry*** how did the player get into the tournament? Qulaification tournament, Wildcard, etc. --> What means empty?  
# ***winner_name***: Keep but correct for Edouard Roger Vasselin/Edouard Roger-Vasselin  
# **winner_hand** --> keep  
# **winner_ht**: size of the winner, keep  
# ***winner_ioc***: country of origin  
# ***winner_age***: --> keep  
# ***loser_id***: drop, we can use name instead  
# ***loser_seed***: indicates world ranking (if empty, ranking not high) --> is this before or after match?    
# ***loser_entry***: how did the player get into the tournament? Qulaification tournament, Wildcard, etc. --> What means empty?  
# ***loser_name***: Keep but correct for Edouard Roger Vasselin/Edouard Roger-Vasselin   
# ***loser_hand***: --> keep   
# ***loser_ht***: size of the loser, keep    
# ***loser_ioc***: country of origin  
# ***loser_age***: --> keep  
# ***score*** --> drop, we don't know before  
# ***best_of***: how many Sets CAN there be? --> keep  
# ***round***: Level in the tournament: Final/Semifinal/Quarterfinal/R R depends on the size of the tournament  
# ***minutes: dependend variable!*** 
# ***w_ace 	w_df 	w_svpt 	w_1stIn 	w_1stWon 	w_2ndWon 	w_SvGms 	w_bpSaved 	w_bpFaced 	l_ace 	l_df 	l_svpt 	l_1stIn l_1stWon 	l_2ndWon 	l_SvGms 	l_bpSaved 	l_bpFaced*** --> drop as they are statistics only known after the tournament  
# ***winner_rank***: interesting, but is it rank before or after? Also watch out counfundation with winner_seed     
# ***winner_rank_points***: interesting, but is it rank before or after?  
# ***loser_rank***: interesting, but is it rank before or after? Also watch out counfundation with loser_seed   
# ***loser_rank_points*** interesting, but is it rank before or after?    
# ***year*** --> keep  
# ***has_stats*** --> drop  
# 
# loser and winner must somehow be randomized per Tournament - we cannot keep information who won or who played against whom but it's important who was in the tournament

# ***keep***   
# tourney_id  
# tourney_name  
# surface  
# draw_size  
# tourney_level  
# winner_name: Keep but correct for Edouard Roger Vasselin/Edouard Roger-Vasselin  
# winner_hand  
# winner_ht  
# winner_ioc  
# winner_age  
# loser_name: Keep but correct for Edouard Roger Vasselin/Edouard Roger-Vasselin  
# loser_hand  
# loser_ht 
# loser_ioc  
# loser_age  
# best_of  
# round  
# year  
# tourney_date  
# winner_rank  
# winner_rank_points  
# loser_rank  
# loser_rank_points  
#    
# *minutes*: dependend variable  
# 
# 
# ***drop***  
# winner_id, loser_id --> duplicate of name  
# score, w_ace w_df w_svpt w_1stIn w_1stWon w_2ndWon w_SvGms w_bpSaved w_bpFaced l_ace l_df l_svpt l_1stIn l_1stWon l_2ndWon l_SvGms l_bpSaved l_bpFaced --> data leakage
# winner_seed/loser_seed
# winner_entry
# loser_entry:
# 
# 
# ***discussion***  
# participants: all participant data is labeled as winner/loser --> data leakage.  
# *Solution*: Randomize winner and loser to player_1 and player_2 (and so all connected data)

# In[6]:


tennis_df.head()


# ### drop columns

# In[10]:


tennis_df = pd.read_csv("ATP_tennis_group_project.csv")

cols_to_drop = [
    "Unnamed: 0", "winner_id", "loser_id",
    "score",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", 
    "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", 
    "l_bpFaced",
    "winner_seed", "loser_seed",
    "winner_entry", "loser_entry", "has_stats", "match_num"
]

tennis_df = tennis_df.drop(columns=cols_to_drop)

tennis_df.head()


# ### extract month from tourney date

# In[11]:


tennis_df["month"] = pd.to_datetime(tennis_df["tourney_date"]).dt.month
tennis_df["tourney_month_sin"] = np.sin(2 * np.pi * tennis_df["month"] / 12)
tennis_df["tourney_month_cos"] = np.cos(2 * np.pi * tennis_df["month"] / 12)
tennis_df = tennis_df.drop(columns=["tourney_date", "month"])
tennis_df.head()


# ### fix name duplicate

# In[12]:


tennis_df = tennis_df.replace("Edouard Roger-Vasselin", "Edouard Roger Vasselin")


# ### blind the information of winning or losing

# In[13]:


# random mask
mask = np.random.rand(len(tennis_df)) > 0.5

# all winner/loser column pairs
winner_cols = ['winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
               'winner_rank', 'winner_rank_points']
loser_cols  = ['loser_name',  'loser_hand',  'loser_ht',  'loser_ioc',  'loser_age',
               'loser_rank',  'loser_rank_points']
new_cols_1  = ['name_1', 'hand_1', 'ht_1', 'ioc_1', 'age_1', 'rank_1', 'rank_points_1']
new_cols_2  = ['name_2', 'hand_2', 'ht_2', 'ioc_2', 'age_2', 'rank_2', 'rank_points_2']

for w, l, c1, c2 in zip(winner_cols, loser_cols, new_cols_1, new_cols_2):
    tennis_df[c1] = np.where(mask, tennis_df[w], tennis_df[l])
    tennis_df[c2] = np.where(mask, tennis_df[l], tennis_df[w])

# drop original columns
tennis_df.drop(columns=winner_cols + loser_cols, inplace=True)

long_df = tennis_df.copy()


# In[14]:


long_df.head()


# ### remove duplicates

# In[15]:


long_df = long_df.drop_duplicates()


# ### deal with missing data

# In[16]:


long_df.info()


# In[17]:


#delete rows without minutes
long_df = long_df.dropna(subset=["minutes"])
long_df = long_df[long_df["minutes"] > 0]

#fill empty rank/rankpoints with 9999 or 0
long_df["rank_1"] = long_df["rank_1"].fillna(9999)
long_df["rank_2"] = long_df["rank_2"].fillna(9999)
long_df["rank_points_1"] = long_df["rank_points_1"].fillna(0)
long_df["rank_points_2"] = long_df["rank_points_2"].fillna(0)

# impute age and ht with MICE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)

long_df[["age_1", "ht_1", "age_2", "ht_2"]] = imputer.fit_transform(
    long_df[["age_1", "ht_1", "age_2", "ht_2"]]
)

long_df.head()


# ### data types

# In[19]:


#label encoding
from sklearn.preprocessing import LabelEncoder

cat_cols = ["tourney_id", "tourney_name", "name_1", "name_2", "ioc_1", "ioc_2"]

le = LabelEncoder()
for col in cat_cols:
    long_df[col] = le.fit_transform(long_df[col].astype(str))

#one hot encoding
long_df = pd.get_dummies(long_df, columns=["tourney_level", "surface", "hand_1", "hand_2"])
bool_cols = long_df.select_dtypes(include="bool").columns
long_df[bool_cols] = long_df[bool_cols].astype(int)

# ordinal encoding
round_order = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7,
    "RR": 3,
    "ER": 1,
    "BR": 6
}
long_df["round"] = long_df["round"].map(round_order)


# In[20]:


long_df.head()


# ## Train Test Split

# In[21]:


from sklearn.model_selection import train_test_split
X = long_df.drop(columns=["minutes"])
y = long_df["minutes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


X_train.to_csv("X_train")
X_test.to_csv("X_test")
y_train.to_csv("y_train")
y_test.to_csv("y_test")


# In[23]:


y_train.describe()


# In[24]:


print(X_test.isnull().sum()[X_test.isnull().sum() > 0])
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])
print(y_train.isnull().sum()[y_train.isnull().sum() > 0])
print(y_test.isnull().sum()[y_test.isnull().sum() > 0])


# ## Model 1  
# basic: all variables, no feature engineering

# In[25]:


#import h2o
#from h2o.automl import H2OAutoML

#h2o.init()

#train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
#test  = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

#aml = H2OAutoML(max_runtime_secs=600, seed=42)
#aml.train(y="minutes", training_frame=train)

#print(aml.leaderboard)


# ## Model 2  
# 15 best variables, no feature engineering

# In[26]:


#from sklearn.feature_selection import SelectKBest, f_regression

#selector = SelectKBest(f_regression, k=15)  # Top 15 Features
#X_train_selected = selector.fit_transform(X_train, y_train)

#selected_features = X_train.columns[selector.get_support()].tolist()
#print(selected_features)


# In[27]:


#train_selected = h2o.H2OFrame(pd.concat([X_train[selected_features], y_train], axis=1))
#test_selected = h2o.H2OFrame(pd.concat([X_test[selected_features], y_test], axis=1))

#aml = H2OAutoML(max_runtime_secs=600, seed=42, sort_metric="RMSE")
#aml.train(y="minutes", training_frame=train_selected)

#print(aml.leaderboard)


# ## Model 3  
# feature engineering: differences between players  
# best 15 fatures

# In[28]:


#from sklearn.feature_selection import SelectKBest, f_regression

#selector = SelectKBest(f_regression, k=15)  # Top 15 Features
#X_train_selected = selector.fit_transform(X_train, y_train)

##selected_features = X_train.columns[selector.get_support()].tolist()
#print(selected_features)


# In[29]:


#train_selected = h2o.H2OFrame(pd.concat([X_train[selected_features], y_train], axis=1))
#test_selected = h2o.H2OFrame(pd.concat([X_test[selected_features], y_test], axis=1))

#aml = H2OAutoML(max_runtime_secs=600, seed=42, sort_metric="RMSE")
#aml.train(y="minutes", training_frame=train_selected)

#print(aml.leaderboard)


# ## Model 4 
# feature engineering: none   
# all fatures

# In[30]:


import h2o
from h2o.automl import H2OAutoML

h2o.init()


# In[31]:


train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test  = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

aml = H2OAutoML(max_runtime_secs=600, seed=42)
aml.train(y="minutes", training_frame=train)

print(aml.leaderboard)


# ### First non-Ensemble Model

# In[32]:


for model_id in aml.leaderboard['model_id'].as_data_frame()['model_id']:
    if 'Stacked' not in model_id:
        best_single = h2o.get_model(model_id)
        break

varimp = best_single.varimp(use_pandas=True)
print(varimp.head(20))


# ### Plot important features

# In[33]:


import matplotlib.pyplot as plt

# Top 10 Features
rename_map = {
    'best_of': 'num sets',
    'draw_size': 'total participants',
    'tourney_level_G': 'Grand Slam',
    'tourney_id': 'tourney',
    'year': 'year',
    'rank_points_1': 'rank points 1',
    'round': 'round',
    'rank_points_2': 'rank points 2',
    'name_1': 'player 1',
    'name_2': 'player 2'
}

top10 = varimp.head(10).sort_values('percentage', ascending=True)
top10 = top10.copy()
top10['variable'] = top10['variable'].map(rename_map).fillna(top10['variable'])

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(top10['variable'], top10['percentage'] * 100, color='steelblue')

for bar, val in zip(bars, top10['percentage'] * 100):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=10)

bars = ax.barh(top10['variable'], top10['percentage'] * 100, color='#176FC1')
ax.set_xlabel('Relative Importance (%)')
ax.set_xlim(0, top10['percentage'].max() * 100 * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('feature_importance_top10.png', dpi=150)
plt.show()


# ## Baseline Model

# In[34]:


import numpy as np

# Durchschnittliche Matchdauer nach best_of
baseline = long_df.groupby('best_of')['minutes'].mean()
print(baseline)

# Vorhersage: einfach den Mittelwert pro Kategorie
def predict_duration(best_of):
    return baseline[best_of]

# RMSE auf Testset
y_pred = X_test['best_of'].map(baseline)
rmse_baseline = np.sqrt(((y_test - y_pred) ** 2).mean())
print(f"\nBaseline RMSE: {rmse_baseline:.2f}")


# ## Improvement Trained Model vs. Baseline Model

# In[35]:


# Baseline RMSE
baseline = long_df.groupby('best_of')['minutes'].mean()
y_pred_baseline = X_test['best_of'].map(baseline)
rmse_baseline = np.sqrt(((y_test - y_pred_baseline) ** 2).mean())

# AutoML RMSE (from Leaderboard)
rmse_automl = aml.leaderboard[0, 'rmse']

# comparison
improvement = rmse_baseline - rmse_automl
improvement_pct = (improvement / rmse_baseline) * 100

print(f"Baseline RMSE (nur best_of): {rmse_baseline:.2f}")
print(f"AutoML RMSE:                 {rmse_automl:.2f}")
print(f"Improvement:                {improvement:.2f} minutes ({improvement_pct:.1f}%)")


# ## Upcoming Match

# In[36]:


predict_df = pd.read_csv("predict_df")
predict_df = predict_df.drop(columns=["Unnamed: 0","minutes"])
predict_df


# In[37]:


# convert predict_df to H2O Frame
predict_h2o = h2o.H2OFrame(predict_df)

# prediction
pred = aml.leader.predict(predict_h2o)
print(pred)

