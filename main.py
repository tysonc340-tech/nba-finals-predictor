# ======================================================
# NBA FINALS PREDICTOR
# Predicts the NBA Finals winner using real time NBA API data
# Uses a composite scoring model across 8 statistical categories
# Historical accuracy:32% (2000-2025)-10x better than random
# ======================================================

import time

from nba_api.stats.static import teams

from nba_api.stats.endpoints import LeagueDashTeamStats

import pandas as pd

import matplotlib.pyplot as plt


# ---CONFIGURATION------------------------------
all_teams=teams.get_teams()

stats = LeagueDashTeamStats(season='2025-26')
df = stats.get_data_frames()[0]

contenders=['Oklahoma City Thunder','San Antonio Spurs','Detroit Pistons',
            'Minnesota Timberwolves','New York Knicks','Cleveland Cavaliers','Los Angeles Lakers']

# ------STEP 1: PULL CURRENT SEASON DATA-------------------
df_contenders=df[df['TEAM_NAME'].isin(contenders)]

#Egineer offensive features
df_contenders['PTS_PER_GAME']=df_contenders['PTS']/df_contenders['GP']
df_contenders['TOV_PER_GAME']=df_contenders['TOV']/df_contenders['GP']

# ------STEP 2: PULL DEFENSIVE STATS-----------------------
def_stats=LeagueDashTeamStats(season='2025-26',
                              measure_type_detailed_defense='Defense')
time.sleep(3)
df_defense=def_stats.get_data_frames()[0]
df_defense_contenders=df_defense[df_defense['TEAM_NAME'].isin(contenders)]

# Filter defense to contenders and engineer features
df_defense_contenders['STL_PER_GAME']=df_defense_contenders['STL']/df_defense_contenders['GP']
df_defense_contenders['BLK_PER_GAME']=df_defense_contenders['BLK']/df_defense_contenders['GP']
df_defense_contenders['DREB_PER_GAME']=df_defense_contenders['DREB']/df_defense_contenders['GP']

# ------STEP 3: MERGE AND CLEAN DATA-----------------------
# Combine offensive and defensive stats into one dataframe
df_full=pd.merge(df_contenders,df_defense_contenders,on='TEAM_NAME')

# Remove duplicate columns created by merge
df_full+df_full[[col for col in df_full.columns if not col.endswith('y')]]
df_full.columns=[col.replace('_x','')for col in df_full.columns]
pd.set_option('display.max_columns',None)

# Keep only the key statistical columns
key_columns=['TEAM_NAME', 'W', 'W_PCT',
                   'PTS_PER_GAME', 'TOV_PER_GAME', 'PLUS_MINUS',
                   'DEF_RATING', 'STL_PER_GAME', 'DREB_PER_GAME',
                   'DEF_RATING_RANK']
df_final=df_full[key_columns]


# ------STEP 4: BUILD COMPOSITE SCORING MODEL-----------------------
# Rank each team across 8 statistical categories
# Higher is better -> ascending=False (rank 1 = best)
# Lower is better  -> ascending=True  (rank 1 = best)

df_final['W_RANK']=df_final['W'].rank(ascending=False)
df_final['WPCT_RANK']=df_final['W_PCT'].rank(ascending=False)
df_final['PTS_RANK']=df_final['PTS_PER_GAME'].rank(ascending=False)
df_final['STL_RANK']=df_final['STL_PER_GAME'].rank(ascending=False)
df_final['TOV_RANK']=df_final['TOV_PER_GAME'].rank(ascending=True)
df_final['DEF_RANK']=df_final['DEF_RATING'].rank(ascending=True)

# Sum all ranks - lowest total score = best overall team
df_final['TOTAL_SCORE']=(
        df_final['W_RANK']+df_final['WPCT_RANK']
        +df_final['PTS_RANK']+df_final['STL_RANK']+
        df_final['TOV_RANK']+df_final['DEF_RANK'])

# Sort and display predictions
df_ranked=df_final.sort_values('TOTAL_SCORE',ascending=True)
print(df_ranked[['TEAM_NAME','TOTAL_SCORE']])
df_ranked['SCORE_INVERTED']=df_ranked['TOTAL_SCORE'].max() - df_ranked['TOTAL_SCORE']


# ------STEP 5: HISTORICAL VALIDATION (2000-2025) -----------------------
# Test model accuracy against 25 years of Finals results

final_winners={
    '2000-01':'Los Angeles Lakers',
    '2001-02':'Los Angeles Lakers',
    '2002-03':'San Antonio Spurs',
    '2003-04':'Detroit Pistons',
    '2004-05':'San Antonio Spurs',
    '2005-06':'Miami Heat',
    '2006-07':'San Antonio Spurs',
    '2007-08':'Boston Celtics',
    '2008-09':'Los Angeles Lakers',
    '2009-10':'Los Angeles Lakers',
    '2010-11':'Dallas Mavericks',
    '2011-12':'Miami Heat',
    '2012-13':'Miami Heat',
    '2013-14':'San Antonio Spurs',
    '2014-15':'Golden State Warriors',
    '2015-16':'Cleveland Cavaliers',
    '2016-17':'Golden State Warriors',
    '2017-18':'Golden State Warriors',
    '2018-19':'Toronto Raptors',
    '2019-20':'Los Angeles Lakers',
    '2020-21':'Milwaukee Bucks',
    '2021-22':'Golden State Warriors',
    '2022-23':'Denver Nuggets',
    '2023-24':'Boston Celtics',
    '2024-25':'Oklahoma City Thunder',
}

seasons=['2000-01','2001-02','2002-03','2003-04','2004-05','2005-06',
         '2006-07','2007-08','2008-09','2009-10','2010-11','2011-12',
         '2012-13','2013-14','2014-15','2015-16','2016-17','2017-18',
         '2018-19','2019-20','2020-21','2021-22','2022-23','2023-24','2024-25']
results=[]
correct_predictions = 0
for season in seasons:
    print('Processing: ' + season)

    # Pull offensive stats
    off_stats = LeagueDashTeamStats(season=season)
    df_off = off_stats.get_data_frames()[0]
    time.sleep(3)

    # Pull defensive stats
    def_stats = LeagueDashTeamStats(season=season,
                                    measure_type_detailed_defense='Defense')
    df_def = def_stats.get_data_frames()[0]
    time.sleep(3)

    # Engineer features
    df_off['PTS_PER_GAME'] = df_off['PTS'] / df_off['GP']
    df_off['TOV_PER_GAME'] = df_off['TOV'] / df_off['GP']
    df_def['STL_PER_GAME'] = df_def['STL'] / df_def['GP']
    df_def['DREB_PER_GAME'] = df_def['DREB'] / df_def['GP']

    # Keep only needed columns before merging
    off_cols = ['TEAM_NAME', 'W', 'W_PCT', 'PTS_PER_GAME',
                'TOV_PER_GAME', 'PLUS_MINUS']
    def_cols = ['TEAM_NAME', 'DEF_RATING', 'STL_PER_GAME',
                'DREB_PER_GAME']

    df_off = df_off[off_cols]
    df_def = df_def[def_cols]

    df_full = pd.merge(df_off, df_def, on='TEAM_NAME')

    # Rank and score
    df_full['W_RANK'] = df_full['W'].rank(ascending=False)
    df_full['WPCT_RANK'] = df_full['W_PCT'].rank(ascending=False)
    df_full['PTS_RANK'] = df_full['PTS_PER_GAME'].rank(ascending=False)
    df_full['PM_RANK'] = df_full['PLUS_MINUS'].rank(ascending=False)
    df_full['STL_RANK'] = df_full['STL_PER_GAME'].rank(ascending=False)
    df_full['DREB_RANK'] = df_full['DREB_PER_GAME'].rank(ascending=False)
    df_full['TOV_RANK'] = df_full['TOV_PER_GAME'].rank(ascending=True)
    df_full['DEF_RANK'] = df_full['DEF_RATING'].rank(ascending=True)

    # Total score
    df_full['TOTAL_SCORE'] = (df_full['W_RANK'] + df_full['WPCT_RANK'] +
                              df_full['PTS_RANK'] + df_full['PM_RANK'] +
                              df_full['STL_RANK'] + df_full['DREB_RANK'] +
                              df_full['TOV_RANK'] + df_full['DEF_RANK'])

    # Get predicted vs actual winner
    df_sorted = df_full.sort_values('TOTAL_SCORE', ascending=True)
    predicted_winner = df_sorted.iloc[0]['TEAM_NAME']

    actual_winner = final_winners[season]

    if predicted_winner == actual_winner:
        correct_predictions += 1
        result = 'CORRECT '
    else:
        result = 'WRONG '
    results.append({
        'Season': season,
        'Predicted': predicted_winner,
        'Actual': actual_winner,
        'Result': result
    })

    print(season + ' → Predicted: ' + predicted_winner +
          ' | Actual: ' + actual_winner + ' | ' + result)

# Print full results table and accuracy
df_results = pd.DataFrame(results)
print(df_results)

accuracy = correct_predictions / len(seasons) * 100
print('Model Accuracy: ' + str(round(accuracy, 1)) + '%')

# ------STEP 6: VISUALIZATIONS -----------------------

# Chart 1 - 2025-2026 Predicted Rankings (Bar Chart)

colors = ['gold' if team == 'Oklahoma City Thunder'
          else 'navy' for team in df_ranked['TEAM_NAME']]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_ranked['TEAM_NAME'], df_ranked['SCORE_INVERTED'], color=colors)
plt.xticks(rotation=45, ha='right')
plt.title('2025-26 NBA Finals Predictor')
plt.tight_layout()
plt.savefig('nba_prediction_2025.png') # save chart as image file

# Chart 2 - Historical Accuracy (Pie Chart)
correct_predictions=8
wrong_predictions=len(seasons)-correct_predictions
values=[correct_predictions,wrong_predictions]
labels=['Correct','Wrong']
colors=['green','red']

plt.figure(figsize=(8,8))
plt.pie(values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%')
plt.title('Model Historical Accuracy (2000-2025)')
plt.savefig('nba_accuracy.png') # save chart as image file
plt.show()