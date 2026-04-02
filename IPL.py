import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

matches = matches[['id', 'team1', 'team2', 'venue', 'toss_decision', 'toss_winner', 'winner', 'date']]
matches = matches[matches['winner'] != 'No Result'].copy()
matches = matches.sort_values('date').reset_index(drop=True)
matches['toss_decision'].fillna('field', inplace=True)
matches['venue'].fillna('Unknown', inplace=True)

# NEW TEAM FORM 
data = pd.merge(deliveries, matches, left_on='match_id', right_on='id', how='inner')
train_match_ids = matches.iloc[:int(len(matches)*0.8)]['id']
team_runs = data.groupby(['match_id', 'batting_team'])['total_runs'].sum().reset_index()
wickets = data.groupby(['match_id', 'batting_team'])['player_dismissed'].count().reset_index()
wickets.rename(columns={'player_dismissed': 'wickets'}, inplace=True)
balls = data.groupby(['match_id', 'batsman'])['ball'].count().reset_index()
runs = data.groupby(['match_id', 'batsman'])['batsman_runs'].sum().reset_index()

batting_stats = pd.merge(runs, balls, on=['match_id', 'batsman'])
batting_stats['strike_rate'] = (batting_stats['batsman_runs'] / batting_stats['ball']) * 100

batting_stats['impact'] = batting_stats['batsman_runs'] * batting_stats['strike_rate']

bowler_runs = data.groupby(['match_id', 'bowler'])['total_runs'].sum().reset_index()
bowler_balls = data.groupby(['match_id', 'bowler'])['ball'].count().reset_index()

bowling_stats = pd.merge(bowler_runs, bowler_balls, on=['match_id', 'bowler'])
bowling_stats['economy'] = (bowling_stats['total_runs'] / bowling_stats['ball']) * 6

# Bowler performance
bowler_stats = data.groupby(['bowler'])['total_runs'].agg(['sum', 'count']).reset_index()

bowler_team = data.groupby('bowler')['bowling_team'].first().reset_index()

bowler_stats = bowler_stats.merge(bowler_team, on='bowler', how='left')

bowler_stats_train = bowler_stats[bowler_stats['bowler'].isin(
    data[data['match_id'].isin(train_match_ids)]['bowler']
)]

top_bowler = bowler_stats_train.copy()

# impact add 
top_bowler['impact'] = top_bowler['count'] / (top_bowler['sum'] + 1)

# best bowler per team (high impact = good)
top_bowler = top_bowler.sort_values(by='impact', ascending=False)\
                       .groupby('bowling_team').first().reset_index()

top_bowler = top_bowler[['bowling_team', 'impact']]

# Batsman performance
batsman_stats = data.groupby(['batsman'])['batsman_runs'].agg(['sum', 'count']).reset_index()
batsman_stats['avg_runs'] = batsman_stats['sum'] / batsman_stats['count']

# Merge batsman with team
batsman_team = data.groupby('batsman')['batting_team'].first().reset_index()

batsman_stats = batsman_stats.merge(batsman_team, on='batsman', how='left')

# Top batsman per team
batsman_stats_train = batsman_stats[batsman_stats['batsman'].isin(
    data[data['match_id'].isin(train_match_ids)]['batsman']
)]

top_batsman = batsman_stats_train.sort_values(by='avg_runs', ascending=False)\
                .groupby('batting_team').first().reset_index()

top_batsman = top_batsman[['batting_team', 'avg_runs']]
top_batsman.rename(columns={'avg_runs': 'top_batsman_avg'}, inplace=True)

team_form = {}

form_list_team1 = []
form_list_team2 = []

def calculate_form(history):
    if len(history) < 3:
        return 0.5
    
    weights = np.linspace(0.2, 1, len(history))
    return np.sum(np.array(history) * weights) / np.sum(weights)
for i, row in matches.iterrows():
    t1 = row['team1']
    t2 = row['team2']
    winner = row['winner']
    
    if t1 not in team_form:
        team_form[t1] = []
    if t2 not in team_form:
        team_form[t2] = []
    
    form_t1 = calculate_form(team_form[t1][-5:])
    form_t2 = calculate_form(team_form[t2][-5:])

    form_list_team1.append(form_t1)
    form_list_team2.append(form_t2)
    
    team_form[t1].append(1 if winner == t1 else 0)
    team_form[t2].append(1 if winner == t2 else 0)

matches['team1_form'] = form_list_team1
matches['team2_form'] = form_list_team2

# H2H Feature
h2h = {}

h2h_team1 = []
h2h_team2 = []

for i, row in matches.iterrows():
    t1 = row['team1']
    t2 = row['team2']
    winner = row['winner']
    
    key = tuple(sorted([t1, t2]))
    
    if key not in h2h:
        h2h[key] = {t1:0, t2:0}
    
    h2h_team1.append(h2h[key].get(t1, 0))
    h2h_team2.append(h2h[key].get(t2, 0))
    
    if winner in h2h[key]:
        h2h[key][winner] += 1

matches['h2h_team1'] = h2h_team1
matches['h2h_team2'] = h2h_team2

# Toss win %
toss_stats = matches.groupby('toss_winner').size().reset_index(name='toss_wins')

total_matches = len(matches)

toss_stats['toss_win_percent'] = toss_stats['toss_wins'] / total_matches

match_features = matches[['id', 'team1', 'team2', 'venue', 'toss_winner', 
                          'toss_decision', 'winner',
                          'team1_form', 'team2_form',
                          'h2h_team1', 'h2h_team2']]

match_features = match_features.copy()  

match_features = match_features.merge( 
                    toss_stats[['toss_winner', 'toss_win_percent']],
                    on='toss_winner', 
                    how='left')

# Fix index alignment
match_features = match_features.reset_index(drop=True)
matches = matches.reset_index(drop=True)

# Venue Advantage Feature

# Venue win percentage 
train_matches = matches.iloc[:int(len(matches)*0.8)]

venue_win = train_matches.groupby(['venue', 'winner']).size().reset_index(name='wins')
venue_total = train_matches.groupby('venue').size().reset_index(name='total')

venue_stats = venue_win.merge(venue_total, on='venue')
venue_stats['win_percent'] = venue_stats['wins'] / venue_stats['total']

# dictionary
venue_dict = {
    (row['venue'], row['winner']): row['win_percent']
    for _, row in venue_stats.iterrows()
}

match_features['venue_team1'] = [
    venue_dict.get((v, t), 0.5)
    for v, t in zip(match_features['venue'], match_features['team1'])
]

match_features['venue_team2'] = [
    venue_dict.get((v, t), 0.5)
    for v, t in zip(match_features['venue'], match_features['team2'])
]

team_stats = pd.merge(team_runs, wickets, on=['match_id', 'batting_team'])
team_stats_train = team_stats[team_stats['match_id'].isin(train_match_ids)]
team_strength = team_stats_train.groupby('batting_team')['total_runs'].mean().reset_index()
team_strength.rename(columns={'total_runs': 'avg_runs'}, inplace=True)

match_features = match_features.merge(team_strength, left_on='team1', right_on='batting_team', how='left')
match_features.drop('batting_team', axis=1, inplace=True)
match_features.rename(columns={'avg_runs': 'team1_strength'}, inplace=True)

# Team1 top batsman
match_features = match_features.merge(top_batsman, left_on='team1', right_on='batting_team', how='left')
match_features.drop('batting_team', axis=1, inplace=True)
match_features.rename(columns={'top_batsman_avg': 'team1_batsman_strength'}, inplace=True)

# Team2 top batsman
match_features = match_features.merge(top_batsman, left_on='team2', right_on='batting_team', how='left')
match_features.drop('batting_team', axis=1, inplace=True)
match_features.rename(columns={'top_batsman_avg': 'team2_batsman_strength'}, inplace=True)
match_features = match_features.merge(team_strength, left_on='team2', right_on='batting_team', how='left')
match_features.drop('batting_team', axis=1, inplace=True)
match_features.rename(columns={'avg_runs': 'team2_strength'}, inplace=True)

# Team1 bowler
match_features = match_features.merge(top_bowler, left_on='team1', right_on='bowling_team', how='left')
match_features.drop('bowling_team', axis=1, inplace=True)
match_features.rename(columns={'impact':'team1_bowler_strength'}, inplace=True)

# Team2 bowler
match_features = match_features.merge(top_bowler, left_on='team2', right_on='bowling_team', how='left')
match_features.drop('bowling_team', axis=1, inplace=True)
match_features.rename(columns={'impact': 'team2_bowler_strength'}, inplace=True)
match_features = match_features.loc[:, ~match_features.columns.duplicated()]
                                    
match_features.fillna(0, inplace=True)

# Toss Advantage
match_features['toss_advantage'] = (
    (match_features['toss_winner'] == match_features['team1']).astype(int)
)

match_features['form_diff'] = match_features['team1_form'] - match_features['team2_form']
match_features['strength_diff'] = match_features['team1_strength'] - match_features['team2_strength']
match_features['h2h_diff'] = match_features['h2h_team1'] - match_features['h2h_team2']
match_features['venue_diff'] = match_features['venue_team1'] - match_features['venue_team2']
match_features['batting_diff'] = match_features['team1_batsman_strength'] - match_features['team2_batsman_strength']
match_features['bowling_diff'] = match_features['team1_bowler_strength'] - match_features['team2_bowler_strength']

match_features.drop([
    'team1_form', 'team2_form',
    'team1_strength', 'team2_strength',
    'h2h_team1', 'h2h_team2',
    'toss_winner','toss_win_percent',
    'toss_decision',   
    'venue'            
], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

encoders = {}

for col in ['team1', 'team2']:
    le = LabelEncoder()
    match_features[col] = le.fit_transform(match_features[col])
    encoders[col] = le

# target 
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(match_features['winner'])

X = match_features.drop(['winner', 'batting_team_x', 'batting_team_y'], axis=1, errors='ignore')
X= X.apply(pd.to_numeric)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

plt.figure(figsize=(10,6))
import xgboost as xgb
xgb.plot_importance(model)
plt.show()
feature_names = X.columns
importance = model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

print(feat_imp)

plt.figure(figsize=(10,6))

plt.barh(feat_imp['Feature'], feat_imp['Importance'])

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance (XGBoost)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

predictions = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))

from collections import Counter

final_prediction = Counter(predictions)
print(final_prediction)

sorted_pred = dict(sorted(final_prediction.items(), key=lambda x: x[1], reverse=True))

labels = target_encoder.inverse_transform(list(sorted_pred.keys()))

plt.figure(figsize=(8,5))
plt.bar(labels, sorted_pred.values(), color='skyblue')

for i, v in enumerate(sorted_pred.values()):
    plt.text(i, v + 0.05, str(v), ha='center')

plt.xticks(rotation=45)
plt.xlabel('Teams')
plt.ylabel('Wins')
plt.title('Predicted Wins Distribution')

plt.tight_layout()
plt.show()






