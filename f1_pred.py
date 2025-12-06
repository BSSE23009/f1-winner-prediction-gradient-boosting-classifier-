import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

def get_race_results(season, round_number):
    url = f"https://api.jolpi.ca/ergast/f1/{season}/{round_number}/results.json"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return None

    results = races[0]["Results"]
    drivers = []

    for r in results:
        flat = {}
        for k, v in r.items():
            if k not in ["Driver", "Constructor", "FastestLap"]:
                flat[k] = v

        d = r["Driver"]
        for k, v in d.items():
            flat[f"Driver.{k}"] = v

        c = r["Constructor"]
        for k, v in c.items():
            flat[f"Constructor.{k}"] = v

        flat["season"] = season
        flat["round"] = round_number
        flat["raceName"] = races[0]["raceName"]

        drivers.append(flat)

    return pd.DataFrame(drivers)

def fetch_season_results(season):
    url = f"https://api.jolpi.ca/ergast/f1/{season}.json"
    data = requests.get(url).json()
    races = data["MRData"]["RaceTable"]["Races"]

    all_races = []
    for r in races:
        round_num = int(r["round"])
        df_race = get_race_results(season, round_num)
        if df_race is not None:
            all_races.append(df_race)

    if all_races:
        return pd.concat(all_races, ignore_index=True)
    else:
        return pd.DataFrame()

# Fetch past seasons for training
df_season = pd.concat([fetch_season_results(2023), fetch_season_results(2024)], ignore_index=True)

# Manually added Abu Dhabi 2025 grid
data_manual = [
    {"Driver.familyName": "Verstappen", "Constructor.name": "Red Bull", "grid": 2, "number": 1},
    {"Driver.familyName": "Russell", "Constructor.name": "Mercedes", "grid": 4, "number": 63},
    {"Driver.familyName": "Antonelli", "Constructor.name": "Mercedes", "grid": 17, "number": 12},
    {"Driver.familyName": "Leclerc", "Constructor.name": "Ferrari", "grid": 9, "number": 16},
    {"Driver.familyName": "Sainz", "Constructor.name": "Williams", "grid": 3, "number": 55},
    {"Driver.familyName": "Hadjar", "Constructor.name": "RB F1 Team", "grid": 8, "number": 6},
    {"Driver.familyName": "Hülkenberg", "Constructor.name": "Sauber", "grid": 11, "number": 27},
    {"Driver.familyName": "Hamilton", "Constructor.name": "Ferrari", "grid": 19, "number": 44},
    {"Driver.familyName": "Ocon", "Constructor.name": "Haas F1 Team", "grid": 13, "number": 31},
    {"Driver.familyName": "Bearman", "Constructor.name": "Haas F1 Team", "grid": 14, "number": 87},
    {"Driver.familyName": "Alonso", "Constructor.name": "Aston Martin", "grid": 7, "number": 14},
    {"Driver.familyName": "Tsunoda", "Constructor.name": "Red Bull", "grid": 20, "number": 22},
    {"Driver.familyName": "Gasly", "Constructor.name": "Alpine F1 Team", "grid": 10, "number": 10},
    {"Driver.familyName": "Lawson", "Constructor.name": "RB F1 Team", "grid": 6, "number": 30},
    {"Driver.familyName": "Colapinto", "Constructor.name": "Alpine F1 Team", "grid": 15, "number": 43},
    {"Driver.familyName": "Albon", "Constructor.name": "Williams", "grid": 16, "number": 23},
    {"Driver.familyName": "Bortoleto", "Constructor.name": "Sauber", "grid": 18, "number": 5},
    {"Driver.familyName": "Stroll", "Constructor.name": "Aston Martin", "grid": 12, "number": 18},
    {"Driver.familyName": "Norris", "Constructor.name": "McLaren", "grid": 1, "number": 4},
    {"Driver.familyName": "Piastri", "Constructor.name": "McLaren", "grid": 5, "number": 81},
]

df_manual = pd.DataFrame(data_manual)
df_manual["season"] = 2025
df_manual["round"] = 24
df_manual["raceName"] = "Abu Dhabi Grand Prix"

df_all = pd.concat([df_season, df_manual], ignore_index=True)

le_driver = LabelEncoder()
le_constructor = LabelEncoder()
df_all['Driver_enc'] = le_driver.fit_transform(df_all['Driver.familyName'])
df_all['Constructor_enc'] = le_constructor.fit_transform(df_all['Constructor.name'])

X = df_all[['grid', 'Driver_enc', 'Constructor_enc']]
df_all['is_winner'] = (df_all['position'] == '1').astype(int)
y = df_all['is_winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

X_upcoming = df_manual[['grid', 'Driver.familyName', 'Constructor.name']].copy()
X_upcoming['Driver_enc'] = le_driver.transform(X_upcoming['Driver.familyName'])
X_upcoming['Constructor_enc'] = le_constructor.transform(X_upcoming['Constructor.name'])
X_upcoming_features = X_upcoming[['grid', 'Driver_enc', 'Constructor_enc']]

X_upcoming['win_prob'] = model.predict_proba(X_upcoming_features)[:, 1]
X_upcoming.sort_values(by='win_prob', ascending=False, inplace=True)

print("Predicted winning probabilities for Abu Dhabi 2025:")
print(X_upcoming[['Driver.familyName', 'Constructor.name', 'grid', 'win_prob']])
