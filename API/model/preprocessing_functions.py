import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from joblib import dump
import os

class Preprocessing():
    def __init__(self, data):
        """
        :param data: A single DataFrame (concatenated seasons) as processed in gor.py
        """
        self.data = data.reset_index(drop=True)
        self.factors = ['ShotDist','TimeoutTeam','Substitution', 'Shooter',
               'Rebounder', 'Blocker','Fouler',
               'ReboundType','ViolationPlayer',
               'FreeThrowShooter','TurnoverPlayer']

        # Generate column names for the flattened 10-event window
        self.fact_cols = [col + str((i // 11) + 1) for i, col in enumerate(self.factors * 10)]
        self.fact_cols.append('class')

    def feats_engineering(self):
        """Vectorized feature engineering using np.select from gor.py"""
        df_new = self.data
        
        # Shot Distance
        cond_shot = [df_new['ShotDist'] <= 10, df_new['ShotDist'] >= 22, pd.notna(df_new['ShotDist'])]
        df_new['ShotDist'] = np.select(cond_shot, ['close', '3pt', 'mid'], default=None)

        # Team specific actions (Home vs Away)
        # Timeout
        cond_to = [pd.notna(df_new['TimeoutTeam']) & (df_new['TimeoutTeam'] == df_new['HomeTeam']),
                   pd.notna(df_new['TimeoutTeam']) & (df_new['TimeoutTeam'] != df_new['HomeTeam'])]
        df_new['TimeoutTeam'] = np.select(cond_to, ['timeout_home', 'timeout_away'], default=None)

        # Generic mapping for player actions
        mappings = {
            'Shooter': 'shooter', 'Rebounder': 'rebounder', 'Blocker': 'blocker',
            'Fouler': 'fouler', 'ViolationPlayer': 'violator', 
            'FreeThrowShooter': 'ft', 'TurnoverPlayer': 'to_player'
        }

        for col, prefix in mappings.items():
            cond = [pd.notna(df_new[col]) & pd.notna(df_new['HomePlay']),
                    pd.notna(df_new[col]) & pd.notna(df_new['AwayPlay'])]
            df_new[col] = np.select(cond, [f'{prefix}_home', f'{prefix}_away'], default=None)

        # Substitution
        cond_sub = [pd.notna(df_new['EnterGame']) & pd.notna(df_new['HomePlay']),
                    pd.notna(df_new['EnterGame']) & pd.notna(df_new['AwayPlay'])]
        df_new['Substitution'] = np.select(cond_sub, ['sub_home', 'sub_away'], default=None)
        
        self.data = df_new

    def get_runs(self, team_side='HomePlay', opposing_side='AwayPlay'):
        """Logic from home_runner/away_runner in gor.py"""
        runs = []
        current_run = []
        for idx in self.data.index:
            made_shot = pd.notna(self.data.at[idx, team_side]) and 'makes' in str(self.data.at[idx, team_side])
            opp_made = pd.notna(self.data.at[idx, opposing_side]) and 'makes' in str(self.data.at[idx, opposing_side])

            if made_shot:
                current_run.append(idx)
            elif opp_made:
                if len(current_run) >= 4:
                    runs.append(current_run.copy())
                current_run.clear()
        
        if len(current_run) >= 4:
            runs.append(current_run.copy())
        return runs

    def runs_iter(self, runs_indices):
        """Flattens 10 events prior to a run into a single row"""
        runs_list = []
        for run in runs_indices:
            # Check if we have 10 preceding events available in index
            if run[0] - 10 in self.data.index:
                window = self.data.loc[run[0]-10 : run[0]-1, self.factors].values.ravel()
                row = np.append(window, 1) # Class 1 for 'Run'
                runs_list.append(row)
        
        return pd.DataFrame(runs_list, columns=self.fact_cols)

    def no_runs_preprocessing(self, runs_indices):
        """Segments non-run data into windows of 10"""
        # Exclude indices involved in runs (and the 10 leading events)
        r_x = []
        for run in runs_indices:
            r_x.extend(range(run[0] - 10, run[0] + 1))
        
        no_runs_data = self.data[~self.data.index.isin(r_x)].reset_index(drop=True)
        
        segment_size = 10
        segments = len(no_runs_data) // segment_size
        
        # Split into blocks of 10
        no_runs_list = []
        for i in range(segments):
            window = no_runs_data.loc[i*10 : (i*10)+9, self.factors].values.ravel()
            row = np.append(window, 0) # Class 0 for 'No Run'
            no_runs_list.append(row)
            
        return pd.DataFrame(no_runs_list, columns=self.fact_cols)

    def encoders(self, df):
        """Encodes categorical data and saves encoders"""
        os.makedirs('model/encoders', exist_ok=True)
        encoded_df = pd.DataFrame()
        
        for i, column in enumerate(df.columns[:-1]):
            le = LabelEncoder()
            # Convert to string to handle potential mixed types/NaNs
            df[column] = df[column].astype(str)
            encoded_df[column] = le.fit_transform(df[column])
            dump(le, f'model/encoders/le_{i}_{column}.joblib')
            
        encoded_df['class'] = df['class'].values
        return encoded_df

    def preprocess(self):
        """Main pipeline combining gor.py logic into the class structure"""
        # 1. Feature Engineering
        self.feats_engineering()
        
        # 2. Identify Runs (Focusing on Home Runs as per gor.py training)
        h_runs_indices = self.get_runs(team_side='HomePlay', opposing_side='AwayPlay')
        
        # 3. Create Positive and Negative Samples
        pos_df = self.runs_iter(h_runs_indices)
        neg_df = self.no_runs_preprocessing(h_runs_indices)
        
        # 4. Combine and Encode
        combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
        encoded_df = self.encoders(combined_df)
        
        # 5. Balanced Undersampling (Logic from gor.py)
        runs_only = encoded_df[encoded_df['class'] == 1]
        no_runs_only = encoded_df[encoded_df['class'] == 0].sample(n=len(runs_only), random_state=43)
        
        final_df = pd.concat([runs_only, no_runs_only]).astype('float32')
        
        # 6. Split Features and Labels
        X = final_df.iloc[:, :-1]
        y = final_df.iloc[:, -1]
        
        return (X, y), final_df