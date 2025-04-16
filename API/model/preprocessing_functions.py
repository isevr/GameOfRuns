# Necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from joblib import dump
import os

class Preprocessing():
    def __init__(self, data):
        self.data = data

        # Feature selection
        self.factors = ['ShotDist','TimeoutTeam','Substitution', 'Shooter',
               'Rebounder', 'Blocker','Fouler',
               'ReboundType','ViolationPlayer',
               'FreeThrowShooter','TurnoverPlayer']

        self.fact_cols = [col + str((i // 11) % 10 + 1) for i, col in enumerate(self.factors * 10)]
        self.fact_cols.append('class')

    # Label encoding
    def encoders(self):

        os.makedirs('model/encoders', exist_ok=True)
        
        self.final_df = pd.concat([self.runs_df,self.no_runs_df],ignore_index=True).astype('str')
        self.encoded_df = pd.DataFrame()
        # encoders = []

        for column, i  in zip(self.final_df.columns[:-1], range(len(self.final_df.columns[:-1]))):
            # print(self.final_df[column].apply(type).value_counts())

            le = LabelEncoder()
            le.fit(self.final_df[column])
            dump(le,'model/encoders/le_'+str(i)+'_'+column)
            # encoders.append(le)
            self.encoded_df[column] = le.transform(self.final_df[column])

        # self.encoded_df = pd.concat([self.encoded_df,self.final_df.iloc[:,-1]],axis=1)
        return self.encoded_df
    
    def feats_engineering(self):
        self.data['ShotDist'] = self.data.ShotDist.apply(lambda x: 'close' if x <= 10 else '3pt' if x >= 22 else 'mid' if pd.notna(x) else x)
    
        self.data['TimeoutTeam'] = self.data.apply(
            lambda row: 'timeout_home' if pd.notna(row['TimeoutTeam']) and row['TimeoutTeam'] == row['HomeTeam'] 
            else 'timeout_away' if pd.notna(row['TimeoutTeam']) 
            else row['TimeoutTeam'], 
            axis=1
        )

        self.data['Shooter'] = self.data.apply(lambda row: 'shooter_home' if pd.notna(row['Shooter']) and pd.notna(row['HomePlay'])
                                            else 'shooter_away' if pd.notna(row['Shooter']) and pd.notna(row['AwayPlay'])
                                            else np.nan,
                                            axis=1)

        self.data['Rebounder'] = self.data.apply(lambda row: 'rebounder_home' if pd.notna(row['Rebounder']) and pd.notna(row['HomePlay'])
                                            else 'rebounder_away' if pd.notna(row['Rebounder']) and pd.notna(row['AwayPlay'])
                                            else np.nan,
                                            axis=1)

        self.data['Blocker'] = self.data.apply(lambda row: 'blocker_home' if pd.notna(row['Blocker']) and pd.notna(row['HomePlay'])
                                            else 'blocker_away' if pd.notna(row['Blocker']) and pd.notna(row['AwayPlay'])
                                            else np.nan,
                                            axis=1)

        self.data['Fouler'] = self.data.apply(lambda row: 'fouler_home' if pd.notna(row['Fouler']) and pd.notna(row['HomePlay'])
                                            else 'fouler_away' if pd.notna(row['Fouler']) and pd.notna(row['AwayPlay'])
                                            else np.nan,
                                            axis=1)

        self.data['ViolationPlayer'] = self.data.apply(lambda row: 'violator_home' if pd.notna(row['ViolationPlayer']) and pd.notna(row['HomePlay'])
                                            else 'violator_away' if pd.notna(row['ViolationPlayer']) and pd.notna(row['AwayPlay'])
                                            else np.nan,
                                            axis=1)

        self.data['FreeThrowShooter'] = self.data.apply(lambda row: 'ft_home' if pd.notna(row['FreeThrowShooter']) and pd.notna(row['HomePlay'])
                                            else 'ft_away' if pd.notna(row['FreeThrowShooter']) and pd.notna(row['AwayPlay'])
                                            else np.nan,
                                            axis=1)

        self.data['TurnoverPlayer'] = self.data.apply(lambda row: 'to_player_home' if pd.notna(row['TurnoverPlayer']) and pd.notna(row['HomePlay'])
                                            else 'to_player_away' if pd.notna(row['TurnoverPlayer']) and pd.notna(row['AwayPlay'])
                                            else np.nan,
                                            axis=1)

        self.data['Substitution'] = self.data.apply(lambda row: 'sub_home' if pd.notna(row['EnterGame']) and pd.notna(row['HomePlay'])
                                                    else 'sub_away' if pd.notna(row['EnterGame']) and pd.notna(row['AwayPlay'])
                                                    else np.nan,
                                                    axis=1)


    # Creating arrays of runs
    def home_runner(self):
        # global home_runs
        run = []
        self.home_runs = []
        for idx in self.data.index:
            if self.data.at[idx,'HomePlay'] is not np.nan:
                    if 'makes' in self.data.at[idx,'HomePlay']:
                        run.append(idx)
            elif self.data.at[idx,'AwayPlay'] is not np.nan:
                    if 'makes' in self.data.at[idx,'AwayPlay']:
                        run.clear()
            if len(run) == 4:
                self.home_runs.append(run.copy())
                run.clear()
        return self.home_runs
                    
    def away_runner(self):
        # global away_runs
        run = []
        self.away_runs = []
        for idx in self.data.index:
            if self.data.at[idx,'AwayPlay'] is not np.nan:
                    if 'makes' in self.data.at[idx,'AwayPlay']:
                        run.append(idx)
            elif self.data.at[idx,'HomePlay'] is not np.nan:
                    if 'makes' in self.data.at[idx,'HomePlay']:
                        run.clear()
            if len(run) == 4:
                self.away_runs.append(run.copy())
                run.clear()
        return self.away_runs

    def all_runner(self):
        self.all_runs = []
        self.all_runs.extend(self.home_runs)
        self.all_runs.extend(self.away_runs)
        return self.all_runs

    # Flattening runs
    def runs_iter(self):
        # global runs_df
        self.runs_df = pd.DataFrame()
        for run in self.home_runs:
            a = self.data.loc[run[0]-10:run[0]-1, self.factors].values.ravel()
            a = np.append(a,1)
            self.runs_df = pd.concat([self.runs_df,pd.DataFrame([a.copy()])])
        self.runs_df.columns = self.fact_cols
        self.runs_df['class'] = self.runs_df['class'].fillna(1)
        return self.runs_df

    # Function to remove runs from original Dataframe
    def no_runs_preprocessing(self, data, runs):
        # global no_runs_split
        
        # find the first index of a run
        r = [i[0] for i in runs]  

        # create a list of runs
        r_x = []
        for num in r:
            r_x.extend(range(num - 10, num + 1))

        # mask the df without runs
        self.no_runs_df = data[~data.index.isin(r_x)].reset_index(drop=True)

        # segment the df and keep those that are length of 10
        segment_size = 10
        segments = len(self.no_runs_df) // segment_size

        self.no_runs_split = np.array_split(self.no_runs_df, segments)

        self.no_runs_split = [x for x in self.no_runs_split if len(x) != 11]

        return self.no_runs_split

    # Flattening no runs
    def no_runs_optimized(self, data, factors, fact_cols):
            # global no_runs_df
            self.no_runs_df = pd.DataFrame([np.append(segment.loc[:, factors].values.ravel(), int(0)) for segment in data])
            self.no_runs_df.columns = fact_cols
            return self.no_runs_df

    # Preparing final Dataframe for training
    def final(self):
        # self.scaler = MinMaxScaler((0,255))
        # self.values = pd.DataFrame(self.scaler.fit_transform(self.encoded_df))
        self.values = self.encoded_df
        self.labels = self.final_df.iloc[:,-1]
        return self.values, self.labels

    # Run everything
    def preprocess(self):
        self.feats_engineering()
        self.home_runner()
        self.away_runner()
        self.all_runner()
        self.runs_iter()
        self.no_runs_optimized(self.no_runs_preprocessing(self.data, self.home_runs), self.factors, self.fact_cols)
        combined_df = pd.concat([self.runs_df,self.no_runs_df],ignore_index=True)
        # combined_df.to_csv('preprocessed_data/combined_df.csv', index=False)
        self.encoders()
        return self.final(), combined_df
