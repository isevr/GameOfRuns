import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

def sequence_mining(team, opponent, df, curr_team):
    # Replace team names with generic labels
    combined_df = df.replace({str(team): 'same', str(opponent): 'other'}, regex=True)
    
    transformed_data = {}
    encoders = []
    for column in combined_df.columns[:-1]:
        le = LabelEncoder()
        encoders.append(le)
        transformed_data[column] = le.fit_transform(combined_df[column])
    
    transformed_df = pd.DataFrame(transformed_data)
    df = pd.concat([transformed_df, combined_df[combined_df.columns[-1]]], axis=1)
    
    # Undersample to balance the dataset
    undersample_len = len(df[df['class'] == 1])
    undersample_df = df[df['class'] == 0].sample(n=undersample_len, random_state=43)
    df = pd.concat([df[df['class'] == 1], undersample_df])
    
    events_idx = {}
    total_sequences = len(df[df['class'] == 1])
    freqs = []
    
    for j, event in zip(range(12, 112, 11), range(10, 0, -1)):
        event_len_df = df[df['class'] == 1].iloc[:, -j:-1] 
        total_sequences_same_length = len(event_len_df)
        
        event_dict = {}
        row_counts = defaultdict(int)
        
        for i in range(len(event_len_df)):
            row_tuple = tuple(event_len_df.iloc[i])
            row_counts[row_tuple] += 1
            
        
        sorted_row_counts = sorted(row_counts.items(), key=lambda x: x[1], reverse=True)
        
        total_seqs = 0
        for i in range(10):
            total_seqs += sorted_row_counts[i][1]

        
        if sorted_row_counts:
            mc_row, max_count = sorted_row_counts[0]
            sc_row, second_max_count = sorted_row_counts[1] if len(sorted_row_counts) > 1 else (None, 0)
        else:
            mc_row, max_count, sc_row, second_max_count = None, 0, None, 0
        
        max_count_ratio = max_count / total_seqs
        second_max_count_ratio = second_max_count / total_seqs
        
        mc_indices = event_len_df.apply(lambda row: tuple(row) == mc_row, axis=1)
        mc_indices = mc_indices[mc_indices].index.tolist()
        
        sc_indices = event_len_df.apply(lambda row: tuple(row) == sc_row, axis=1)
        sc_indices = sc_indices[sc_indices].index.tolist()
        
        events_idx[event] = mc_indices
        
        event_dict['Event'] = abs(event - 11)
        event_dict['Frequency'] = max_count
        event_dict['Ratio'] = np.round(max_count_ratio, 4)
        event_dict['Sec Frequency'] = second_max_count
        event_dict['Sec Ratio'] = np.round(second_max_count_ratio, 4)
        
        try:
            event_dict['Sequence'] = str(combined_df.iloc[mc_indices[0], -j:-1].to_frame().dropna().T.to_dict(orient="records"))
        except IndexError:
            print(mc_indices, j, curr_team)

        try:
            event_dict['Sec Sequence'] = str(combined_df.iloc[sc_indices[0], -j:-1].to_frame().dropna().T.to_dict(orient="records"))
        except IndexError:
            print(sc_indices, j, curr_team)
        
        freqs.append(event_dict)
    
    return pd.DataFrame(freqs)