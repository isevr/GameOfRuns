import pandas as pd
from API.model.preprocessing_functions import Preprocessing

def data_load(df):
    # df = pd.read_csv(data_path)

    train = Preprocessing(df)
    
    return train.preprocess()