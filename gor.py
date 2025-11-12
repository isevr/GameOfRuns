import os
os.environ['PYTHONHASHSEED'] = '41'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.random.set_seed(41)

import random
random.seed(41)

import numpy as np
np.random.seed(41)

import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import tree
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


df_train1 = pd.read_csv('data/NBA_PBP_2015-16.csv')
df_train2 = pd.read_csv('data/NBA_PBP_2016-17.csv')
df_train3 = pd.read_csv('data/NBA_PBP_2017-18.csv')
df_train4 = pd.read_csv('data/NBA_PBP_2018-19.csv')
df_train5 = pd.read_csv('data/NBA_PBP_2019-20.csv')

del df_train5['Unnamed: 40']

df_train = pd.concat([df_train1, df_train2, df_train3, df_train4, df_train5])

df_train = df_train.reset_index(drop=True)

df_test = pd.read_csv('test_data/NBA_PBP_2020-21.csv')

factors = ['ShotDist','TimeoutTeam','Substitution', 'Shooter',
           'Rebounder', 'Blocker','Fouler',
          'ReboundType','ViolationPlayer',
          'FreeThrowShooter','TurnoverPlayer']

fact_cols = [col + str((i // 11) % 10 + 1) for i, col in enumerate(factors * 10)]
fact_cols.append('class')


def df_xform(df):
    df_new = df.copy()

    conditions_shot = [
        df_new['ShotDist'] <= 10,
        df_new['ShotDist'] >= 22,
        pd.notna(df_new['ShotDist'])
    ]
    choices_shot = ['close', '3pt', 'mid']
    df_new['ShotDist'] = np.select(conditions_shot, choices_shot, default=None)

    conditions_timeout = [
        pd.notna(df_new['TimeoutTeam']) & (df_new['TimeoutTeam'] == df_new['HomeTeam']),
        pd.notna(df_new['TimeoutTeam']) & (df_new['TimeoutTeam'] != df_new['HomeTeam'])
    ]
    choices_timeout = ['timeout_home', 'timeout_away']
    df_new['TimeoutTeam'] = np.select(conditions_timeout, choices_timeout, default=None)

    conditions_shooter = [
        pd.notna(df_new['Shooter']) & pd.notna(df_new['HomePlay']),
        pd.notna(df_new['Shooter']) & pd.notna(df_new['AwayPlay'])
    ]
    choices_shooter = ['shooter_home', 'shooter_away']
    df_new['Shooter'] = np.select(conditions_shooter, choices_shooter, default=None)

    conditions_rebounder = [
        pd.notna(df_new['Rebounder']) & pd.notna(df_new['HomePlay']),
        pd.notna(df_new['Rebounder']) & pd.notna(df_new['AwayPlay'])
    ]
    choices_rebounder = ['rebounder_home', 'rebounder_away']
    df_new['Rebounder'] = np.select(conditions_rebounder, choices_rebounder, default=None)

    conditions_blocker = [
        pd.notna(df_new['Blocker']) & pd.notna(df_new['HomePlay']),
        pd.notna(df_new['Blocker']) & pd.notna(df_new['AwayPlay'])
    ]
    choices_blocker = ['blocker_home', 'blocker_away']
    df_new['Blocker'] = np.select(conditions_blocker, choices_blocker, default=None)

    conditions_fouler = [
        pd.notna(df_new['Fouler']) & pd.notna(df_new['HomePlay']),
        pd.notna(df_new['Fouler']) & pd.notna(df_new['AwayPlay'])
    ]
    choices_fouler = ['fouler_home', 'fouler_away']
    df_new['Fouler'] = np.select(conditions_fouler, choices_fouler, default=None)

    conditions_violator = [
        pd.notna(df_new['ViolationPlayer']) & pd.notna(df_new['HomePlay']),
        pd.notna(df_new['ViolationPlayer']) & pd.notna(df_new['AwayPlay'])
    ]
    choices_violator = ['violator_home', 'violator_away']
    df_new['ViolationPlayer'] = np.select(conditions_violator, choices_violator, default=None)

    conditions_ft = [
        pd.notna(df_new['FreeThrowShooter']) & pd.notna(df_new['HomePlay']),
        pd.notna(df_new['FreeThrowShooter']) & pd.notna(df_new['AwayPlay'])
    ]
    choices_ft = ['ft_home', 'ft_away']
    df_new['FreeThrowShooter'] = np.select(conditions_ft, choices_ft, default=None)

    conditions_to = [
        pd.notna(df_new['TurnoverPlayer']) & pd.notna(df_new['HomePlay']),
        pd.notna(df_new['TurnoverPlayer']) & pd.notna(df_new['AwayPlay'])
    ]
    choices_to = ['to_player_home', 'to_player_away']
    df_new['TurnoverPlayer'] = np.select(conditions_to, choices_to, default=None)

    conditions_sub = [
        pd.notna(df_new['EnterGame']) & pd.notna(df_new['HomePlay']),
        pd.notna(df_new['EnterGame']) & pd.notna(df_new['AwayPlay'])
    ]
    choices_sub = ['sub_home', 'sub_away']
    df_new['Substitution'] = np.select(conditions_sub, choices_sub, default=None)
    
    return df_new


def home_runner(data):

    home_runs = []
    current_run = []
    for idx in data.index:
        home_made_shot = pd.notna(data.at[idx, 'HomePlay']) and 'makes' in data.at[idx, 'HomePlay']
        
        away_made_shot = pd.notna(data.at[idx, 'AwayPlay']) and 'makes' in data.at[idx, 'AwayPlay']

        if home_made_shot:
            current_run.append(idx)
        elif away_made_shot:
            if len(current_run) >= 4:
                home_runs.append(current_run.copy())
            current_run.clear()
            
    if len(current_run) >= 4:
        home_runs.append(current_run.copy())
        
    return home_runs

def away_runner(data):

    away_runs = []
    current_run = []
    for idx in data.index:
        away_made_shot = pd.notna(data.at[idx, 'AwayPlay']) and 'makes' in data.at[idx, 'AwayPlay']
        
        home_made_shot = pd.notna(data.at[idx, 'HomePlay']) and 'makes' in data.at[idx, 'HomePlay']

        if away_made_shot:
            current_run.append(idx)
        elif home_made_shot:
            if len(current_run) >= 4:
                away_runs.append(current_run.copy())
            current_run.clear()

    if len(current_run) >= 4:
        away_runs.append(current_run.copy())
        
    return away_runs

dfs = [df_train1,df_train2,df_train3,df_train4,df_train5]

dfs_xformed = [df_xform(df) for df in dfs]

h_runs = [home_runner(df) for df in dfs_xformed]
a_runs = [away_runner(df) for df in dfs_xformed]

all_runs = [h+a for h,a in zip(h_runs,a_runs)]

x = [len(season_runs) for season_runs in all_runs] 

h = [len(season_runs) for season_runs in h_runs]

a = [len(season_runs) for season_runs in a_runs] 

seasons = ['2015-16','2016-17','2017-18','2018-19','2019-20']

total_runs_df = pd.DataFrame({'total_runs':x,'season':seasons,'home_runs':h,'away_runs':a})
melted_total_runs = total_runs_df.melt(id_vars='season',var_name='type',value_name='runs')

sns.lineplot(melted_total_runs,x='season',y='runs',hue='type')
plt.title('Runs per Season')
plt.ylabel('Runs')
plt.xlabel('Seasons')
plt.savefig('images/new_algo/runs_per_season.png')


def runs_iter(data, runs, factors):
    runs_list = []
    for run in runs:
        try:
            if run[0]-10 in data.index and run[0]-1 in data.index:
                a = data.loc[run[0]-10:run[0]-1, factors].values.ravel()
                a = np.append(a,1)
                runs_list.append(a) 
        except KeyError:
            pass

    if not runs_list: 
        return pd.DataFrame(columns=fact_cols)

    runs_df = pd.DataFrame(runs_list)
    return runs_df

def no_runs_preprocessing(data, runs):
    r = [i[0] for i in runs]  
    r_x = []
    for num in r:
        r_x.extend(range(num - 10, num + 1))

    no_runs_df = data[~data.index.isin(r_x)].reset_index(drop=True)
    
    segment_size = 10
    segments = len(no_runs_df) // segment_size

    no_runs_split = [] 
    if segments > 0: 
        no_runs_split = np.array_split(no_runs_df, segments)

    no_runs_split = [x for x in no_runs_split if len(x) == 10]
    
    return no_runs_split

def no_runs_optimized(data, factors, fact_cols):
    if not data: 
        return pd.DataFrame(columns=fact_cols)
    no_runs_df = pd.DataFrame([np.append(segment.loc[:, factors].values.ravel(), int(0)) for segment in data])
    no_runs_df.columns = fact_cols
    return no_runs_df


all_seasons_runs_df = []
all_seasons_no_runs_df = []


print("Processing all 5 seasons...")
for season_df, season_h_runs in zip(dfs_xformed, h_runs):
    season_factors_df = season_df[factors]
    
    runs_df = runs_iter(season_factors_df, season_h_runs, factors)
    if not runs_df.empty:
        runs_df.columns = fact_cols
        all_seasons_runs_df.append(runs_df)
    
    no_runs_segments = no_runs_preprocessing(season_factors_df, season_h_runs)
    no_runs_df = no_runs_optimized(no_runs_segments, factors, fact_cols)
    if not no_runs_df.empty:
        all_seasons_no_runs_df.append(no_runs_df)

combined_runs = pd.concat(all_seasons_runs_df, ignore_index=True)
combined_no_runs = pd.concat(all_seasons_no_runs_df, ignore_index=True)

combined_df = pd.concat([combined_runs, combined_no_runs], ignore_index=True)

print("Processing complete.")


combined_df = pd.concat([runs_df,no_runs_df],ignore_index=True)
combined_df.to_csv('home_runs.csv', index=False)
combined_df = pd.read_csv('home_runs.csv')

encoders = []
transformed_data = {} 

for column in combined_df.columns[:-1]:
    le = LabelEncoder()
    encoders.append(le)
    transformed_data[column] = le.fit_transform(combined_df[column]) # NEW

df = pd.DataFrame(transformed_data)
df = pd.concat([df,combined_df.iloc[:,-1]],axis=1)

undersample_df = df[df['class'] == 0 ].sample(n=len(df[df['class'] == 1]),random_state=43)
df = pd.concat([df[df['class'] == 1], undersample_df]).astype('float32')

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1],df.iloc[:, -1], 
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=42)

sal = X_train[y_train == 1]

num_categories = 2
y_train = keras.utils.to_categorical(y_train.values, num_categories).astype('float32')
y_test = keras.utils.to_categorical(y_test.values, num_categories).astype('float32')

a = X_train.values.reshape(-1,10,11).astype('float32')
b = X_test.values.reshape(-1,10,11).astype('float32')


model = Sequential()
model.add(Conv2D(50, (5, 5), strides=1, padding="same", activation="relu", input_shape=(10,11,1)))   
model.add(MaxPool2D((2, 2), strides=1, padding="same"))
model.add(Conv2D(25,(3,3), strides=1, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2), strides=1, padding="same"))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=32, activation="relu"))


model.add(Dense(units=num_categories, activation="softmax"))
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='rmsprop', jit_compile=False)


early_stopping = EarlyStopping(patience=5,verbose=1,monitor='val_accuracy')

history = model.fit(
    a, y_train, epochs=1, verbose=1, validation_data=(b, y_test),callbacks=[early_stopping])



stopped_epoch = len(history.history['loss'])

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(1, stopped_epoch + 1), history.history['loss'])
plt.plot(range(1, stopped_epoch + 1), history.history['val_loss'])
plt.xticks([i for i in range(1, stopped_epoch + 1)])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc='upper right')

plt.subplot(1,2,2)
plt.plot(range(1, stopped_epoch + 1), history.history['accuracy'])
plt.plot(range(1, stopped_epoch + 1), history.history['val_accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'], loc='lower right')
plt.xticks([i for i in range(1, stopped_epoch + 1)])
plt.savefig('images/new_algo/loss.png')


preds = model.predict(a)

print(classification_report(np.argmax(y_train,axis=1), np.argmax(preds,axis=1)))

def loss(x_partial):
    x_partial_tensor = tf.convert_to_tensor(x_partial, dtype=tf.float32)

    full_input = tf.concat([x_partial_tensor, x_missing], axis=0)  
    full_input = tf.expand_dims(full_input, axis=0)  

    predictions = model(full_input)

    run_probability = predictions[0, 1]

    return -tf.math.log(run_probability + 1e-8) 

def confidence_metric(loss_value, length, max_len=10, alpha=1.0):
    normalized_length = length / max_len
    
    confidence = (1 / (1 + alpha * loss_value)) * normalized_length
    return confidence

ranges = {}
confidences = {}

for i in range(1, 10):
    gen_preds = []
    confs = []
    for arr in b[:100]:
        rand_len = i
        
        arr_partial = arr[:rand_len, :].reshape(rand_len, 11, 1)

        x_missing = tf.Variable(np.random.rand(10 - rand_len, 11, 1), dtype=tf.float32)

        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        loss_value = None  
        for step in range(500):
            with tf.GradientTape() as tape:
                loss_value = loss(arr_partial)

            grads = tape.gradient(loss_value, [x_missing])

            optimizer.apply_gradients(zip(grads, [x_missing]))

        missing = np.clip(np.round(x_missing.numpy()), 0, 3)
        
        full = np.concatenate([arr_partial, missing], axis=0)
        
        exp_full = np.expand_dims(full, axis=0)
        exp_full_tensor = tf.convert_to_tensor(exp_full, dtype=tf.float32)
        
        pred = np.argmax(model.predict(exp_full_tensor))
        
        gen_preds.append(pred)
        
        confidence = confidence_metric(loss_value.numpy(), rand_len)
        confs.append(confidence)

    ranges[i] = gen_preds
    confidences[i] = confs

for i in range(1,10):
    print(np.mean(ranges[i]), np.mean(confidences[i]))

predicted_full = pd.DataFrame(full.reshape(-1,110))
predicted_full.columns = X_train.columns
predicted_full = predicted_full.astype(int)

rules = []
for encoder, column in zip(encoders, predicted_full.columns):
    rules.append(encoder.inverse_transform(predicted_full[column])[0])

generated_rules = pd.DataFrame(rules).T
generated_rules.columns = X_train.columns

model.save('pretrained_model_le.keras')
np.save('training_history_le.npy', history.history)

def compute_saliency_map(model, input_image):
    input_image = np.expand_dims(input_image, axis=0)  

    input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)

        prediction = model(input_tensor)

        output_value = prediction

    gradients = tape.gradient(output_value, input_tensor)
    
    saliency = np.abs(gradients.numpy())

    saliency = np.squeeze(saliency)

    return saliency

def compute_saliency_for_dataset(model, dataset):
    num_samples = len(dataset)
    saliency_sum = None

    for i, sample in enumerate(dataset):
        print(f"Processing sample {i+1}/{num_samples}")
        
        saliency_map = compute_saliency_map(model, sample)

        if saliency_sum is None:
            saliency_sum = saliency_map
        else:
            saliency_sum += saliency_map

    saliency_avg = saliency_sum / num_samples

    return saliency_avg

saliency_avg_map = compute_saliency_for_dataset(model, sal.values.reshape(-1,10,11))

plt.figure(figsize=(10, 10))

plt.imshow(saliency_avg_map, cmap='hot', aspect='auto')

plt.grid(False)
plt.axis('on')

plt.title('Average Saliency Map for Scoring Runs')
plt.ylabel('Events\n$\longleftarrow$', rotation=90, labelpad=50, fontsize=12)
plt.xlabel('Features')
plt.xticks(np.arange(len(factors)), factors, rotation=45) 
plt.yticks(np.arange(10), [i for i in range(1, 11)])     

plt.colorbar(shrink=0.5)

plt.tight_layout()

plt.savefig('sal.png')


def trad_class(X_train, y_train, X_test, y_test):
    y_train_1d = np.argmax(y_train, axis=1)
    y_test_1d = np.argmax(y_test, axis=1)

    rfc = RandomForestClassifier()
    dtc = DecisionTreeClassifier()
    svc = SVC()
    knc = KNeighborsClassifier()

    clfs = [rfc, dtc, svc, knc]
    preds = {}

    for clf in clfs:
        print(f"--- Training {clf.__class__.__name__} ---")
        clf.fit(X_train, y_train_1d) 
        pred = clf.predict(X_test)
        preds[clf] = pred
        print(classification_report(y_test_1d, pred)) 

    return preds


trad_class(X_train, y_train, X_test, y_test)