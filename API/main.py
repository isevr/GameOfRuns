import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import time
from uuid import uuid4
from fastapi.staticfiles import StaticFiles
from model.preprocessing import data_load
from model.model_training import model_training
from sequence_mining.sequence_mining import sequence_mining
from optimizer.optimization import SequenceOptimization
import io



app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_baseline_form(request: Request, response: Response):

    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/preprocess", response_class=JSONResponse)
async def preprocess(response: Response, file: UploadFile = File(...)):

    teams = ['DET', 'CLE', 'NOP', 'WAS', 'PHI', 'CHI', 'UTA', 'CHO', 'IND',
       'DEN', 'NYK', 'SAS', 'DAL', 'LAC', 'MIN', 'MEM', 'ATL', 'MIA',
       'OKC', 'TOR', 'BRK', 'GSW', 'LAL', 'POR', 'PHO', 'SAC', 'HOU',
       'MIL', 'ORL', 'BOS']
    
    data_path = f"./uploaded_files/{file.filename}"
    with open(data_path, "wb") as f:
        f.write(file.file.read())

    # # overall
    # df = pd.read_csv(data_path)
    # events, labels, combined_df = data_load(df)
    # events.to_csv('preprocessed_data/events.csv', index=False)
    # labels.to_csv('preprocessed_data/labels.csv', index=False)
    # combined_df.to_csv('preprocessed_data/combined_df.csv', index=False)

    # per team
    for team in teams:
        df = pd.read_csv(data_path)
        df = df[df.HomeTeam == team]
        events_labels, combined_df = data_load(df)
        events, labels = events_labels
        os.makedirs('preprocessed_data/'+str(team), exist_ok=True)
        events.to_csv('preprocessed_data/'+str(team)+'/events.csv', index=False)
        labels.to_csv('preprocessed_data/'+str(team)+'/labels.csv', index=False)
        combined_df.to_csv('preprocessed_data/'+str(team)+'/combined_df.csv', index=False)

    return {
        "message": "Done.'"
    }

@app.get("/model_train", response_class=JSONResponse)
async def model_train(request: Request):
    
    pbp_data = pd.read_csv('preprocessed_data/events.csv')
    labels = pd.read_csv('preprocessed_data/labels.csv')
    model_training(pbp_data, labels)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')  
    buf.seek(0) 

    return StreamingResponse(buf, media_type="image/png")


@app.get("/sequence_mining", response_class=HTMLResponse)
async def seq_min(request: Request, team: str):
    pbp_data = pd.read_csv(f'preprocessed_data/{team}/combined_df.csv')

    df = sequence_mining("home", "away", pbp_data, team)
    
    html_table = df.to_html(classes='table table-striped')
    
    return HTMLResponse(content=html_table)

@app.get("/optimization", response_class=HTMLResponse)
async def optimize(request: Request):

    arr = np.random.rand(4,11)

    opt = SequenceOptimization('pretrained_models/runs_predictor.keras')

    full_sequence, prediction, confidence = opt.opt_loop(arr, steps=5)

    print(prediction, confidence)

    return None