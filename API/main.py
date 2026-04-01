import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # uncomment to disable GPU processing
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
from fastapi.staticfiles import StaticFiles
from model.preprocessing import data_load
from model.model_training import model_training
from sequence_mining.sequence_mining import sequence_mining
from optimizer.optimization import SequenceOptimization
import io
from pydantic import BaseModel
import joblib


class BasketballEvent(BaseModel):
    ShotDist: str
    TimeoutTeam: str
    Substitution: str
    Shooter: str
    Rebounder: str
    Blocker: str
    Fouler: str
    ReboundType: str
    ViolationPlayer: str
    FreeThrowShooter: str
    TurnoverPlayer: str

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse(
    request=request, 
    name="index.html"
)

@app.get("/upload", response_class=HTMLResponse)
async def upload_baseline_form(request: Request, response: Response):

    return templates.TemplateResponse(
    request=request, 
    name="upload.html"
)

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

@app.get("/optimization_form", response_class=HTMLResponse)
async def show_opt_form(request: Request, team: str):
    return templates.TemplateResponse(
        request=request, 
        name="optimize.html", 
        context={"team": team}
    )

@app.post("/optimization", response_class=HTMLResponse)
async def optimize_sequence(request: Request, team: str):
    form_data = await request.form()
    
    factors = ['ShotDist','TimeoutTeam','Substitution', 'Shooter',
               'Rebounder', 'Blocker','Fouler', 'ReboundType',
               'ViolationPlayer', 'FreeThrowShooter','TurnoverPlayer']
    
    shot_distances = form_data.getlist("ShotDist")
    num_rows = len(shot_distances)
    
    user_events = []
    encoder_path = f'model/encoders'
    
    for i in range(num_rows):
        row_values = []
        for j, factor in enumerate(factors):
            val = form_data.getlist(factor)[i]
            col_index = (i * len(factors)) + j 
            col_name = f"{factor}{i + 1}"
            encoder_filename = f"le_{col_index}_{col_name}.joblib"
            le_path = os.path.join(encoder_path, encoder_filename)
            
            try:
                le = joblib.load(le_path)
                encoded_val = le.transform([str(val)])[0]
            except Exception:
                encoded_val = 0 
            row_values.append(encoded_val)
        user_events.append(row_values)

    partial_sequence = np.array(user_events).astype('float32')
    
    opt = SequenceOptimization('pretrained_models/runs_predictor.keras', encoder_path)
    full_sequence, prediction, confidence = opt.opt_loop(partial_sequence, steps=100)
    
    decoded_events = []
    
    clean_sequence = np.array(full_sequence).reshape(10, len(factors))
    
    for i in range(len(clean_sequence)):
        row_strings = []
        for j, factor in enumerate(factors):
            val = int(clean_sequence[i][j].item())
            
            col_index = (i * len(factors)) + j 
            col_name = f"{factor}{i + 1}"
            encoder_filename = f"le_{col_index}_{col_name}.joblib"
            le_path = os.path.join(encoder_path, encoder_filename)
            
            try:
                le = joblib.load(le_path)
                decoded_str = le.inverse_transform([val])[0]
                
                if decoded_str == 'nan':
                    decoded_str = '-'
            except Exception:
                decoded_str = str(val) 
                
            row_strings.append(decoded_str)
        decoded_events.append(row_strings)

    decoded_df = pd.DataFrame(decoded_events, columns=factors)
    
    decoded_df.index = [f"Event {i+1}" for i in range(len(decoded_df))]

    try:
        conf_score = round(float(np.array(confidence).flatten()[0]), 4)
    except:
        conf_score = confidence

    return templates.TemplateResponse(
        request=request,
        name="results.html", 
        context={
            "prediction": "Run" if prediction == 1 else "No Run",
            "confidence": conf_score,
            "table": decoded_df.to_html(classes='table table-striped table-hover')
        }
    )