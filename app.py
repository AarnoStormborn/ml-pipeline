import pickle
import uvicorn
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from utils import data_pipeline

class TrainingSample(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str
    charges: float 

class TestingSample(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

app = FastAPI()


@app.get('/checkhealth')
async def root():
    return {"Status": "Application Running"}

@app.post('/train')
async def train_model(samples:List[TrainingSample]):

    samples_list = [sample.model_dump() for sample in samples]
    df = pd.DataFrame(samples_list)

    X = df.drop(['charges'], axis=1)
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = data_pipeline(X)

    rgs = RandomForestRegressor()

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rgs)
    ])

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    score = f"{score*100:.2f}%"

    filename = 'model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    return {"Message": "Model Trained", "Score": score}


@app.post('/test')
async def test_model(sample:TestingSample):

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.DataFrame([sample.model_dump()])

    y_pred = model.predict(df)
    return {"Prediction":y_pred[0]}


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
