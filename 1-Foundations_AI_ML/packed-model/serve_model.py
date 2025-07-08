from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Carrega o modelo
model = joblib.load('iris_model.pkl')

class PredictionInput(BaseModel):
    input: list

@app.post("/predict")
async def predict(data: PredictionInput):
    # Converte a entrada para array e ajusta a forma
    input_data = np.array(data.input).reshape(1, -1)
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
