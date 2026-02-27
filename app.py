from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from typing import List
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
model_data = joblib.load('disease_model.joblib')
model = model_data['model']
features = model_data['features']
symptom_choices = sorted([s.replace("_", " ") for s in features])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "symptom_choices": symptom_choices
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    symptom1: str = Form(...),
    symptom2: str = Form(None),
    symptom3: str = Form(None),
    symptom4: str = Form(None),
    symptom5: str = Form(None)
):
    try:
        # Get selected symptoms (filter out None values)
        selected_symptoms = [s for s in [symptom1, symptom2, symptom3, symptom4, symptom5] if s]
        
        # Prepare input data
        symptom_values = {s.replace(" ", "_"): 1 for s in selected_symptoms}
        input_data = pd.DataFrame([{f: symptom_values.get(f, 0) for f in features}])
        
        # Get predictions
        probabilities = model.predict_proba(input_data)[0]
        results = sorted(zip(model_data['classes'], probabilities), 
                       key=lambda x: x[1], reverse=True)[:5]  # Top 5 results
        
        # Calculate confidence scores
        predictions = []
        for disease, prob in results:
            score = int(prob * 100)
            if score >= 80:
                confidence = "Very High"
            elif score >= 60:
                confidence = "High"
            elif score >= 40:
                confidence = "Moderate"
            else:
                confidence = "Low"
            
            predictions.append({
                "disease": disease,
                "probability": f"{prob*100:.1f}%",
                "confidence": confidence,
                "score": score,
                "score_color": "green" if score >= 60 else "orange" if score >= 40 else "red"
            })
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "predictions": predictions,
            "selected_symptoms": selected_symptoms
        })
    
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8007, log_level="info")