import os
import pandas as pd
import joblib
from fastapi import FastAPI, Request
import requests

app = FastAPI()

# --- CONFIGURACIÓN ---
# IMPORTANTE: En Koyeb, crea una variable llamada NOCODB_TOKEN y pega tu clave ahí
NOCODB_API_TOKEN = os.getenv("NOCODB_TOKEN") 
BASE_URL = "https://app.nocodb.com/api/v1/db/data/v1"
BASE_ID = "prz1t6q6jcgmw7i"
TABLE_ID = "miik72oo6rv6liy"

@app.get("/")
def home():
    return {"message": "API de Predicción Activa"}

@app.post("/predict")
async def predict(request: Request):
    try:
        # Cargamos el modelo (ahora funcionará gracias al requirements.txt)
        model = joblib.load("modelo_potencial.pkl")
        payload = await request.json()
        
        row_data = payload.get('data', payload)
        row_id = row_data.get('Id') or row_data.get('id') or row_data.get('#')

        if not row_id:
            return {"error": "No se recibió un ID válido"}

        # Mapeo de datos (Asegúrate que los nombres coincidan con NocoDB)
        input_df = pd.DataFrame([{
            'Sector': row_data.get('Sector'),
            'In store/Ecomm': row_data.get('In store/Ecomm'),
            'Plug In': row_data.get('Plug In'),
            'ORIGEN DEL LEAD': row_data.get('ORIGEN DEL LEAD'),
            'ESTRATEGIA': row_data.get('ESTRATEGIA'),
            'Transacciones': row_data.get('Transacciones'),
            'Ticket promedio': row_data.get('Ticket promedio')
        }])

  

        # Predicción
        prediction = model.predict(input_df)[0]
        prediction_final = round(float(prediction), 2)

        # --- ENVÍO DE VUELTA A NOCODB ---
        patch_url = f"{BASE_URL}/{BASE_ID}/{TABLE_ID}/{row_id}"
        headers = {
            "xc-token": NOCODB_API_TOKEN,
            "Content-Type": "application/json"
        }
        
        # Realizamos la actualización
        response = requests.patch(
            patch_url, 
            json={"Potencial": prediction_final}, 
            headers=headers
        )

        # ESTO ES LO QUE TIENES QUE VER EN LOS LOGS DE KOYEB PARA DEBUGEAR
        print(f"--- INTENTO DE ACTUALIZACIÓN ---")
        print(f"ID: {row_id} | Predicción: {prediction_final}")
        print(f"Respuesta de NocoDB: {response.status_code} - {response.text}")

        return {
            "status": "success",
            "prediction": prediction_final,
            "noco_debug": {
                "status_code": response.status_code,
                "response_text": response.text
            }
        }

    except Exception as e:
        # Esto te avisará en Koyeb si el código falla antes de llegar a NocoDB
        print(f"ERROR CRÍTICO: {str(e)}")
        return {
            "status": "error", 
            "message": str(e)
        }