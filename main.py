import os
import pandas as pd
import joblib
from fastapi import FastAPI, Request
import requests

app = FastAPI()

# --- CONFIGURACIÓN ---
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
        # 1. Cargar el modelo
        model = joblib.load("modelo_potencial.pkl")
        payload = await request.json()
        
        # LOG: Ver qué datos exactos manda NocoDB
        print(f"DEBUG - Datos recibidos: {payload}")
        
        row_data = payload.get('data', payload)
        
        # 2. Capturar ID (todas las variantes posibles)
        row_id = row_data.get('Id') or row_data.get('id') or row_data.get('ID') or row_data.get('#')

        if not row_id:
            print("ERROR: ID no encontrado")
            return {"error": "Falta ID", "payload": payload}

        # 3. Preparar datos para el modelo
        input_df = pd.DataFrame([{
            'Sector': row_data.get('Sector'),
            'In store/Ecomm': row_data.get('In store/Ecomm'),
            'Plug In': row_data.get('Plug In'),
            'ORIGEN DEL LEAD': row_data.get('ORIGEN DEL LEAD'),
            'ESTRATEGIA': row_data.get('ESTRATEGIA'),
            'Transacciones': row_data.get('Transacciones'),
            'Ticket promedio': row_data.get('Ticket promedio')
        }])

        # --- IMPORTANTE: Llenar vacíos para que RandomForest no falle ---
        input_df = input_df.fillna(0)

        # 4. Predicción
        prediction = model.predict(input_df)[0]
        prediction_final = round(float(prediction), 2)

        # 5. Enviar a NocoDB y capturar su respuesta
        patch_url = f"{BASE_URL}/{BASE_ID}/{TABLE_ID}/{row_id}"
        headers = {
            "xc-token": NOCODB_API_TOKEN,
            "Content-Type": "application/json"
        }
        
        # Realizamos el PATCH
        response = requests.patch(
            patch_url, 
            json={"Potencial": prediction_final}, 
            headers=headers
        )

        # --- LOGS DE DIAGNÓSTICO ---
        print(f"DEBUG - Fila: {row_id}")
        print(f"DEBUG - Predicción: {prediction_final}")
        print(f"DEBUG - NocoDB Status: {response.status_code}")
        print(f"DEBUG - NocoDB Body: {response.text}")

        # 6. Retornar todo para ver el resultado en el historial del Webhook
        return {
            "status": "success",
            "prediction": prediction_final,
            "nocodb_info": {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text
            }
        }

    except Exception as e:
        print(f"ERROR CRÍTICO: {str(e)}")
        return {"status": "error", "message": str(e)}