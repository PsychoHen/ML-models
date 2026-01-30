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

        prediction = model.predict(input_df)[0]
        prediction_final = round(float(prediction), 2)

        # Enviar de vuelta a NocoDB
        patch_url = f"{BASE_URL}/{BASE_ID}/{TABLE_ID}/{row_id}"
        headers = {
            "xc-token": NOCODB_API_TOKEN,
            "Content-Type": "application/json"
        }
        
        # Usamos "Potencial" sin el símbolo #
        response = requests.patch(
            patch_url, 
            json={"Potencial": prediction_final}, 
            headers=headers
        )

        # --- AQUÍ SE ENVÍA EL DATO ---
        response = requests.patch(
            patch_url, 
            json={"Potencial": prediction_final}, 
            headers=headers
        )

        # --- REEMPLAZA TU RETURN VIEJO POR ESTE ---
        # Esto imprimirá en los logs de Koyeb lo que NocoDB responda
        print(f"DEBUG - Status Code: {response.status_code}")
        print(f"DEBUG - Respuesta completa: {response.text}")

        return {
            "status": "success",
            "prediction": prediction_final,
            "debug_noco": {
                "id_usado": row_id,
                "status_noco": response.status_code,
                "detalle": response.text  # Aquí verás si NocoDB dice "error: column not found"
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}