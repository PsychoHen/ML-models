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
        
        print(f"DEBUG - Payload completo recibido: {payload}")

        # 2. EXTRAER DATOS E ID (Navegación profunda según tu log)
        # Tu log mostró que los datos están en payload['data']['data']['rows'][0]
        try:
            data_wrapper = payload.get('data', {})
            inner_data = data_wrapper.get('data', {})
            rows = inner_data.get('rows', [])
            
            if rows:
                row_data = rows[0]
                row_id = row_data.get('Id')
                print(f"DEBUG - ID encontrado en rows: {row_id}")
            else:
                # Fallback si la estructura cambia
                row_data = payload.get('data', payload)
                row_id = row_data.get('Id') or row_data.get('id')
                print(f"DEBUG - ID buscado en fallback: {row_id}")
        except Exception as e:
            print(f"DEBUG - Error extrayendo datos: {e}")
            row_data = {}
            row_id = None

        if not row_id:
            print("ERROR: No se pudo obtener el ID numérico de la fila")
            return {"status": "error", "message": "ID no encontrado", "debug": payload}

        # 3. Preparar DataFrame para el modelo
        input_df = pd.DataFrame([{
            'Sector': row_data.get('Sector'),
            'In store/Ecomm': row_data.get('In store/Ecomm'),
            'Plug In': row_data.get('Plug In'),
            'ORIGEN DEL LEAD': row_data.get('ORIGEN DEL LEAD'),
            'ESTRATEGIA': row_data.get('ESTRATEGIA'),
            'Transacciones': row_data.get('Transacciones'),
            'Ticket promedio': row_data.get('Ticket promedio')
        }])

        # Llenar vacíos con 0 para evitar el error de RandomForest
        input_df = input_df.fillna(0)

        # 4. Predicción
        prediction = model.predict(input_df)[0]
        prediction_final = round(float(prediction), 2)

        # 5. Enviar a NocoDB
        patch_url = f"{BASE_URL}/{BASE_ID}/{TABLE_ID}/{row_id}"
        headers = {
            "xc-token": NOCODB_API_TOKEN,
            "Content-Type": "application/json"
        }
        
        response = requests.patch(
            patch_url, 
            json={"Potencial": prediction_final}, 
            headers=headers
        )

        # --- LOGS DE DEBUG FINALES ---
        print(f"DEBUG - ID Fila Procesada: {row_id}")
        print(f"DEBUG - Predicción: {prediction_final}")
        print(f"DEBUG - NocoDB Status: {response.status_code}")
        print(f"DEBUG - NocoDB Respuesta: {response.text}")

        return {
            "status": "success",
            "prediction": prediction_final,
            "noco_debug": {
                "id_utilizado": row_id,
                "noco_status": response.status_code,
                "noco_response": response.text
            }
        }

    except Exception as e:
        print(f"ERROR CRÍTICO: {str(e)}")
        return {"status": "error", "message": str(e)}