


import pandas as pd
import joblib
from fastapi import FastAPI, Request
import requests
import uvicorn
import os

app = FastAPI()

# --- CONFIGURACIÓN DE VARIABLES ---
# Estos valores se leerán de lo que configuraste en el panel de Koyeb
NOCODB_API_TOKEN = os.getenv("rw_AZF4bcQqEbO1lfVFd-vHo3gCRqnPMaBG9Co-W")
BASE_URL = "https://app.nocodb.com/api/v1/db/data/v1"
BASE_ID = "prz1t6q6jcgmw7i"
TABLE_ID = "miik72oo6rv6liy"

@app.get("/")
def home():
    return {"message": "API de Predicción NocoDB - Activa"}

@app.post("/predict")
async def predict(request: Request):
    try:
        model = joblib.load("modelo_potencial.pkl")
        payload = await request.json()
        
        # NocoDB envía los datos dentro de 'data' o directamente
        row_data = payload.get('data', payload)
        row_id = row_data.get('Id') or row_data.get('id')

        if not row_id:
            return {"error": "No se recibió un ID válido"}

        # Preparar datos para el modelo
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

        # --- CAMBIO AQUÍ: Nombre de columna con el símbolo # ---
        patch_url = f"{BASE_URL}/{BASE_ID}/{TABLE_ID}/{row_id}"
        headers = {
            "xc-token": NOCODB_API_TOKEN,
            "Content-Type": "application/json"
        }
        
        # Enviamos el dato a la columna '# Potencial'
        response = requests.patch(
            patch_url, 
            json={"# Potencial": prediction_final}, 
            headers=headers
        )

        return {
            "status": "success",
            "prediction": prediction_final,
            "nocodb_response_code": response.status_code,
            "nocodb_response_body": response.text
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)