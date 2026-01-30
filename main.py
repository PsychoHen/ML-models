import pandas as pd
import joblib
from fastapi import FastAPI, Request
import requests
import uvicorn
import os

app = FastAPI()

# --- CONFIGURACIÓN CON TUS DATOS REALES ---
# El código buscará estos valores en el panel de Koyeb
NOCODB_API_TOKEN = os.getenv("rw_AZF4bcQqEbO1lfVFd-vHo3gCRqnPMaBG9Co-W")
BASE_URL = "https://app.nocodb.com/api/v1/db/data/v1"
BASE_ID = "prz1t6q6jcgmw7i"
TABLE_ID = "miik72oo6rv6liy"

# Cargar el modelo .pkl
# Asegúrate de que el archivo en GitHub se llame exactamente así
model = joblib.load("modelo_potencial.pkl")

@app.get("/")
def home():
    return {"message": "API de Predicción NocoDB - Activa"}

@app.post("/predict")
async def predict(request: Request):
    try:
        # 1. Recibir datos de NocoDB
        payload = await request.json()
        row_data = payload.get('data', payload)
        row_id = row_data.get('Id') or row_data.get('id')

        if not row_id:
            return {"error": "No se recibió un ID válido de NocoDB"}

        # 2. Mapear datos al formato del modelo
        # Usamos los nombres de columnas de tu CSV original
        input_df = pd.DataFrame([{
            'Sector': row_data.get('Sector'),
            'In store/Ecomm': row_data.get('In store/Ecomm'),
            'Plug In': row_data.get('Plug In'),
            'ORIGEN DEL LEAD': row_data.get('ORIGEN DEL LEAD'),
            'ESTRATEGIA': row_data.get('ESTRATEGIA'),
            'Transacciones': row_data.get('Transacciones'),
            'Ticket promedio': row_data.get('Ticket promedio')
        }])

        # 3. Realizar predicción
        prediction = model.predict(input_df)[0]
        prediction_final = round(float(prediction), 2)

        # 4. Enviar resultado de vuelta a NocoDB
        # IMPORTANTE: Tu columna en NocoDB debe llamarse "Potencial"
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

        return {
            "status": "success",
            "row_id": row_id,
            "prediction": prediction_final,
            "nocodb_http_code": response.status_code
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)