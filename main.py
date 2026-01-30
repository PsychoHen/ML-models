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
    # Esta es la respuesta rápida para que Koyeb vea que la app está viva
    return {"message": "API de Predicción NocoDB - Activa"}

@app.post("/predict")
async def predict(request: Request):
    try:
        # 1. Cargamos el modelo aquí adentro para que el arranque sea veloz
        # Asegúrate de que el archivo en GitHub sea 'modelo_potencial.pkl'
        model = joblib.load("modelo_potencial.pkl")

        # 2. Recibir datos de NocoDB
        payload = await request.json()
        row_data = payload.get('data', payload)
        row_id = row_data.get('Id') or row_data.get('id')

        if not row_id:
            return {"error": "No se recibió un ID válido"}

        # 3. Preparar datos para el modelo (usando los nombres de tu CSV)
        input_df = pd.DataFrame([{
            'Sector': row_data.get('Sector'),
            'In store/Ecomm': row_data.get('In store/Ecomm'),
            'Plug In': row_data.get('Plug In'),
            'ORIGEN DEL LEAD': row_data.get('ORIGEN DEL LEAD'),
            'ESTRATEGIA': row_data.get('ESTRATEGIA'),
            'Transacciones': row_data.get('Transacciones'),
            'Ticket promedio': row_data.get('Ticket promedio')
        }])

        # 4. Predicción
        prediction = model.predict(input_df)[0]
        prediction_final = round(float(prediction), 2)

        # 5. Enviar a NocoDB
        # Asegúrate de que la columna en NocoDB se llame exactamente 'Potencial'
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
            "prediction": prediction_final,
            "nocodb_response": response.status_code
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Puerto 8000 como configuraste en Koyeb
    uvicorn.run(app, host="0.0.0.0", port=8000)