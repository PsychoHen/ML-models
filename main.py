import pandas as pd
import joblib
from fastapi import FastAPI, Request
import requests
import uvicorn

app = FastAPI()

# --- CONFIGURACIÓN DE NOCODB ---
# He usado el token que me pasaste y los IDs de tu URL
NOCODB_API_TOKEN = "rw_AZF4bcQqEbO1lfVFd-vHo3gCRqnPMaBG9Co-W"
BASE_URL = "https://app.nocodb.com/api/v1/db/data/v1"
BASE_ID = "prz1t6q6jcgmw7i"
TABLE_ID = "miik72oo6rv6liy"

# 1. Cargar el modelo .pkl al iniciar la aplicación
# Asegúrate de que el archivo se llame exactamente así en GitHub
model = joblib.load("modelo_potencial.pkl")

@app.get("/")
def home():
    return {"message": "API de Predicción de Potencial activa"}

@app.post("/predict")
async def predict(request: Request):
    try:
        # 2. Recibir datos del Webhook de NocoDB
        payload = await request.json()
        
        # NocoDB suele enviar la fila dentro de una clave llamada 'data'
        row_data = payload.get('data', payload)
        row_id = row_data.get('Id') or row_data.get('id')

        if not row_id:
            return {"error": "No se encontró el ID de la fila"}

        # 3. Mapear los datos recibidos al formato del modelo
        # Importante: Los nombres a la izquierda deben ser IGUALES a los del Excel original
        input_df = pd.DataFrame([{
            'Sector': row_data.get('Sector'),
            'In store/Ecomm': row_data.get('In store/Ecomm'),
            'Plug In': row_data.get('Plug In'),
            'ORIGEN DEL LEAD': row_data.get('ORIGEN DEL LEAD'),
            'ESTRATEGIA': row_data.get('ESTRATEGIA'),
            'Transacciones': row_data.get('Transacciones'),
            'Ticket promedio': row_data.get('Ticket promedio')
        }])

        # 4. Ejecutar la predicción
        prediction = model.predict(input_df)[0]
        prediction_final = round(float(prediction), 2)

        # 5. Enviar el resultado de vuelta a NocoDB (PATCH)
        # Se asume que tu columna en NocoDB se llama "Potencial"
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
            "nocodb_status": response.status_code
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)