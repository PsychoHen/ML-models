# --- CONFIGURACIÓN DE VARIABLES ---
# En Koyeb, crea una variable llamada NOCODB_TOKEN y pega tu clave ahí
NOCODB_API_TOKEN = os.getenv("NOCODB_TOKEN") 
BASE_URL = "https://app.nocodb.com/api/v1/db/data/v1"
BASE_ID = "prz1t6q6jcgmw7i"
TABLE_ID = "miik72oo6rv6liy"

@app.post("/predict")
async def predict(request: Request):
    try:
        model = joblib.load("modelo_potencial.pkl")
        payload = await request.json()
        
        row_data = payload.get('data', payload)
        # NocoDB suele enviar el ID en la columna '#' o 'Id'
        row_id = row_data.get('Id') or row_data.get('id') or row_data.get('#')

        if not row_id:
            return {"error": "No se recibió un ID válido. Revisa si la columna '#' existe."}

        # Preparar datos EXACTAMENTE como los envía NocoDB
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

        # --- ENVÍO DE VUELTA A NOCODB ---
        patch_url = f"{BASE_URL}/{BASE_ID}/{TABLE_ID}/{row_id}"
        headers = {
            "xc-token": NOCODB_API_TOKEN,
            "Content-Type": "application/json"
        }
        
        # Ajustado: Usamos "Potencial" sin el símbolo # si así está en tu tabla
        response = requests.patch(
            patch_url, 
            json={"Potencial": prediction_final}, 
            headers=headers
        )

        return {
            "status": "success",
            "prediction": prediction_final,
            "nocodb_status": response.status_code
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}