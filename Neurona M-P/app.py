from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
# CORS permite el HTML (que es un frontend separado) se pueda comunicar con este backend sin bloqueos de seguridad.
CORS(app) 

# Es obligatorio definir la clase idéntica a como se entrenó para que pickle sepa cómo reconstruirla.
class MpNeurona:
    def __init__(self):
        self.threshold = None
    def model(self, x):
         # INVERTIMOS LA LÓGICA AQUÍ
        # Si la suma de anomalías es menor o igual al umbral, es Benigno (1)
        return sum(x) <= self.threshold
    def predict(self, X):
        return np.array([self.model(x) for x in X])

# Cargar el modelo que exportamos en el Jupyter Notebook
with open('mp_neurona_optimizada.pkl', 'rb') as file:
    modelo = pickle.load(file)

# creacion del endpoint (la ruta) a la que el HTML enviará los datos
@app.route('/predict', methods=['POST'])
def predict():
    # Recibimos el JSON desde el HTML
    data = request.json
    parametros = data.get('features', [])
    
    # Transformamos la lista en el formato que espera la neurona
    X_input = np.array([parametros])
    prediccion = modelo.predict(X_input)
    
    # En el dataset de sklearn: 0 significa Maligno, 1 significa Benigno
    resultado = "Benigno" if prediccion[0] == 1 else "Maligno"
    
    return jsonify({"diagnostico": resultado})

if __name__ == '__main__':
    # Levantamos el servidor web en el puerto 5000
    app.run(debug=True, port=5000)