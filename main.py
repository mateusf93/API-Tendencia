from flask import Flask, request, jsonify
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import pickle

colunas = ['Open', 'High', 'Low', 'Close', 'Volume']
model = pickle.load(open('model.sav', 'rb'))

app = Flask(__name__)


@app.route('/tendencia/', methods=['POST'])
def tendencia():
    dados = request.get_json()
    dados_input = pd.DataFrame([dados], columns = colunas)
    tendencia = model.predict(dados_input)
    tendencia = np.asscalar(tendencia)
    
    if tendencia == 0:
        tendencia = 'queda'
    else:   
        tendencia = 'alta'
    
    
    response = {
        'tendencia': tendencia
    }
    return jsonify(response)    

# Configuração do Flask-Migrate
if __name__ == '__main__':
    app.run(debug=True)