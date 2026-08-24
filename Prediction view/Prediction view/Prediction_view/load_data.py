# app/routes.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
from Prediction_view import app
from flask import jsonify, request
from sklearn.preprocessing import StandardScaler
from Prediction_view.pyscripts.get_teams import cargar_equipos_net, get_rivales_validos_net
from Prediction_view.pyscripts.get_teams import cargar_equipos_boost, get_rivales_validos_boost
from Prediction_view.pyscripts.predecir_partida_net import extraer_enfrentamientos,simular_x_partidas_net
from Prediction_view.pyscripts.predecir_partida_boost import predecir_partido

BASE_DIR = Path(__file__).resolve().parent

"""
--------------
--  MODELS  --
--------------
"""
net_model = keras.models.load_model(f"{BASE_DIR}/models/Neuronal_net.keras")
boost_model = model = joblib.load(f"{BASE_DIR}/models/worlds_xgb_model.pkl")

"""
--------------
--  STATS   --
--------------
"""
train_set_net = pd.read_csv(f"{BASE_DIR}/models/stats/train_data2.csv")
net_df = pd.read_csv(f"{BASE_DIR}/models/stats/complete_set.csv")
team_stats = pd.read_csv(f"{BASE_DIR}/models/stats/team_stats.csv")

"""
-----------------------
--  TRANSFORMATION   --
-----------------------
"""
net_scaler = StandardScaler()

lista_columnas =["side","goldat10","goldat15","xpat10","xpat15","avg_dragons_team","avg_barons_team","avg_heralds_team","avg_towers_team","avg_dragons_vs_opp","avg_barons_vs_opp","avg_heralds_vs_opp","avg_towers_vs_opp","wins_vs_opponent"]
train_set_net = train_set_net[lista_columnas]
train_set_net = pd.DataFrame(
    net_scaler.fit_transform(train_set_net),
    columns=train_set_net.columns,
    index=train_set_net.index)

"""
------------------
--  ENDPOINTS   --
-------------------
"""

#Get teams from json file
@app.route("/get_equipos_net")
def get_equipos_net():
    return jsonify({"equipos": cargar_equipos_net()})

@app.route("/get_equipos_boost")
def get_equipos_boost():
    return jsonify({"equipos": cargar_equipos_boost()})

#Given a selected team, get its posibles rivals
@app.route("/get_rivales_net")
def get_rivales_net():
    equipo = request.args.get("equipo")
    return jsonify({"rivales": get_rivales_validos_net(equipo)})

@app.route("/get_rivales_boost")
def get_rivales_boost():
    equipo = request.args.get("equipo")
    return jsonify({"rivales": get_rivales_validos_boost(equipo)})

#Call to predict with the net model
@app.route('/predict-net', methods=['POST'])
def predict_net():
    data = request.json
    equipoA = data['equipoA']
    equipoB = data['equipoB']
    partidas = data['partidas']

    df_partidas = extraer_enfrentamientos(net_df, equipoA, equipoB)
    
    # Llama a tu modelo Keras para hacer la predicción
    prediccion = simular_x_partidas_net(equipoA, equipoB, partidas, net_model, net_scaler, df_partidas)

    #return jsonify({'prediccion': str(prediccion)})
    return jsonify({'prediccion': prediccion})

#Call to predict with the boost model
@app.route('/predict-boost', methods=['POST'])
def predict_boost():
    data = request.json
    equipoA = data['equipoA']
    equipoB = data['equipoB']

    
    # Llama a tu modelo Keras para hacer la predicción
    prediccion = predecir_partido(equipoA, equipoB, team_stats, boost_model)
    print(prediccion)
    return jsonify({'prediccion': prediccion})