import numpy as np
import pandas as pd
from tensorflow import keras

def extraer_enfrentamientos(df, team_a, team_b):
    """
    Extrae todas las filas de enfrentamientos entre team_a y team_b.
    Usa operaciones vectorizadas de pandas (muy rápido).
    """
    # Paso 1: Encontrar los gameid donde jugaron AMBOS equipos
    # groupby agrupa por gameid, y apply(set) crea un conjunto con los 2 equipos
    teams_por_game = df.groupby('gameid')['team'].apply(set)
    
    # Filtramos los gameid donde están team_a Y team_b
    gameids_validos = teams_por_game[
        teams_por_game.apply(lambda equipos: team_a in equipos and team_b in equipos)
    ].index
    
    # Paso 2: Extraer todas las filas de esos gameid
    df_enfrentamientos = df[df['gameid'].isin(gameids_validos)].copy()
    
    return df_enfrentamientos

def simular_x_partidas_net(equipo_a, equipo_b, num_partidas, modelo, scaler, df_equipos):
    """
    Simula num_partidas entre equipo_a y equipo_b.
    Añade ruido para que no siempre salga lo mismo.
    """

    lista_columnas_entrenamiento = ["side","goldat10","goldat15","xpat10","xpat15","avg_dragons_team","avg_barons_team","avg_heralds_team","avg_towers_team","avg_dragons_vs_opp","avg_barons_vs_opp","avg_heralds_vs_opp","avg_towers_vs_opp","wins_vs_opponent"]
    
    victorias_a = 0
    victorias_b = 0
    
    for _ in range(num_partidas):
        # 1. Obtener features base del equipo_a
        features_a = df_equipos[df_equipos['team'] == equipo_a].copy()
        features_a = features_a[lista_columnas_entrenamiento]
        
        # 2. AÑADIR RUIDO (variabilidad)
        # Esto simula que cada partida es diferente (draft, estado de los jugadores, etc.)
        ruido = np.random.normal(0, 0.1, features_a.shape)  # Media 0, desviación 0.1
        features_a_con_ruido = features_a.values + ruido
        
        # 3. Escalar
        features_escaladas = scaler.transform(features_a_con_ruido)
        
        # 4. Predecir
        prob_a_gana = modelo.predict(features_escaladas)[0][0]
        
        # 5. Decidir ganador basado en la probabilidad
        if np.random.random() < prob_a_gana:
            victorias_a += 1
        else:
            victorias_b += 1
    
    return {
        'equipo_a': equipo_a,
        'equipo_b': equipo_b,
        'victorias_a': victorias_a,
        'victorias_b': victorias_b,
        'porcentaje_a': (victorias_a / num_partidas) * 100,
        'porcentaje_b': (victorias_b / num_partidas) * 100
    }

"""
# Simular 100 partidas entre LNG Esports y JD Gaming
resultado = simular_x_partidas(
    equipo_a='LNG Esports',
    equipo_b='JD Gaming',
    num_partidas=100,
    modelo=model,
    scaler=scaler,
    df_equipos=df_equipos
)

print(f"Resultado de 100 simulaciones:")
print(f"{resultado['equipo_a']}: {resultado['victorias_a']} victorias ({resultado['porcentaje_a']:.1f}%)")
print(f"{resultado['equipo_b']}: {resultado['victorias_b']} victorias ({resultado['porcentaje_b']:.1f}%)")
"""