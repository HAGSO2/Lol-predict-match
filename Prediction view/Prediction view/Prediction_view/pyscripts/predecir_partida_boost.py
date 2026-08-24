import pandas as pd

def obtener_estadisticas(team_name, team_stats):
    
    equipo = team_stats[team_stats["team"] == team_name]

    if equipo.empty:
        raise ValueError(f"No existe el equipo '{team_name}'")

    return equipo.iloc[-1]


def construir_features(team1, team2):

    return pd.DataFrame([{

        "winrate_diff":
            team1["winrate"] - team2["winrate"],

        "gold15_diff":
            team1["gold15"] - team2["gold15"],

        "xp15_diff":
            team1["xp15"] - team2["xp15"],

        "cs15_diff":
            team1["cs15"] - team2["cs15"],

        "dragon_diff":
            team1["dragons"] - team2["dragons"],

        "baron_diff":
            team1["barons"] - team2["barons"],

        "tower_diff":
            team1["towers"] - team2["towers"]

    }])


def predecir_partido(team1_name, team2_name, team_stats, model):

    team1 = obtener_estadisticas(team1_name, team_stats)
    team2 = obtener_estadisticas(team2_name, team_stats)

    X = construir_features(team1, team2)

    pred = model.predict(X)[0]

    prob = model.predict_proba(X)[0]

    if pred == 1:
        ganador = team1_name
        confianza = prob[1]
    else:
        ganador = team2_name
        confianza = prob[0]

    print(f"{team1_name} vs {team2_name}")
    print("-" * 30)
    print(f"Ganador: {ganador}")
    print(f"Probabilidad: {confianza:.2%}")

    return {"ganador":ganador,"confianza": float(confianza)}