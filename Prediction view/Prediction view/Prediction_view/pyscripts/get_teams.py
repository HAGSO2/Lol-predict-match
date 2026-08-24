# utils/team_loader.py
import json
import numpy as np
import pandas as pd
from pathlib import Path

def cargar_equipos_net():
    BASE_DIR = Path(__file__).resolve().parent.parent
    with open(f"{BASE_DIR}/static/content/data/equipos-net.json", encoding="utf-8") as f:
        datos = json.load(f)
    return list(datos["equipos"].keys())

def cargar_equipos_boost():
    BASE_DIR = Path(__file__).resolve().parent.parent
    df = pd.read_csv(f"{BASE_DIR}/static/content/data/team_stats.csv")
    return list(df["team"].unique())

def get_rivales_validos_net(equipo_seleccionado):
    BASE_DIR = Path(__file__).resolve().parent.parent
    with open(f"{BASE_DIR}/static/content/data/equipos-net.json", encoding="utf-8") as f:
        datos = json.load(f)
    return datos["equipos"].get(equipo_seleccionado, [])

def get_rivales_validos_boost(equipo_seleccionado):
    BASE_DIR = Path(__file__).resolve().parent.parent
    df = pd.read_csv(f"{BASE_DIR}/static/content/data/team_stats.csv")
    lista = list(df["team"].unique())
    print(len(lista))
    print(equipo_seleccionado)
    lista.remove(equipo_seleccionado)
    print(len(lista))
    return lista