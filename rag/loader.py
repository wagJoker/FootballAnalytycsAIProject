# rag/loader.py
from pathlib import Path
import pandas as pd


def load_matches_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Загружает метрики/события по матчам из CSV.

    Ожидается, что CSV содержит столбцы вроде:
    match_id, team, player_id, xG, passes, shots, pressure_zone, ...
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    return df
