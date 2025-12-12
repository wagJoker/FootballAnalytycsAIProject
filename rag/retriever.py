# rag/retriever.py
import pandas as pd


def simple_filter(
    df: pd.DataFrame,
    match_id: int | None = None,
    team: str | None = None,
    player_id: int | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Простейший retriever: фильтрация по match_id / team / player_id
    и выбор топ-N строк по важности (например, по xG или shots).
    """
    q = df
    if match_id is not None:
        q = q[q["match_id"] == match_id]
    if team is not None:
        q = q[q["team"].str.contains(team, case=False, na=False)]
    if player_id is not None:
        q = q[q["player_id"] == player_id]

    # пример: сортируем по xG, если есть такой столбец
    if "xG" in q.columns:
        q = q.sort_values(by="xG", ascending=False)

    return q.head(top_n)
