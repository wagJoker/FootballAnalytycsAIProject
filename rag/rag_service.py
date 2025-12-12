# rag/rag_service.py
from pathlib import Path
from typing import Optional

from .loader import load_matches_csv
from .retriever import simple_filter


def build_context_from_rows(rows) -> str:
    """
    Превращает строки DataFrame в текстовый контекст.
    Это то, что будет "R" в RAG.
    """
    lines = []
    for _, r in rows.iterrows():
        line = (
            f"Match {r.get('match_id')} | Team {r.get('team')} | "
            f"Player {r.get('player_id')} | xG={r.get('xG')} | "
            f"Shots={r.get('shots')} | Passes={r.get('passes')}"
        )
        lines.append(line)
    return "\\n".join(lines)


def generate_answer_with_llm(question: str, context: str) -> str:
    """
    Заглушка для LLM. Здесь ты подключаешь свой API
    (OpenAI, local LLM, etc.), передавая question + context.
    """
    # Псевдокод:
    # response = llm.chat(
    #   system="Ты футбольный аналитик...",
    #   user=f"Вопрос: {question}\\n\\nКонтекст:\\n{context}"
    # )
    # return response.text
    return (
        "RAG stub:\\n\\n"
        f"Question: {question}\\n\\n"
        f"Context used:\\n{context[:1000]}..."
    )


class FootballRAGService:
    def __init__(self, csv_path: str | Path):
        self.df = load_matches_csv(csv_path)

    def ask(
        self,
        question: str,
        match_id: Optional[int] = None,
        team: Optional[str] = None,
        player_id: Optional[int] = None,
        top_n: int = 20,
    ) -> str:
        """
        Основной метод RAG:
        1) достаём релевантные строки;
        2) строим контекст;
        3) генерируем ответ через LLM.
        """
        rows = simple_filter(
            self.df,
            match_id=match_id,
            team=team,
            player_id=player_id,
            top_n=top_n,
        )
        context = build_context_from_rows(rows)
        answer = generate_answer_with_llm(question, context)
        return answer
