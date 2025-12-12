# src/rag_demo.py
from pathlib import Path
from rag.rag_service import FootballRAGService


def main():
    # Using the dummy file we created (or expect to be there)
    csv_path = Path("runs_kaggle/tracking/metrics.csv")
    
    # Fallback to verify it works even if the specific file path above is problematic in some envs, 
    # but for now we assume it exists as we tried to create it.
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Please ensure data exists.")
        return

    service = FootballRAGService(csv_path)

    question = "Как сыграла команда Home в матче 101 по качеству моментов и xG?"
    print(f"Question: {question}")
    
    answer = service.ask(
        question=question,
        match_id=101,
        team="Home",
        top_n=30,
    )

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
