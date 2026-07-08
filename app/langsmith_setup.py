import os

from dotenv import load_dotenv
from langsmith import traceable

try:
    from app.augmentated_generation import generate_augmented_answer
except ModuleNotFoundError:
    from augmentated_generation import generate_augmented_answer

load_dotenv()


# Keep this test tool for quick agent experiments later.
# def get_weather(city: str) -> str:
#     """Get weather for a given city."""
#     return f"It's always sunny in {city}!"


if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file or environment.")

# LangSmith tracing setup.
has_langsmith_key = bool(os.getenv("LANGSMITH_API_KEY"))
os.environ["LANGSMITH_PROJECT"] = "production-rag"
os.environ["LANGSMITH_TRACING"] = "true" if has_langsmith_key else "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true" if has_langsmith_key else "false"
os.environ["LANGCHAIN_TRACING"] = "false"

if not has_langsmith_key:
    print("LANGSMITH_API_KEY not found: running pipeline without uploaded traces.")


@traceable(name="augmented_generation_pipeline", tags=["rag", "pipeline", "augmented_generation"])
def run_augmented_pipeline(query: str, top_k: int = 4, model_name: str = "gpt-4o-mini") -> dict:
    result = generate_augmented_answer(query=query, top_k=top_k, model_name=model_name)

    return {
        "query": result.query,
        "answer": result.answer,
        "source_count": result.source_count,
        "sources": result.sources,
    }


if __name__ == "__main__":
    output = run_augmented_pipeline(
        query="What is the attention mechanism in transformers?",
        top_k=4,
        model_name="gpt-4o-mini",
    )

    print("\n=== LS Pipeline Output ===")
    print(f"Query: {output['query']}")
    print(f"Retrieved sources: {output['source_count']}")
    print("\nAnswer:")
    print(output["answer"])