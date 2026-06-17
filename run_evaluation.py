"""
3x3 RAG evaluation: 3 embedding models x 3 LLM models x 20 golden questions = 180 answers.
Scores each combination with Ragas (Faithfulness, Answer Relevancy, Context Precision, Context Recall).
"""

import csv
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path


def _load_dotenv():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# Skip FYP.py auto pip install / Gradio is not needed for batch eval.
os.environ.setdefault("SKIP_AUTO_INSTALL", "1")

import pandas as pd
from datasets import Dataset


def _install_ragas_compat_shims():
    """Ragas still imports removed langchain_community VertexAI chat models."""
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        vertexai_chat = types.ModuleType("langchain_community.chat_models.vertexai")

        class _ChatVertexAI:  # noqa: N801
            pass

        vertexai_chat.ChatVertexAI = _ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai_chat


_install_ragas_compat_shims()

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

answer_relevancy.strictness = 1

from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# Config
# ==========================================
GOLDEN_DATASET = [
    {
        "question": "What is the price of iPhone 17?",
        "ground_truth": "RM3,999.00 - RM4,999.00",
    },
    {
        "question": "What color options does the iPhone 17 have?",
        "ground_truth": "Black, Lavender, Mist Blue, Sage, and White.",
    },
    {
        "question": "What variations does the iPhone 17 have?",
        "ground_truth": "256GB, 512GB",
    },
    {
        "question": "Can iPhone 17 accept trade in or not?",
        "ground_truth": "Yes, can",
    },
    {
        "question": "How many payment option iphone 17 has?",
        "ground_truth": "Full Amount, Deposit of RM300.00",
    },
    {
        "question": "iPhone 17 has any gift?",
        "ground_truth": "Screen Protector",
    },
    {
        "question": "What is the price of the vivo x300?",
        "ground_truth": "RM4,699.00",
    },
    {
        "question": "What is the weight of the vivo x300?",
        "ground_truth": "0.226 kg",
    },
    {
        "question": "What is the dimensions of the vivo x300?",
        "ground_truth": "16.198 × 7.548 × 0.799 cm",
    },
    {
        "question": "What is the color of the vivo x300?",
        "ground_truth": "Dune Brown, Mist Blue, Phantom Black",
    },
    {
        "question": "What is the storage of the vivo x300?",
        "ground_truth": "16GB+512GB, 16GB+1TB",
    },
    {
        "question": "Galaxy Tab S11 Wi-Fi has any Free gift?",
        "ground_truth": "Free Samsung S Pen",
    },
    {
        "question": "What is the price of the Xiaomi REDMI Note 15?",
        "ground_truth": "RM1899.00",
    },
    {
        "question": "What is the color of the Xiaomi REDMI Note 15?",
        "ground_truth": "Black, Glacier Blue, Mocha Brown",
    },
    {
        "question": "What are the policies for Shipping, Delivery, and Pick-Up?",
        "ground_truth": (
            "Delivery will take from 3 to 7 working days upon confirmation of order where your order will be processed and dispatched from our warehouse"
            "All small appliances & digital gadgets will be charged with a standard rate of RM8 as courier fee."
            "Delivery is only applicable for West Malaysia only. Applicable to selected items on the Site only."
            "Pick up will be available at the selected locations from 5pm onwards"
            "Majority of the products are entitled for pick up services, except for big items"
        ),
    },
    {
        "question": "What are the policies for cancellation policy?",
        "ground_truth": (
            "The item is not / no longer available, Pricing disputes arise"
            "Product illustration differs from actual product"
            "Upon cancellation, refunds will be processed within 14 working days."
        ),
    },
    {
        "question": "What are the policies for Data Security?",
        "ground_truth": (
            "implement and maintain reasonable and appropriate administrative, physical, and technical security measures"
            "to protect your Personal Data from unauthorised or unlawful processing, and against accidental loss, destruction, damage, alteration, or disclosure."
        ),
    },
    {
        "question": "How to contact your company?",
        "ground_truth": (
            "Livechat: Weekdays, 9:30am - 6:30pm. Hotline: +6011 3600 4040 "
            "Email: ccc@senheng.com.my."
        ),
    },
    {
        "question": "Where your company address?",
        "ground_truth": (
            "44B, Jalan Pandan 3/2, Pandan Jaya, 55100 Kuala Lumpur"
        ),
    },
    {
        "question": "What about policies of Return Policy?",
        "ground_truth": (
            "Most orders are returnable within 7 calendar days upon receipt if they are incorrect, damaged or defective."
            "Incorrect: The item is not the item you ordered. "
        ),
    },
]

LLM_MODELS = ["llama3.2:1b","qwen2.5:1.5b", "gemma3:1b"]

RESULTS_CSV = "FYP_3x3_Evaluation_Results.csv"
RESPONSES_CSV = "FYP_3x3_All_Responses.csv"
CHECKPOINT_JSON = "FYP_3x3_Checkpoint.json"

FORCE_CPU = os.getenv("EXPERIMENT_FORCE_CPU", "1") != "0"
RAGAS_LLM_PROVIDER = os.getenv("RAGAS_LLM_PROVIDER", "google").strip().lower()
RAGAS_LLM_MODEL = os.getenv(
    "RAGAS_LLM_MODEL",
    "gemini-3.1-flash-lite" if RAGAS_LLM_PROVIDER == "google" else "llama3.2:1b",
)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
RAGAS_TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "600"))
MAX_QUESTIONS = int(os.getenv("EXPERIMENT_MAX_QUESTIONS", "0"))  # 0 = all 20


def _build_ragas_llm():
    if RAGAS_LLM_PROVIDER == "google":
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "RAGAS_LLM_PROVIDER=google but GOOGLE_API_KEY is missing. "
                "Set it in .env or your environment."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=RAGAS_LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
        )

    from FYP import OLLAMA_BASE_URL, OLLAMA_NUM_CTX

    return ChatOllama(
        model=RAGAS_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_ctx=OLLAMA_NUM_CTX,
    )


def _mean_or_nan(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return float(numeric.mean())
    return float("nan")


def _fmt_score(value):
    if value != value:  # NaN
        return "nan"
    return f"{value:.4f}"


def _invoke_qa_chain(qa_chain, question: str, use_new_api: bool = True):
    payload = {"input": question}
    if use_new_api:
        return qa_chain.invoke(payload)
    return qa_chain(payload)


def _load_checkpoint():
    if not Path(CHECKPOINT_JSON).exists():
        return {"completed": [], "final_results": [], "responses": []}
    try:
        return json.loads(Path(CHECKPOINT_JSON).read_text(encoding="utf-8"))
    except Exception:
        return {"completed": [], "final_results": [], "responses": []}


def _save_checkpoint(state):
    Path(CHECKPOINT_JSON).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_responses_csv(rows):
    file_exists = Path(RESPONSES_CSV).is_file()
    with open(RESPONSES_CSV, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "embedding_model",
                "llm_model",
                "question",
                "ground_truth",
                "contexts",
                "answer",
                "status",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _save_summary_csv(final_results):
    pd.DataFrame(final_results).to_csv(RESULTS_CSV, index=False)


def main():
    from FYP import (
        MODEL_CONFIGS,
        USE_NEW_API,
        get_qa_chain,
        resolve_source_filter_from_question,
    )

    embedding_models = list(MODEL_CONFIGS.keys())

    dataset_items = GOLDEN_DATASET[:MAX_QUESTIONS] if MAX_QUESTIONS > 0 else GOLDEN_DATASET
    total_combos = len(embedding_models) * len(LLM_MODELS)
    total_answers = total_combos * len(dataset_items)

    print("=" * 72)
    print("FYP 3x3 RAG Evaluation")
    print(f"Embeddings : {len(embedding_models)}")
    for name in embedding_models:
        print(f"  - {name}")
    print(f"LLMs       : {len(LLM_MODELS)}")
    print(f"Questions  : {len(dataset_items)}")
    print(f"Total runs : {total_answers}")
    print(f"Force CPU  : {FORCE_CPU}")
    print(f"Ragas LLM  : {RAGAS_LLM_PROVIDER}:{RAGAS_LLM_MODEL}")
    print("=" * 72)

    eval_llm = _build_ragas_llm()
    eval_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    run_config = RunConfig(
        timeout=RAGAS_TIMEOUT,
        max_workers=2 if RAGAS_LLM_PROVIDER == "google" else 4,
        max_retries=2 if RAGAS_LLM_PROVIDER == "google" else 1,
    )

    state = _load_checkpoint()
    completed = set(tuple(x) for x in state.get("completed", []))
    final_results = list(state.get("final_results", []))

    answer_count = len(state.get("responses", []))
    combo_idx = 0

    for emb_name in embedding_models:
        for llm_name in LLM_MODELS:
            combo_idx += 1
            combo_key = (emb_name, llm_name)
            if combo_key in completed:
                print(f"\n[skip] Combo {combo_idx}/{total_combos} already done: {emb_name} + {llm_name}")
                continue

            print(f"\n[{combo_idx}/{total_combos}] Embedding [{emb_name}] + LLM [{llm_name}]")

            test_data = {
                "question": [],
                "answer": [],
                "contexts": [],
                "ground_truth": [],
            }
            response_rows = []

            for q_idx, item in enumerate(dataset_items, start=1):
                question = item["question"]
                ground_truth = item["ground_truth"]
                test_data["question"].append(question)
                test_data["ground_truth"].append(ground_truth)

                answer_count += 1
                print(f"  Q{q_idx}/{len(dataset_items)} ({answer_count}/{total_answers}): {question[:70]}...")

                try:
                    source_filter = resolve_source_filter_from_question(emb_name, question)
                    qa_chain = get_qa_chain(
                        emb_name,
                        llm_name,
                        force_cpu=FORCE_CPU,
                        source_filter=source_filter,
                    )
                    result = _invoke_qa_chain(qa_chain, question, use_new_api=USE_NEW_API)

                    if isinstance(result, dict):
                        answer = result.get("answer", "No answer generated")
                        context_docs = result.get("context", [])
                        context_texts = (
                            [doc.page_content for doc in context_docs]
                            if isinstance(context_docs, list)
                            else []
                        )
                    else:
                        answer = str(result) if result else "No answer generated"
                        context_texts = []

                    test_data["answer"].append(answer)
                    test_data["contexts"].append(context_texts)
                    status = "ok"
                except Exception as e:
                    print(f"    [error] {e}")
                    answer = "Error"
                    context_texts = []
                    test_data["answer"].append(answer)
                    test_data["contexts"].append(context_texts)
                    status = f"error: {e}"

                response_rows.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding_model": emb_name,
                    "llm_model": llm_name,
                    "question": question,
                    "ground_truth": ground_truth,
                    "contexts": json.dumps(context_texts, ensure_ascii=False),
                    "answer": answer,
                    "status": status,
                })

            _append_responses_csv(response_rows)

            print("  Scoring with Ragas...")
            ragas_dataset = Dataset.from_dict(test_data)
            eval_result = evaluate(
                dataset=ragas_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=eval_llm,
                embeddings=eval_embeddings,
                run_config=run_config,
                batch_size=1,
                raise_exceptions=False,
            )
            score_df = eval_result.to_pandas()

            combo_result = {
                "Embedding Model": emb_name,
                "LLM Model": llm_name,
                "Faithfulness": _mean_or_nan(score_df["faithfulness"]),
                "Answer Relevancy": _mean_or_nan(score_df["answer_relevancy"]),
                "Context Precision": _mean_or_nan(score_df["context_precision"]),
                "Context Recall": _mean_or_nan(score_df["context_recall"]),
            }
            final_results.append(combo_result)
            completed.add(combo_key)

            state["completed"] = [list(x) for x in completed]
            state["final_results"] = final_results
            state["responses"] = state.get("responses", []) + response_rows
            _save_checkpoint(state)
            _save_summary_csv(final_results)

            print(
                "  Scores -> "
                f"Faithfulness={_fmt_score(combo_result['Faithfulness'])}, "
                f"Answer Relevancy={_fmt_score(combo_result['Answer Relevancy'])}, "
                f"Context Precision={_fmt_score(combo_result['Context Precision'])}, " 
                f"Context Recall={_fmt_score(combo_result['Context Recall'])}, "              
            )

    print("\n" + "=" * 72)
    print("Evaluation complete.")
    final_df = pd.DataFrame(final_results)
    print(final_df.to_string(index=False))
    _save_summary_csv(final_results)
    print(f"\nSaved summary : {RESULTS_CSV}")
    print(f"Saved responses: {RESPONSES_CSV}")
    print(f"Saved checkpoint: {CHECKPOINT_JSON}")


if __name__ == "__main__":
    main()
