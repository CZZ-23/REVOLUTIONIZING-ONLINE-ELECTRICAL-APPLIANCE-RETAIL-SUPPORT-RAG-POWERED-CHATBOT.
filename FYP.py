import os
import csv
import json
import hashlib
from datetime import datetime
import importlib
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

def ensure_packages():
    if os.getenv("SKIP_AUTO_INSTALL") == "1":
        print("[setup] SKIP_AUTO_INSTALL=1 -> skipping pip installs.")
        return
    sentinel = Path(".deps_installed")
    sentinel_token = "deps-v10"
    if sentinel.exists():
        try:
            if sentinel.read_text(encoding="utf-8").strip() == sentinel_token:
                print("[setup] Detected existing dependencies; skipping pip installs.")
                return
            sentinel.unlink()
        except Exception:
            sentinel.unlink(missing_ok=True)

    print("[setup] Installing required packages...")
    packages = [
        "numpy<2",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-huggingface>=0.1.0",
        "langchain-ollama>=0.1.0",
        "langchain-classic>=0.1.0",
        "langchain-core>=0.1.0",
        "transformers>=4.34.0,<5.0.0",
        "sentence-transformers==2.2.2",
        "faiss-cpu==1.13.0",
        "pdfplumber==0.11.8",
        "pypdf==6.4.0",
        "gradio==6.5.1",
        "huggingface-hub==0.36.0",
        "ollama>=0.1.0",
    ]
    try:
        for pkg in packages:
            print(f"[setup] Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        sentinel.write_text(sentinel_token, encoding="utf-8")
        print("[setup] All packages installed successfully!")
    except Exception:
        sentinel.unlink(missing_ok=True)
        raise


ensure_packages()

from huggingface_hub import login

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("[warn] HUGGINGFACEHUB_API_TOKEN not set; skipping login.")

try:
    import huggingface_hub as _hfh
    _hfh_ver = getattr(_hfh, "__version__", None)
    from huggingface_hub import hf_hub_download as _hf_hub_download

    def _parse_hf_repo_url(url: str):
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if "resolve" not in parts:
            raise ValueError(f"Unsupported huggingface URL format: {url}")
        resolve_idx = parts.index("resolve")
        repo_id = "/".join(parts[:resolve_idx])
        if not repo_id:
            raise ValueError(f"Unable to determine repo_id from URL: {url}")
        if len(parts) <= resolve_idx + 1:
            raise ValueError(f"Unable to determine revision from URL: {url}")
        revision = parts[resolve_idx + 1]
        filename = "/".join(parts[resolve_idx + 2 :]) or None
        if not filename:
            raise ValueError(f"Unable to determine filename from URL: {url}")
        return repo_id, revision, filename

    def _install_cached_download_shim():
        def _cached_download(*args, **kwargs):
            if args:
                raise TypeError("cached_download only supports keyword arguments in this shim.")

            url = kwargs.pop("url", None)
            repo_id = kwargs.get("repo_id")
            filename = kwargs.get("filename")
            revision = kwargs.get("revision")

            if url:
                try:
                    parsed_repo_id, parsed_revision, parsed_filename = _parse_hf_repo_url(url)
                    repo_id = repo_id or parsed_repo_id
                    filename = filename or parsed_filename
                    if revision is None:
                        revision = parsed_revision
                except ValueError as parse_err:
                    raise RuntimeError(f"Failed to convert huggingface download URL: {parse_err}") from parse_err

            if not repo_id or not filename:
                raise TypeError("cached_download requires either (repo_id, filename) or a valid huggingface URL.")

            kwargs["repo_id"] = repo_id
            kwargs["filename"] = filename
            if revision is not None:
                kwargs["revision"] = revision

            kwargs.pop("legacy_cache_layout", None)
            return _hf_hub_download(**kwargs)

        _hfh.cached_download = _cached_download
        sys.modules["huggingface_hub"].cached_download = _cached_download

    _needs_shim = not hasattr(_hfh, "cached_download")
    if _hfh_ver is not None:
        from pkg_resources import parse_version

        _ver = parse_version(_hfh_ver)
        if _ver < parse_version("0.16.4"):
            raise RuntimeError(f"Incompatible huggingface_hub version: {_hfh_ver}")
        if _ver >= parse_version("0.37.0"):
            _needs_shim = True

    if not _needs_shim:
        _needs_shim = getattr(_hfh.cached_download, "__name__", "") == "hf_hub_download"

    if _needs_shim:
        _install_cached_download_shim()

    from huggingface_hub import cached_download as _cd
    import sentence_transformers as _st
except Exception as _err:
    print("\n[warn] Compatibility issue detected!")
    print("Error details:", _err)
    raise


try:
    try:
        import importlib.metadata as _metadata
    except Exception:
        import importlib_metadata as _metadata

    def _ver(pkg):
        try:
            return _metadata.version(pkg)
        except Exception:
            return "not installed"

    print("\nInstalled package versions:")
    for _pkg in ["sentence-transformers", "huggingface-hub", "pydantic", "langchain", "langchain-community", "langchain-huggingface", "faiss-cpu"]:
        print(f" - {_pkg}: {_ver(_pkg)}")
except Exception:
    pass

# ---------------------
# File: run_chatbot.py
# ---------------------

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import ChatOllama with retry mechanism (importlib avoids stale static-import paths)
def _load_chat_ollama():
    try:
        mod = importlib.import_module("langchain_ollama")
        return mod.ChatOllama
    except ImportError:
        print("[warn] langchain_ollama not found. Attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-ollama>=0.1.0"])
            mod = importlib.import_module("langchain_ollama")
            print("[info] Successfully installed and imported langchain_ollama")
            return mod.ChatOllama
        except Exception as e:
            print(f"[error] Failed to install langchain_ollama: {e}")
            print("[info] Please manually install: pip install langchain-ollama")
            raise


ChatOllama = _load_chat_ollama()

from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# LangChain 1.x exposes chain helpers via langchain_classic (not langchain.chains)
try:
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    USE_NEW_API = True
except ImportError:
    USE_NEW_API = False
    print("[info] Using manual chain construction (langchain_classic not available)")

# Import loaders with Windows compatibility workaround
try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
except (ModuleNotFoundError, ImportError) as e:
    if "pwd" in str(e) or "No module named 'pwd'" in str(e):
        print("[warn] Windows compatibility issue with langchain_community detected.")
        print("[warn] Attempting workaround by patching pwd module...")

        import sys
        class DummyPwd:
            @staticmethod
            def getpwnam(name):
                class Passwd:
                    pw_uid = 0
                    pw_gid = 0
                    pw_dir = ""
                    pw_shell = ""
                return Passwd()
        sys.modules['pwd'] = DummyPwd()

        from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
        print("[info] Successfully imported document loaders with workaround.")
    else:
        raise

DATA_DIR = "D:/Fyp/data"
SUPPORTED_TEXT_GLOBS = ["**/*.txt", "**/*.md", "**/*.csv"]
VECTORSTORE_META_FILE = "data_meta.json"
RAG_EVAL_LOG_FILE = "rag_evaluation_log.csv"

# ============================================
# ★ Modified model list (your requested change)
# ============================================
MODEL_CONFIGS = {
    "BGE Base (bge-base-en-v1.5)": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "vector_root": "D:/Fyp/vectorstores/bge-base-en-v1.5",
        "encode_kwargs": {"normalize_embeddings": True},
    },
    "BERT Base (bert-base-nli-mean-tokens)": {
        "model_name": "sentence-transformers/bert-base-nli-mean-tokens",
        "vector_root": "D:/Fyp/vectorstores/bert-base-nli-mean-tokens",
        "encode_kwargs": {},
    },
    "All-MiniLM L6 (all-MiniLM-L6-v2)": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_root": "D:/Fyp/vectorstores/all_minilm_l6_v2",
        "encode_kwargs": {"normalize_embeddings": True},
    }
}

DEFAULT_MODEL_KEY = next(iter(MODEL_CONFIGS))
DEFAULT_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Ollama LLM configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Models like llama3.2:1b default to 128k context, which needs ~4GB KV cache on CPU.
# RAG only needs a few thousand tokens; keep this low to avoid OOM on modest RAM.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
LLM_MODEL_CHOICES = ["llama3.2:1b","qwen2.5:1.5b","gemma3:1b"]
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
if DEFAULT_LLM_MODEL not in LLM_MODEL_CHOICES:
    LLM_MODEL_CHOICES = [DEFAULT_LLM_MODEL, *LLM_MODEL_CHOICES]

for cfg in MODEL_CONFIGS.values():
    cfg["vector_root"] = Path(cfg["vector_root"])
    cfg["vector_root"].mkdir(parents=True, exist_ok=True)

_embedding_cache = {}
_vectorstore_cache = {}
_vectorstore_signature_cache = {}
_llm_cache = {}
_qa_chain_cache = {}
PHONE_BRANDS = {"vivo", "oppo", "xiaomi", "redmi", "realme", "huawei", "honor", "samsung", "iphone"}

def load_all_documents(path):
    documents = []
    pdf_loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())
    for text_glob in SUPPORTED_TEXT_GLOBS:
        txt_loader = DirectoryLoader(path, glob=text_glob, loader_cls=TextLoader)
        documents.extend(txt_loader.load())
    print(f"Loaded {len(documents)} documents from {path}")
    return documents


def compute_data_signature(path: str) -> dict:
    base = Path(path)
    if not base.exists():
        return {"files": []}

    files = []
    for pattern in ["**/*.pdf", *SUPPORTED_TEXT_GLOBS]:
        for file_path in base.glob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "rel_path": str(file_path.relative_to(base)).replace("\\", "/"),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                })

    files.sort(key=lambda x: x["rel_path"])
    return {"files": files}


def save_vectorstore_metadata(vector_root: Path, signature: dict):
    meta_path = vector_root / VECTORSTORE_META_FILE
    meta_payload = {
        "data_dir": DATA_DIR,
        "signature": signature,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_vectorstore_metadata(vector_root: Path):
    meta_path = vector_root / VECTORSTORE_META_FILE
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_text(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def extract_phone_model_key(text: str):
    tokens = normalize_text(text).split()
    for i, tk in enumerate(tokens):
        if tk not in PHONE_BRANDS:
            continue
        model_tokens = [tk]
        for nxt in tokens[i + 1 : i + 6]:
            if nxt in PHONE_BRANDS:
                break
            if any(ch.isdigit() for ch in nxt) or nxt in {"pro", "plus", "ultra", "max", "mini", "se", "5g", "4g"}:
                model_tokens.append(nxt)
            elif len(model_tokens) >= 2:
                break
        if len(model_tokens) >= 2:
            return " ".join(model_tokens[:3])
    return None


def resolve_source_filter_from_question(model_key: str, question: str):
    question_model = extract_phone_model_key(question)
    if not question_model:
        return None

    cfg = MODEL_CONFIGS[model_key]
    meta = load_vectorstore_metadata(cfg["vector_root"])
    if not meta:
        return None

    files = meta.get("signature", {}).get("files", [])
    q_tokens = set(normalize_text(question_model).split())
    for item in files:
        rel_path = item.get("rel_path", "")
        stem = Path(rel_path).stem
        stem_tokens = set(normalize_text(stem).split())
        if q_tokens and q_tokens.issubset(stem_tokens):
            return normalize_text(stem)
    return None


def get_embedding(model_key: str) -> HuggingFaceEmbeddings:
    if model_key not in _embedding_cache:
        cfg = MODEL_CONFIGS[model_key]
        default_model_kwargs = {"device": cfg.get("device", DEFAULT_DEVICE)}
        custom_model_kwargs = cfg.get("model_kwargs", {})
        model_kwargs = {**default_model_kwargs, **custom_model_kwargs}
        
        _embedding_cache[model_key] = HuggingFaceEmbeddings(
            model_name=cfg["model_name"],
            model_kwargs=model_kwargs,
            encode_kwargs=cfg.get("encode_kwargs") or {},
        )
    return _embedding_cache[model_key]


def build_vectorstore(model_key: str):
    cfg = MODEL_CONFIGS[model_key]
    current_signature = compute_data_signature(DATA_DIR)
    documents = load_all_documents(DATA_DIR)
    if not documents:
        raise RuntimeError(
            "No supported files found in data directory. "
            "Supported formats: .pdf, .txt, .md, .csv"
        )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")
    embedding = get_embedding(model_key)
    _faiss_db = FAISS.from_documents(docs, embedding)
    _faiss_db.save_local(str(cfg["vector_root"]))
    save_vectorstore_metadata(cfg["vector_root"], current_signature)
    print(f"FAISS index saved to: {cfg['vector_root']}")
    return _faiss_db


def load_or_build_vectorstore(model_key: str):
    cfg = MODEL_CONFIGS[model_key]
    index_file = cfg["vector_root"] / "index.faiss"
    store_file = cfg["vector_root"] / "index.pkl"
    existing_meta = load_vectorstore_metadata(cfg["vector_root"])
    current_signature = compute_data_signature(DATA_DIR)
    data_changed = (
        existing_meta is None
        or existing_meta.get("signature") != current_signature
        or existing_meta.get("data_dir") != DATA_DIR
    )

    if data_changed:
        print("Data directory changed; rebuilding index...")
        try:
            index_file.unlink(missing_ok=True)
            store_file.unlink(missing_ok=True)
        except Exception:
            pass
        return build_vectorstore(model_key)

    if index_file.exists() and store_file.exists():
        print(f"Loading existing FAISS index from: {cfg['vector_root']}")
        try:
            # Try with allow_dangerous_deserialization first (newer API)
            try:
                return FAISS.load_local(
                    str(cfg["vector_root"]),
                    embeddings=get_embedding(model_key),
                    allow_dangerous_deserialization=True,
                )
            except TypeError:
                # Fallback for older FAISS versions that don't support this parameter
                return FAISS.load_local(
                    str(cfg["vector_root"]),
                    embeddings=get_embedding(model_key),
                )
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            print(f"[warn] Failed to load index: {e}")
            print("[info] Rebuilding index...")
            try:
                index_file.unlink(missing_ok=True)
                store_file.unlink(missing_ok=True)
            except Exception:
                pass
            return build_vectorstore(model_key)
    print("Index missing; rebuilding...")
    return build_vectorstore(model_key)


def get_vectorstore(model_key: str):
    current_signature = compute_data_signature(DATA_DIR)
    cached_signature = _vectorstore_signature_cache.get(model_key)
    data_changed_while_running = (
        model_key in _vectorstore_cache and cached_signature != current_signature
    )

    if model_key not in _vectorstore_cache or data_changed_while_running:
        if data_changed_while_running:
            print("Detected data changes while running; rebuilding vectorstore...")
            _vectorstore_cache.pop(model_key, None)
            # QA chain holds retriever objects; invalidate chains for this embedding model.
            stale_keys = [k for k in _qa_chain_cache if k.startswith(f"{model_key}_")]
            for key in stale_keys:
                _qa_chain_cache.pop(key, None)

        _vectorstore_cache[model_key] = load_or_build_vectorstore(model_key)
        _vectorstore_signature_cache[model_key] = current_signature
    return _vectorstore_cache[model_key]


def get_llm(model_name: str = DEFAULT_LLM_MODEL, force_cpu: bool = False):
    """Get or create Ollama ChatModel instance"""
    cache_key = f"{model_name}__cpu" if force_cpu else model_name
    if cache_key not in _llm_cache:
        try:
            _llm_cache[cache_key] = ChatOllama(
                model=model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=0.7,
                num_ctx=OLLAMA_NUM_CTX,
                **({"num_gpu": 0} if force_cpu else {}),
            )
            if force_cpu:
                print(f"Initialized Ollama ChatModel (CPU mode): {model_name}")
            else:
                print(f"Initialized Ollama ChatModel: {model_name}")
        except Exception as e:
            print(f"Error initializing Ollama ChatModel {model_name}: {e}")
            print("Make sure Ollama is running and the model is downloaded.")
            print(f"Try running: ollama pull {model_name}")
            raise
    return _llm_cache[cache_key]


def get_qa_chain(model_key: str, llm_model: str = DEFAULT_LLM_MODEL, force_cpu: bool = False, source_filter: str = None):
    """Get or create retrieval chain using compatible LangChain API"""
    # Always refresh vectorstore first so runtime file changes are detected.
    vectorstore = get_vectorstore(model_key)
    signature_payload = _vectorstore_signature_cache.get(model_key, {"files": []})
    signature_text = json.dumps(signature_payload, sort_keys=True, ensure_ascii=False)
    signature_hash = hashlib.md5(signature_text.encode("utf-8")).hexdigest()[:12]
    source_filter_key = source_filter or "all_sources"
    cache_key = f"{model_key}_{llm_model}_{'cpu' if force_cpu else 'gpu'}_{signature_hash}_{source_filter_key}"
    if cache_key not in _qa_chain_cache:
        llm = get_llm(llm_model, force_cpu=force_cpu)
        search_kwargs = {"k": 3}
        if source_filter:
            def source_filter_fn(metadata):
                source = metadata.get("source", "")
                source_norm = normalize_text(str(source))
                return source_filter in source_norm
            search_kwargs["filter"] = source_filter_fn

        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        # Create custom prompt template for ChatModel
        prompt_template = """Use the following pieces of context to answer the question. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {input}

Answer in a helpful, detailed, and accurate manner:"""
        
        # Use ChatPromptTemplate for ChatModel
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        if USE_NEW_API:
            # Use new API if available
            document_chain = create_stuff_documents_chain(llm, prompt)
            _qa_chain_cache[cache_key] = create_retrieval_chain(retriever, document_chain)
        else:
            # Manual chain construction using RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            def get_answer_and_docs(input_dict):
                question = input_dict.get("input", input_dict) if isinstance(input_dict, dict) else input_dict
                docs = retriever.invoke(question)
                context = format_docs(docs)
                messages = prompt.format_messages(context=context, input=question)
                response = llm.invoke(messages)
                answer = response.content if hasattr(response, 'content') else str(response)
                return {"answer": answer, "context": docs}
            
            _qa_chain_cache[cache_key] = get_answer_and_docs
        
        print(f"Created QA chain for embedding: {model_key}, LLM: {llm_model}")
    
    return _qa_chain_cache[cache_key]


# Warm load
get_vectorstore(DEFAULT_MODEL_KEY)

def answer_question(question: str, model_key: str = DEFAULT_MODEL_KEY,
                   llm_model: str = DEFAULT_LLM_MODEL) -> str:
    question = question.strip()
    if not question:
        return "Please input the question."

    source_filter = resolve_source_filter_from_question(model_key, question)
    if source_filter:
        print(f"[info] Applying source filter for model-specific query: {source_filter}")
    force_cpu = True
    for attempt in range(3):
        try:
            qa_chain = get_qa_chain(model_key, llm_model, force_cpu=force_cpu, source_filter=source_filter)
            if USE_NEW_API:
                result = qa_chain.invoke({"input": question})
            else:
                result = qa_chain({"input": question})

            if isinstance(result, dict):
                answer = result.get("answer", str(result) if result else "Sorry, I couldn't generate an answer.")
                context_docs = result.get("context", [])
                context_texts = (
                    [doc.page_content for doc in context_docs]
                    if isinstance(context_docs, list)
                    else []
                )
            else:
                answer = str(result) if result else "Sorry, I couldn't generate an answer."
                context_texts = []

            file_exists = os.path.isfile(RAG_EVAL_LOG_FILE)
            with open(RAG_EVAL_LOG_FILE, mode="a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "embedding_model", "llm", "question", "contexts", "answer"])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    model_key,
                    llm_model,
                    question,
                    json.dumps(context_texts, ensure_ascii=False),
                    answer,
                ])

            return answer
        except Exception as e:
            err_text = str(e)
            if attempt == 0 and "model" in err_text.lower() and "not found" in err_text.lower():
                print(f"[warn] Model '{llm_model}' not found. Attempting to pull it now...")
                try:
                    subprocess.check_call(["ollama", "pull", llm_model])
                    _llm_cache.pop(llm_model, None)
                    _llm_cache.pop(f"{llm_model}__cpu", None)
                    _qa_chain_cache.pop(f"{model_key}_{llm_model}_gpu", None)
                    _qa_chain_cache.pop(f"{model_key}_{llm_model}_cpu", None)
                    print(f"[info] Successfully pulled model '{llm_model}'. Retrying question...")
                    continue
                except Exception as pull_err:
                    print(f"[error] Auto-pull failed for model '{llm_model}': {pull_err}")
            if not force_cpu and "cuda error" in err_text.lower():
                print(f"[warn] CUDA error for model '{llm_model}'. Falling back to CPU mode and retrying...")
                force_cpu = True
                _qa_chain_cache.pop(f"{model_key}_{llm_model}_gpu", None)
                _llm_cache.pop(llm_model, None)
                continue
            print(f"Error with LLM: {e}")
            return f"Sorry, I encountered an error while generating the answer: {str(e)}\n\nPlease make sure Ollama is running and the model '{llm_model}' is available."


def respond(chat_history, message, model_key, llm_model):
    answer = answer_question(message, model_key, llm_model)
    if chat_history is None:
        chat_history = []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history, gr.update(value="", interactive=True)


def clear_chat():
    return [], gr.update(value="", interactive=True)


with gr.Blocks() as demo:
    gr.Markdown("##  🤖SenQ-CHATBOT\nEnter your query, and I will search the knowledge base and generate an intelligent answer using Ollama LLM.")
    chatbot = gr.Chatbot(height=400)
    with gr.Row():
        query_box = gr.Textbox(label="Your Question", placeholder="Please enter the question you would like to ask....", scale=4)
    with gr.Row():
        embedding_model_selector = gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()),
            value=DEFAULT_MODEL_KEY,
            label="Embedding Model",
            scale=1,
        )
        llm_model_selector = gr.Dropdown(
            choices=LLM_MODEL_CHOICES,
            value=DEFAULT_LLM_MODEL if DEFAULT_LLM_MODEL in LLM_MODEL_CHOICES else LLM_MODEL_CHOICES[0],
            label="LLM Model (Ollama)",
            scale=1,
        )
    submit_event = query_box.submit(
        respond,
        [chatbot, query_box, embedding_model_selector, llm_model_selector],
        [chatbot, query_box],
    )
    clear_btn = gr.Button("Clear Chatbox")
    clear_btn.click(
        clear_chat,
        inputs=None,
        outputs=[chatbot, query_box],
        cancels=[submit_event],
        queue=False,
    )


if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port=7860, max_attempts=10):
        """Find a free port starting from start_port"""
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        return None
    
    server_port_env = os.getenv("GRADIO_PORT")
    if server_port_env:
        server_port = int(server_port_env)
    else:
        # Auto-find available port
        server_port = find_free_port()
        if server_port is None:
            print("[warn] Could not find free port, using default 7860")
            server_port = 7860
        else:
            print(f"[info] Using port {server_port}")
    
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    share_pref = os.getenv("GRADIO_SHARE", "auto").lower()
    if share_pref in {"1", "true", "yes"}:
        share = True
    elif share_pref in {"0", "false", "no"}:
        share = False
    else:
        share = True  # default True for environments without localhost access

    try:
        # Gradio 6.x compatible launch
        demo.launch(server_name=server_name, server_port=server_port, share=share)
    except Exception as e:
        print(f"[warn] Launch failed: {e}")
        print("[info] Trying minimal launch configuration...")
        try:
            # Fallback: minimal configuration
            demo.launch(server_name=server_name, server_port=server_port, share=False)
        except Exception as e2:
            print(f"[error] Failed to launch: {e2}")
            raise
