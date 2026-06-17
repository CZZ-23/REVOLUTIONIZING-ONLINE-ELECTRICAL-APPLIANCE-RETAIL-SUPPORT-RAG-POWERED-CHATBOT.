# REVOLUTIONIZING-ONLINE-ELECTRICAL-APPLIANCE-RETAIL-SUPPORT-RAG-POWERED-CHATBOT

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-00599C?style=for-the-badge&logo=meta&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)

## About the Project

This project is developed as a Final Year Project for the **Bachelor of Computer Science (Honours) (Artificial Intelligence)** program.

It focuses on developing a localized Retrieval-Augmented Generation (RAG) chatbot for the electrical appliance retail industry. The system retrieves relevant information from product manuals and technical documents stored in a local vector database and generates accurate, context-aware responses to user queries. The solution leverages Large Language Models (LLMs), Vector Search, and Natural Language Processing (NLP) techniques to improve information accessibility while ensuring data privacy and security through local deployment.

## Key Features

* **Document-Based Knowledge Retrieval**: Retrieves relevant information from product manuals and technical documents stored in a local vector database for accurate question answering.

* **Retrieval-Augmented Generation (RAG) Pipeline**:
  * **Semantic Search**: Uses embedding models and FAISS vector search to identify the most relevant document chunks based on user queries.
  * **LLM-Powered Response Generation**: Utilizes locally deployed Large Language Models to generate context-aware responses from retrieved information.

* **Intelligent Query Processing**: Supports:
  * Product information retrieval
  * User manual consultation
  * Frequently asked question answering

* **Privacy-Preserving Local Deployment**: Ensures all document processing, vector search, and model inference are performed locally without relying on external cloud services.

* **Interactive Chat Interface**: Provides a user-friendly conversational interface with real-time responses and source-grounded answers for efficient knowledge access.

## How to run 

1. Download Project Files

First, download or clone the full project repository to your local machine.

2. Install Ollama

Download and install Ollama from the official website:

https://ollama.com

After installation, open Command Prompt (CMD) and run the following commands to download the required LLM models:

```bash
ollama run llama3.2:1b
```
```bash
ollama run gemma3:1b
```
```bash
ollama run qwen2.5:1.5b
```

3. Verify Models Installation

After downloading, you can check whether the models are successfully installed by running:

```bash
ollama list
```
4. Run the Application

Navigate into the project folder:

```bash
cd Save location
```

Run the main Python file:

```bash
python FYP.py
```
5. Install Requirements (Auto Setup)

All required dependencies are listed inside the project. When you run FYP.py for the first time, missing packages will be installed automatically.

6. Start Using the System

Once the program is running, open the provided local URL in your browser. You can then start interacting with the RAG chatbot.
