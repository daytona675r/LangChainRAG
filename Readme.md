# LangChainRAG

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/daytona675r/LangChainRAG.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/daytona675r/LangChainRAG.svg)](https://github.com/daytona675r/LangChainRAG/commits/main)
[![Issues](https://img.shields.io/github/issues/daytona675r/LangChainRAG.svg)](https://github.com/daytona675r/LangChainRAG/issues)
[![Stars](https://img.shields.io/github/stars/daytona675r/LangChainRAG.svg)](https://github.com/yourusername/LangChainRAG/stargazers)

A Retrieval-Augmented Generation (RAG) application using [LangChain](https://github.com/langchain-ai/langchain), OpenAI, and LangGraph for advanced question answering and agent workflows.

## Features

- Loads and splits web documents for retrieval
- Uses OpenAI embeddings and chat models
- In-memory vector store for fast prototyping
- Graph-based and agent-based RAG workflows
- Streaming responses and iterative retrieval

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/daytona675r/LangChainRAG.git
   cd LangChainRAG
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file with your OpenAI and LangSmith API keys:
     ```
     OPENAI_API_KEY=your-openai-key
     LANGSMITH_API_KEY=your-langsmith-key
     ```

## Usage

- Run the main RAG pipeline:
  ```sh
  python app.py
  ```

- Run the graph-based agent example:
  ```sh
  python graphApp.py
  ```

## License

This project is licensed under the MIT License.

---

*Badges will update automatically based on your GitHub repository. Replace `yourusername` with your actual GitHub username/repo if publishing.*