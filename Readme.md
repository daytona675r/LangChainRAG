# LangChainRAG

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/daytona675r/LangChainRAG.svg)](LICENSE)

<!-- Framework & Tool Badges -->
[![LangChain](https://img.shields.io/badge/LangChain-2B5DFF?logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjZmZmIiB2aWV3Qm94PSIwIDAgMzAgMzAiIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCI+PHJlY3Qgd2lkdGg9IjMwIiBoZWlnaHQ9IjMwIiByeD0iNSIgZmlsbD0iIzJiNWRmZiIvPjxwYXRoIGQ9Ik0xNSAxM2MtMS43IDAtMy0xLjMtMy0zcyAxLjMtMyAzLTMgMyAxLjMgMyAzLTEuMyAzLTMgM3oiIGZpbGw9IiNmZmYiLz48L3N2Zz4=)](https://github.com/langchain-ai/langchain)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-FFB800?logo=github&logoColor=black)](https://github.com/langchain-ai/langgraph)
[![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-4B8BBE?logo=python&logoColor=white)](https://www.crummy.com/software/BeautifulSoup/)
[![dotenv](https://img.shields.io/badge/dotenv-10AA50?logo=python&logoColor=white)](https://pypi.org/project/python-dotenv/)
[![IPython](https://img.shields.io/badge/IPython-007ACC?logo=python&logoColor=white)](https://ipython.org/)

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

