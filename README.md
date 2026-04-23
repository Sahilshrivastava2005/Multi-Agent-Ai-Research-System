# ResearchMind: Multi-Agent AI Research System

A powerful research pipeline powered by **LangGraph**, **LangChain**, and **Google Gemini**. Four specialized AI agents collaborate to deliver a polished research report on any topic.

## 🚀 Features
- **Search Agent**: Gathers recent web information using Tavily.
- **Reader Agent**: Scrapes and extracts deep content from relevant resources.
- **Writer Chain**: Drafts a structured research report in Markdown.
- **Critic Chain**: Reviews and scores the report for quality and accuracy.
- **Modern UI**: Sleek, dark-mode Streamlit interface with real-time pipeline tracking.

## 🛠️ Setup

1. **Clone the repository** (or navigate to the directory).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   Create a `.env` file in the root directory with your API keys:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## 🧠 Architecture
The system uses **LangGraph** to manage the multi-agent workflow:
`Search` → `Reader` → `Writer` → `Critic` → `Final Report`

## 📦 Requirements
- Python 3.9+
- LangChain / LangGraph
- Streamlit
- Google Generative AI (Gemini)
- Tavily Search
# Multi-Agent-Ai-Research-System
