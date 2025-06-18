#  GENESIS: Generative AI for Strategy and Industry Solutions

<p align="center">
  <img src="https://github.com/contributerMe/GENESIS/blob/main/Model%20Outline.png" alt="GENESIS Architecture" width="300">
</p>

<p align="center">
  <strong>MAS for AI use case identification and implementation</strong>
</p>

---

##  Overview

GENESIS streamlines the process of identifying and implementing AI/GenAI solutions for specific industries and companies through intelligent web scraping, RAG systems, and LLM integration.

##  Features

- 🌐 **Web Scraping**: Automated data collection from multiple sources
- 🧠 **RAG System**: Intelligent document processing and retrieval  
- 🤖 **GPT-4 Integration**: Optimized API calls for strategic insights
- 💬 **Chat Interface**: Interactive conversational AI for quick insights
- 📊 **Report Generation**: Executive-ready market research documents
- 📁 **Multiple Formats**: Export as DOCX, TXT, or PDF

---


<p align="center">
  <img src="https://github.com/contributerMe/GENESIS/blob/master/Genesis1.png" alt="Demo" width="1200">
</p>



---

##  Environment Setup

Create `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
SCRAPING_DELAY_MIN=1
SCRAPING_DELAY_MAX=3
VECTOR_STORE_PATH=./data/vector_store
```


---
##  Quick Start

1. **Clone and Setup**
   ```bash
   git clone https://github.com/contributerMe/GENESIS.git
   cd GENESIS
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Add your OpenAI API key to .env
   OPENAI_API_KEY = ""
   ```

3. **Run Application**
   ```bash
   streamlit run app.py
   ```

4. **Access Interface**
   - Choose between **Chat** or **Report Generation** modes


