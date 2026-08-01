# 🧠 DataMind AI

An Agentic GenAI Platform for Automated Data Science.

DataMind AI is an end-to-end AI-powered data science platform that allows users to upload datasets, perform exploratory data analysis, preprocess data, engineer features, train multiple machine learning models, chat with their data using AI, interact with PDF documents using Retrieval-Augmented Generation (RAG), and use an AI Agent that intelligently selects the appropriate tool based on the user's request.

---

## 🚀 Live Demo

🔗 https://datamindai-agentic-ai-platform.streamlit.app

---

## 📌 Features

### 📂 Dataset Upload
- Upload CSV and Excel datasets
- Automatic dataset loading

### 👀 Dataset Preview
- View dataset rows and columns
- Display shape and basic information

### 📊 Dataset Summary
- Dataset overview
- Missing values
- Data types
- Unique values
- Statistical summary

### 📈 Exploratory Data Analysis (EDA)
- Histograms
- Box plots
- Scatter plots
- Correlation heatmap
- Distribution analysis

### 🧹 Data Preprocessing
- Handle missing values
- Remove duplicates
- Encode categorical variables
- Scale numerical features

### ⚙️ Feature Engineering
- Create new features
- Transform existing features
- Feature selection

### 🤖 Machine Learning
- Automatic problem type detection
- Trains multiple ML models
- Compares model performance
- Selects the best model automatically

### 💬 AI Chat
- Ask questions about the uploaded dataset
- Explain EDA results
- Explain preprocessing steps
- Explain ML results
- Provides business insights
- Supports follow-up conversations using chat history

### 📄 RAG (Retrieval-Augmented Generation)
- Upload PDF documents
- Ask questions about PDF content
- Retrieves relevant information using FAISS vector search
- Supports conversational memory

### 🤖 AI Agent
- Automatically selects the appropriate tool
- Routes requests to:
  - AI Chat
  - Machine Learning
  - EDA
  - Preprocessing
  - RAG
- Supports multi-tool reasoning
- Maintains conversation context

---

## 🛠 Technologies Used

### Frontend
- Streamlit
- Streamlit Option Menu

### Programming Language
- Python

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn
- Plotly

### Machine Learning
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

### Generative AI
- Groq API
- Llama 3.3 70B

### RAG
- Sentence Transformers
- FAISS
- PyMuPDF

### Others
- Python Dotenv
- OpenPyXL

---

## 📂 Project Structure

```
DataMindAI
│
├── app/
├── rag/
├── agents/
├── utils/
├── prompts/
├── config/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚡ Installation

Clone the repository

```bash
git clone https://github.com/NaveenM7777/DataMindAI.git
```

Go to project folder

```bash
cd DataMindAI
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

Run the application

```bash
streamlit run app.py
```
---

## 🎯 Future Improvements

- User Authentication (Login & Register)
- Report Generation (PDF)
- Model Download
- Cloud Storage Integration
- Multi-Agent Collaboration
- Dashboard Export
- Deployment on AWS/Azure

---

## 👨‍💻 Author

**Naveen Mahasamudram**

LinkedIn:
[(https://www.linkedin.com/in/naveen101022/)](https://www.linkedin.com/in/naveen101022/)

GitHub:
https://github.com/NaveenM7777

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub.
