# 📊 AI Business Assistant — Intelligent Sales Analytics, ML Forecasting & Sentiment Analysis

**منصة ذكية مدعومة بالذكاء الاصطناعي وتعلم الآلة لمساعدة أصحاب المتاجر والشركات على تحليل المبيعات، التنبؤ بالمبيعات المستقبلية، تحليل مشاعر وآراء العملاء، وتقديم توصيات عملية عبر مساعد ذكي (Chatbot) يدعم اللغتين العربية والإنجليزية.**

An AI-powered decision-support platform designed to help business and store owners analyze historical sales data and customer reviews, forecast future demand using Machine Learning, and gauge customer satisfaction with products and services using Natural Language Processing (NLP). It includes a bilingual **Smart Business Chatbot** that answers owners' questions based strictly on real computed metrics and provides actionable recommendations to boost sales.

---

## ✨ Key Features

1. **📊 Comprehensive Sales Analytics (`Pandas` & `Plotly`)**
   - Tracks core business KPIs: **Total Revenue**, **Total Orders**, **Unique Customers**, **Units Sold**, and **Average Order Value (AOV)**.
   - Interactive charts for **Monthly Revenue Trends**, **Top-Selling vs. Lowest-Performing Products**, and **Regional/Country Revenue Breakdown**.
2. **📈 Machine Learning Sales Forecasting (`Random Forest`)**
   - Engineers time-series lag features (`lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`), rolling window averages (`7-day` and `14-day` rolling mean/std), and calendar indicators (`day of week`, `is_weekend`, `month`).
   - Evaluates a **Random Forest Regressor** against a baseline **Dummy Regressor** on a chronological 60-day test window using **MAE**, **RMSE**, and **R² Score**, and generates an interactive **14-to-90 Day Future Sales Forecast** with feature importance rankings.
3. **💬 Customer Review Sentiment Analysis (`NLP`, `TF-IDF`, `Logistic Regression`)**
   - Classifies customer feedback into **Positive**, **Neutral**, and **Negative** using uni-gram/bi-gram **TF-IDF Vectorization** and balanced **Logistic Regression** (supporting both **English** and **Arabic** reviews).
   - Displays classification **Accuracy**, **Macro F1-Score**, **Confusion Matrix**, and extracts the **Top Customer Complaints** (most frequent negative phrases). Includes a live review tester with confidence scores.
4. **🤖 Smart Bilingual Business Chatbot (English & Arabic)**
   - Answers natural-language questions about best/worst products, total revenue, future sales trends, regional performance, and customer complaints using 100% real data from the analytics and ML pipelines.
5. **💡 Actionable Business Recommendations**
   - Automatically synthesizes insights from low-performing SKUs, top customer complaints, AOV benchmarks, and the 30-day forecast trend into prioritized action items.

---

## 🛠️ Technologies Used

| Layer | Technologies & Libraries |
| :--- | :--- |
| **Programming Language** | Python 3.9+ |
| **Data Manipulation & KPIs** | Pandas, NumPy, OpenPyXL |
| **Machine Learning & NLP** | Scikit-learn (`RandomForestRegressor`, `DummyRegressor`, `TfidfVectorizer`, `LogisticRegression`), Joblib |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend Dashboard** | Streamlit, Plotly Express (Bilingual EN/AR with RTL support) |
| **Experimentation & Sequence** | Jupyter Notebook (`notebook/main.ipynb`) |

---

## 📁 Project Structure

The project is organized into a clean, modular architecture separating the **FastAPI Backend**, **Streamlit Frontend**, **Trained AI Models**, and **Jupyter Notebook**:

```text
ai_business_assistant/
├── backend/                  # FastAPI backend server & core ML/analytics modules
│   ├── __init__.py
│   ├── main.py               # FastAPI REST API script (endpoints for KPIs, Forecast, Sentiment, Chat)
│   ├── data_prep.py          # Data cleaning & built-in store dataset generator (no external CSV needed)
│   ├── analytics.py          # Sales KPIs, product rankings, monthly & country aggregations
│   ├── forecasting.py        # Random Forest time-series forecaster & baseline comparison
│   ├── sentiment.py          # TF-IDF + Logistic Regression NLP pipeline & complaint extractor
│   ├── assistant.py          # Bilingual Business Chatbot & recommendation engine
│   └── i18n.py               # English & Arabic UI and chatbot translations
├── frontend/                 # Streamlit Frontend application
│   └── app.py                # Interactive Bilingual Streamlit Dashboard
├── models/                   # Trained AI models & evaluation benchmark metrics
│   ├── README.md             # Documentation of saved model files
│   ├── metrics.json          # Saved evaluation metrics (MAE, RMSE, R2, Accuracy, F1)
│   ├── forecaster.joblib     # Saved Random Forest model (auto-generated on run)
│   └── sentiment.joblib      # Saved TF-IDF + Logistic Regression pipeline (auto-generated on run)
├── notebook/                 # Step-by-step ML workflow & sequence
│   └── main.ipynb            # Complete Jupyter Notebook showing EDA, training, evaluation & demo
├── .gitignore                # Files excluded from GitHub (venv/, raw datasets, __pycache__/, *.joblib)
├── requirements.txt          # Required Python packages and library versions
└── README.md                 # Project documentation and usage guide
```

> **Note on Datasets & Virtual Environment (`venv`):**
> - **No External Dataset Download Required:** You do **not** need to manually download external CSV datasets to run this project. `backend/data_prep.py` includes a built-in realistic store transaction and bilingual customer review generator so the project runs immediately out of the box. You can also upload any custom Sales `.csv`/`.xlsx` or Reviews `.csv` directly from the Streamlit sidebar!
> - **`venv/` & Raw Data Ignored in Git:** Local virtual environments (`venv/`) and external dataset files are listed in `.gitignore` to keep the repository lightweight and clean.

---

## 🚀 How to Use and Try the Project

Follow these step-by-step instructions to set up, run, and test the project on your computer:

### 1. Open the Project Folder
Extract the project zip file and open a terminal (Command Prompt, PowerShell, or VS Code Terminal) inside the `ai_business_assistant` directory:
```bash
cd ai_business_assistant
```

### 2. Create and Activate a Virtual Environment (`venv`)
Creating a virtual environment keeps all project libraries isolated:

- **On Windows (PowerShell / CMD):**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Required Libraries (`requirements.txt`)
Install all needed packages with a single command:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Frontend Dashboard (Main App)
Launch the interactive web dashboard:
```bash
streamlit run frontend/app.py
```
- Your browser will automatically open at **`http://localhost:8501`**.
- The app will automatically load the built-in store dataset, train and save the models (`models/forecaster.joblib` and `models/sentiment.joblib`), and display all 5 interactive tabs.

### 5. Run the FastAPI Backend Server (Optional REST API Mode)
If you want to run and test the standalone FastAPI backend server:
```bash
uvicorn backend.main:app --reload --port 8000
```
- Open **`http://127.0.0.1:8000`** to verify the API status.
- Open **`http://127.0.0.1:8000/docs`** to explore and test all endpoints interactively via the **Swagger UI** (`/api/kpis`, `/api/forecast`, `/api/sentiment`, `/api/sentiment/predict`, `/api/chat`, `/api/recommendations`).

### 6. Explore the Jupyter Notebook (`notebook/main.ipynb`)
To review the step-by-step Machine Learning sequence, open `notebook/main.ipynb` in **VS Code**, **Jupyter Notebook/Lab**, or upload it to **Google Colab**:
```bash
jupyter notebook notebook/main.ipynb
```

---

## 🧪 What to Try Inside the Dashboard

Once the Streamlit app is open at `http://localhost:8501`, try these interactive features:

1. **Switch Languages (English / العربية):**
   - Use the **Language / اللغة** toggle in the left sidebar to switch the entire dashboard, charts, chatbot, and recommendations between **English** and **Arabic (RTL)**.
2. **Explore Sales Analytics & Forecast Horizon:**
   - In **📊 Sales Analytics**, hover over the monthly revenue curve, top/bottom products, and country breakdown.
   - In **📈 Sales Forecasting**, move the **Forecast Horizon** slider between **14 and 90 days** to see how the Random Forest model projects future revenue and which lag/rolling features matter most.
3. **Test Live Customer Sentiment (Tab 3 — 💬 Sentiment Analysis):**
   - Scroll to **Test Live Sentiment Analysis** and enter any review in English or Arabic, then click **Analyze Sentiment**:
     - *Positive example (EN):* `"The build quality is great and delivery was super fast!"`
     - *Negative example (EN):* `"Late delivery took over two weeks and the item arrived damaged."`
     - *Positive example (AR):* `"جودة ممتازة جداً والتوصيل سريع والخدمة رائعة"`
     - *Negative example (AR):* `"التوصيل متأخر جداً والجودة سيئة والخامة ضعيفة"`
4. **Chat with the AI Business Assistant (Tab 4 — 🤖 Business Chatbot):**
   - Click any of the **Quick Questions** buttons at the top of the chat tab, or type questions like:
     - `"What are the top selling products?"` / `"إيه أكتر المنتجات مبيعاً؟"`
     - `"What are the lowest selling products?"` / `"إيه أقل المنتجات مبيعاً؟"`
     - `"What is our total revenue and average order value?"` / `"كم إجمالي المبيعات ومتوسط الطلب؟"`
     - `"Will sales increase or decrease in the next 30 days?"` / `"هل المبيعات متوقع تزيد ولا تقل؟"`
     - `"What are customers complaining about the most?"` / `"العملاء بيشتكوا من إيه؟"`
     - `"Give me recommendations to boost sales"` / `"أعطني توصيات لزيادة المبيعات"`
5. **Upload Your Own Store Data (Optional):**
   - Use the sidebar file uploaders to upload your own Sales `.csv`/`.xlsx` (`InvoiceNo, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`) or Reviews `.csv` (`review, sentiment`) to dynamically retrain the models on your custom dataset.
