#  Stock Price Prediction Web App

A Machine Learning-based web application that predicts future stock prices using historical data. Built with **Streamlit**, **Scikit-learn**, and **Python**, this project allows users to upload stock datasets and visualize predictions interactively.

---

##  Features

*  Upload your own stock CSV dataset
*  Automatic data preprocessing
*  Machine Learning model (Linear Regression)
*  Future price prediction (user-defined days)
*  Dynamic graph visualization
*  Trend detection (UP/DOWN)
*  Buy/Sell recommendation system

---

##  Tech Stack

* Python 3.11
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---



##  Installation & Setup

###  Clone the repository

```bash
git clone https://github.com/priyalkhandelwal0608/Stock-price-prediction.git
cd Stock-price-prediction
```

---

###  Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

###  Install dependencies

```bash
pip install -r requirements.txt
```

---

###  Run the application

```bash
streamlit run app.py
```

---

###  Open in browser

```
http://localhost:8501
```

---

##  How It Works

1. User uploads a CSV file containing stock prices
2. The app extracts the **'Close' price**
3. Data is scaled using MinMaxScaler
4. A **Linear Regression model** is trained
5. Future prices are predicted based on previous trends
6. Results are visualized using graphs

---



##  Output

* Predicted future prices
* Trend (UP 📈 / DOWN 📉)
* Buy/Sell recommendation
* Interactive graph

---

##  Use Cases

* Stock market trend analysis
* Beginner-friendly ML project
* Data visualization practice
* Resume/portfolio project

---

