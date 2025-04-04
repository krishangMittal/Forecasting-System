# Receipt Forecasting System

This project provides a time series forecasting solution for predicting receipt volumes in 2022 based on historical 2021 data. The system uses a custom LSTM (Long Short-Term Memory) neural network built from scratch in PyTorch and provides an interactive web dashboard for visualization.

---

##  Overview

The Receipt Forecasting System processes daily receipt count data from 2021, trains a deep learning model to identify patterns and trends, and generates predictions for each month of 2022. The dashboard allows for parameter tuning and real-time model retraining.

---

##  Requirements

- Python 3.9+
- PyTorch
- Flask
- NumPy, Pandas
- Chart.js (via CDN)
- Font Awesome (via CDN)

---

##  Getting Started

### Option 1: Running Locally (Manual Setup)  
>  *If you want a quicker setup, consider using **Option 2 (Docker)** below.* 

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/receipt-forecasting.git
cd receipt-forecasting
```

#### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Prepare the Data
- Place your 2021 receipt data in the `data/` directory as `receipts_2021.csv`
- CSV Format: `Date`, `Receipt_Count`

#### 5. Run the Application
```bash
python app.py
```

#### 6. Access the Dashboard
Open your browser and go to:
```
http://localhost:5000
```

---

### Option 2: Using Docker

#### 1. Build the Docker Image
```bash
docker build -t receipt-forecaster .
```

#### 2. Run the Container
```bash
docker run -p 5000:5000 receipt-forecaster
```

#### 3. Access the Dashboard
Open your browser and go to:
```
http://localhost:5000
```

---

##  Using the Dashboard

###  View Forecasts
- The dashboard displays monthly and daily receipt predictions for 2022.

### ⚙️ Adjust Model Parameters
- **Input Sequence Length**: Number of historical days considered (7-365)
- **Hidden Dimension**: Size of LSTM hidden state (16-256)
- **LSTM Layers**: Number of stacked LSTM layers (1-4)

###  Retrain the Model
- After adjusting parameters, click **Retrain Model** to update the forecasts.

---

##  Project Structure
```
receipt-forecasting/
├── app.py               # Flask web application
├── model.py             # LSTM model logic
├── train.py             # Training script
├── data/
│   └── receipts_2021.csv  # Input data
├── templates/
│   └── index.html       # Dashboard UI
├── models/              # Saved model files
├── Dockerfile           # Docker configuration
└── requirements.txt     # Python dependencies
```

---

##  Model Architecture

- **Input Features**: Receipt counts, day of week, and month (one-hot encoded)
- **Architecture**: Multi-layer LSTM with configurable hidden layers
- **Training**: Sliding window with MSE loss function
- **Output**: Daily forecasts, aggregated into monthly totals

---

##  License
MIT License

---

##  Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## Acknowledgements
- PyTorch
- Flask
- Chart.js
- The Open Source Community
