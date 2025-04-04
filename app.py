import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from model import ReceiptForecaster

app = Flask(__name__)

forecaster = None
monthly_predictions = None
daily_predictions = None
prediction_dates = None

def load_model_and_data():
    global forecaster, monthly_predictions, daily_predictions, prediction_dates
    
    # Initialize forecaster
    forecaster = ReceiptForecaster()
    
    # Load data
    data_path = os.path.join('data', 'receipts_2021.csv')
    data = forecaster.load_data(data_path)
    
    # Since load_model is not available, create and train a new model
    forecaster.create_model(
        hidden_dim=64,
        num_layers=2
    )
    
    # Train the model
    print("Training a new model...")
    forecaster.train(
        num_epochs=30,  # Enough epochs for reasonable predictions
        verbose=True
    )
    
    # Generate predictions for 2022
    prediction_dates, daily_predictions, monthly_predictions = forecaster.predict_future(
        start_date='2022-01-01',
        num_days=365
    )
    
    return {
        'daily': list(zip(prediction_dates, daily_predictions)),
        'monthly': sorted(monthly_predictions.items())
    }

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    """API endpoint to get predictions data"""
    global monthly_predictions, daily_predictions, prediction_dates
    
    if not monthly_predictions:
        predictions = load_model_and_data()
    else:
        predictions = {
            'daily': list(zip(prediction_dates, daily_predictions)),
            'monthly': sorted(monthly_predictions.items())
        }
    
    monthly_data = []
    for month, count in predictions['monthly']:
        monthly_data.append({
            'month': month,
            'count': int(count)
        })
    
    daily_data = []
    for date, count in predictions['daily']:
        daily_data.append({
            'date': date,
            'count': int(count)
        })
    
    total_annual = sum(count for _, count in predictions['monthly'])
    avg_monthly = total_annual / 12
    max_month = max(predictions['monthly'], key=lambda x: x[1])
    min_month = min(predictions['monthly'], key=lambda x: x[1])
    
    return jsonify({
        'monthly': monthly_data,
        'daily': daily_data[:365],  # First 3 months only
        'stats': {
            'total_annual': int(total_annual),
            'avg_monthly': int(avg_monthly),
            'max_month': {
                'month': max_month[0],
                'count': int(max_month[1])
            },
            'min_month': {
                'month': min_month[0],
                'count': int(min_month[1])
            }
        }
    })

@app.route('/api/rerun_model', methods=['POST'])
def rerun_model():
    """API endpoint to rerun the model with different parameters"""
    global forecaster, monthly_predictions, daily_predictions, prediction_dates
    
    params = request.get_json()
    
    seq_length = int(params.get('seq_length', 30))
    hidden_dim = int(params.get('hidden_dim', 64))
    num_layers = int(params.get('num_layers', 2))
    
    forecaster = ReceiptForecaster()
    
    data_path = os.path.join('data', 'receipts_2021.csv')
    data = forecaster.load_data(
        data_path,
        seq_length=seq_length,
        target_horizon=30
    )
    
    forecaster.create_model(
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    forecaster.train(
        num_epochs=20,  # Quick training for web interface
        verbose=False
    )
    
    prediction_dates, daily_predictions, monthly_predictions = forecaster.predict_future(
        start_date='2022-01-01',
        num_days=365
    )
    
    return get_predictions()

if __name__ == '__main__':
    load_model_and_data()
    app.run(host='0.0.0.0', port=5000, debug=False)