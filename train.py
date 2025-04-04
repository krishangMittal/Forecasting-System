import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import ReceiptForecaster

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train receipt forecasting model')
    parser.add_argument('--data', type=str, default='data/receipts_2021.csv',
                        help='Path to input CSV file (default: data/receipts_2021.csv)')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save the model (default: models)')
    parser.add_argument('--plots_dir', type=str, default='plots',
                        help='Directory to save plots (default: plots)')
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Input sequence length in days (default: 30)')
    parser.add_argument('--target_horizon', type=int, default=30,
                        help='Prediction horizon in days (default: 30)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension of LSTM (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability (default: 0.2)')
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Initialize forecaster
    print(f"Initializing ReceiptForecaster...")
    forecaster = ReceiptForecaster()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = forecaster.load_data(
        args.data,
        seq_length=args.seq_length,
        target_horizon=args.target_horizon
    )
    
    # Print data summary
    print(f"Loaded {len(data)} days of data")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Receipt count range: {data['Receipt_Count'].min():,} to {data['Receipt_Count'].max():,}")
    print(f"Average daily receipts: {data['Receipt_Count'].mean():,.0f}")
    
    # Create model
    print(f"Creating model with {args.num_layers} LSTM layers, {args.hidden_dim} hidden units...")
    forecaster.create_model(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Train model
    print(f"Training model for {args.epochs} epochs...")
    losses = forecaster.train(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_plot_path = os.path.join(args.plots_dir, 'training_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training loss plot saved to {loss_plot_path}")
    
    # Predict 2022 values
    print("Generating predictions for 2022...")
    start_date = '2022-01-01'
    dates, daily_predictions, monthly_predictions = forecaster.predict_future(
        start_date=start_date,
        num_days=365  # Full year 2022
    )
    
    # Plot daily predictions
    plt.figure(figsize=(15, 6))
    plt.plot(dates[:90], daily_predictions[:90])  # First 3 months
    plt.title('Daily Receipt Count Predictions (First 3 months of 2022)')
    plt.xlabel('Date')
    plt.ylabel('Receipt Count')
    plt.grid(True)
    plt.xticks(rotation=45)
    day_plot_path = os.path.join(args.plots_dir, 'daily_predictions.png')
    plt.tight_layout()
    plt.savefig(day_plot_path)
    plt.close()
    print(f"Daily predictions plot saved to {day_plot_path}")
    
    # Plot monthly predictions
    months = []
    counts = []
    for month, count in sorted(monthly_predictions.items()):
        months.append(month)
        counts.append(count)
    
    plt.figure(figsize=(12, 6))
    plt.bar(months, counts)
    plt.title('Monthly Receipt Count Predictions for 2022')
    plt.xlabel('Month')
    plt.ylabel('Receipt Count')
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    month_plot_path = os.path.join(args.plots_dir, 'monthly_predictions.png')
    plt.savefig(month_plot_path)
    plt.close()
    print(f"Monthly predictions plot saved to {month_plot_path}")
    
    # Save monthly predictions to CSV
    monthly_df = pd.DataFrame({
        'Month': months,
        'Predicted_Receipt_Count': counts
    })
    monthly_csv_path = os.path.join(args.plots_dir, 'monthly_predictions.csv')
    monthly_df.to_csv(monthly_csv_path, index=False)
    print(f"Monthly predictions saved to {monthly_csv_path}")
    
    # Save model
    model_path = os.path.join(args.model_dir, 'receipt_forecaster.pth')
    forecaster.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Print monthly predictions
    print("\nPredicted monthly receipt counts for 2022:")
    for month, count in sorted(monthly_predictions.items()):
        print(f"{month}: {count:,}")

if __name__ == "__main__":
    main()