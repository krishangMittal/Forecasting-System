import os
import argparse
import subprocess
import pandas as pd

def main():
    """Main function to run the entire pipeline"""
    parser = argparse.ArgumentParser(description='Run the receipt forecasting pipeline')
    parser.add_argument('--data', type=str, default='data/receipts_2021.csv',
                        help='Path to input CSV file (default: data/receipts_2021.csv)')
    parser.add_argument('--mode', type=str, choices=['train', 'serve', 'full'], default='full',
                        help='Pipeline mode: train, serve, or full (default: full)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for web application (default: 5000)')
    
    args = parser.parse_args()
    
    for directory in ['data', 'models', 'plots', 'templates']:
        os.makedirs(directory, exist_ok=True)
    
    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found.")
        
        if args.data == 'data/receipts_2021.csv' and os.path.exists('paste.txt'):
            print("Found 'receipts_2021.csv'. Converting to CSV format...")
            
            with open('receipts_2021.csv', 'r') as f:
                content = f.readlines()
            
            with open(args.data, 'w') as f:
                for line in content:
                    f.write(line)
            
            print(f"Created sample data file at '{args.data}'")
        else:
            print("Please provide a valid data file path.")
            return
    
    model_path = 'models/receipt_forecaster.pth'
    if not os.path.exists(model_path) and args.mode == 'serve':
        print(f"Model not found at '{model_path}'. Setting mode to 'full' to train the model first.")
        args.mode = 'full'
    
    if args.mode in ['train', 'full']:
        print("\n=== Training the forecasting model ===")
        train_cmd = [
            'python', 'train.py',
            '--data', args.data,
            '--epochs', str(args.epochs)
        ]
        subprocess.run(train_cmd)
    
    if args.mode in ['serve', 'full']:
        print("\n=== Starting the web application ===")
        print(f"The dashboard will be accessible at http://localhost:{args.port}")
        serve_cmd = [
            'python', 'app.py',
            '--port', str(args.port)
        ]
        subprocess.run(serve_cmd)

if __name__ == "__main__":
    main()