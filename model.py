import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class TimeSeriesDataset:
    def __init__(self, data, seq_length=30, target_horizon=30):
        self.data = data
        self.seq_length = seq_length
        self.target_horizon = target_horizon
        self.receipt_count = data['Receipt_Count'].values
        self.min_val = np.min(self.receipt_count)
        self.max_val = np.max(self.receipt_count)
        self.normalized_data = (self.receipt_count - self.min_val) / (self.max_val - self.min_val)
        self.dates = pd.to_datetime(data['Date'])
        self.day_of_week = np.array([date.weekday() for date in self.dates])
        self.day_of_month = np.array([date.day for date in self.dates])
        self.month = np.array([date.month for date in self.dates])
        self.day_of_week_onehot = np.zeros((len(self.day_of_week), 7))
        self.day_of_week_onehot[np.arange(len(self.day_of_week)), self.day_of_week] = 1
        self.month_onehot = np.zeros((len(self.month), 12))
        self.month_onehot[np.arange(len(self.month)), self.month - 1] = 1
        self._create_sequences()

    def _create_sequences(self):
        self.X = []
        self.y = []
        for i in range(len(self.normalized_data) - self.seq_length - self.target_horizon + 1):
            receipt_seq = self.normalized_data[i:i+self.seq_length]
            dow_seq = self.day_of_week_onehot[i:i+self.seq_length]
            month_seq = self.month_onehot[i:i+self.seq_length]
            features = np.column_stack([receipt_seq.reshape(-1, 1), dow_seq, month_seq])
            target_idx = i + self.seq_length
            target = self.normalized_data[target_idx:target_idx+self.target_horizon]
            if len(target) == self.target_horizon:
                self.X.append(features)
                self.y.append(target)
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

    def get_feature_dim(self):
        return self.X.shape[2]

    def denormalize(self, normalized_value):
        return normalized_value * (self.max_val - self.min_val) + self.min_val

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output

class ReceiptForecaster:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self, csv_path, seq_length=30, target_horizon=30):
        data = pd.read_csv(csv_path)
        if '# Date' in data.columns:
            data = data.rename(columns={'# Date': 'Date'})
        if 'Date' not in data.columns:
            first_col = data.columns[0]
            if 'date' in first_col.lower():
                data = data.rename(columns={first_col: 'Date'})
        self.dataset = TimeSeriesDataset(data, seq_length, target_horizon)
        return data

    def create_model(self, hidden_dim=64, num_layers=2, dropout=0.2):
        input_dim = self.dataset.get_feature_dim()
        output_dim = self.dataset.target_horizon
        self.model = LSTMForecaster(input_dim, hidden_dim, num_layers, output_dim, dropout)
        self.model.to(self.device)

    def train(self, batch_size=16, num_epochs=50, learning_rate=0.001, verbose=True):
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        epoch_losses = []
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            if verbose and (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
        return epoch_losses

    def predict_future(self, start_date, num_days=365):
        self.model.eval()
        last_seq = self.dataset.X[-1]
        last_seq_tensor = torch.FloatTensor(last_seq).unsqueeze(0).to(self.device)
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]
        predictions = []
        remaining_days = num_days
        current_input = last_seq_tensor
        while remaining_days > 0:
            with torch.no_grad():
                pred = self.model(current_input)
                pred = pred.cpu().numpy()[0]
            denorm_pred = self.dataset.denormalize(pred)
            pred_days = min(remaining_days, len(pred))
            predictions.extend(denorm_pred[:pred_days].tolist())
            remaining_days -= pred_days
            if remaining_days <= 0:
                break
            new_seq = np.copy(current_input.cpu().numpy()[0])
            new_seq[1:, 0] = new_seq[:-1, 0]
            new_seq[0, 0] = pred[-1]
            current_input = torch.FloatTensor(new_seq).unsqueeze(0).to(self.device)
        monthly_data = {}
        for date, count in zip(dates, predictions):
            year_month = date[:7]
            if year_month not in monthly_data:
                monthly_data[year_month] = 0
            monthly_data[year_month] += int(count)
        return dates, predictions, monthly_data

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'min_val': self.dataset.min_val,
            'max_val': self.dataset.max_val
        }, path)

def load_model(self, path, input_dim, hidden_dim=64, num_layers=2, output_dim=30):
    """Load model from file"""
    # Add weights_only=False to fix PyTorch 2.6 compatibility issue
    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
    
    self.model = LSTMForecaster(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim
    )
    
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.model.to(self.device)
    
    # Update dataset normalization parameters if dataset exists
    if self.dataset:
        self.dataset.min_val = checkpoint['min_val']
        self.dataset.max_val = checkpoint['max_val']
    
    self.model.eval()
    
if __name__ == "__main__":
    forecaster = ReceiptForecaster()
    data = forecaster.load_data('data/receipts_2021.csv')
    forecaster.create_model()
    losses = forecaster.train(num_epochs=30)
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    start_date = '2022-01-01'
    dates, daily_predictions, monthly_predictions = forecaster.predict_future(start_date)
    print("\nPredicted monthly receipt counts for 2022:")
    for month, count in sorted(monthly_predictions.items()):
        print(f"{month}: {count:,}")
    forecaster.save_model('models/receipt_forecaster.pth')
