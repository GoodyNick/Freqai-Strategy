import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LSTMTrainer:
    """
    Handles model training and optimization for the LSTM model.
    """
    def __init__(self, model, learning_rate=0.001, batch_size=64, epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

    def train(self, train_data, train_labels):
        dataset = TensorDataset(train_data, train_labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            self.scheduler.step(epoch_loss)
            print(f'Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}')
        
        print("Training complete.")
