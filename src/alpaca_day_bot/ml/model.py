import torch
import torch.nn as nn
import torch.nn.functional as F

class OrderBookCNN(nn.Module):
    """
    Extracts spatial features from the 40 order book features (10 levels of P/S).
    """
    def __init__(self, input_dim=40, output_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        # x shape: (Batch, SeqLen, 40)
        # Reshape for Conv1d: (Batch, 40, SeqLen)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Global pooling or just take the last state? 
        # For a transformer, we want the sequence preserved.
        x = x.transpose(1, 2) # (Batch, SeqLen, 64)
        return x

class TemporalTransformer(nn.Module):
    """
    Processes the sequence of spatial features using Multi-Head Attention.
    """
    def __init__(self, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (Batch, SeqLen, d_model)
        return self.transformer(x)

class AetherisDeepHybrid(nn.Module):
    """
    The Institutional CNN-Transformer Hybrid.
    """
    def __init__(self, book_dim=40, seq_len=50):
        super().__init__()
        self.cnn = OrderBookCNN(input_dim=book_dim, output_dim=128)
        self.transformer = TemporalTransformer(d_model=128, nhead=8, num_layers=3)
        
        # Prediction Head: Probability of [UP/TP, DOWN/SL, STATIONARY/TIMEOUT]
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) # 3 Triple-Barrier outcomes
        )

    def forward(self, x_book):
        # 1. Spatial Feature Extraction (CNN)
        spatial_feats = self.cnn(x_book)
        
        # 2. Temporal Feature Extraction (Transformer)
        temporal_feats = self.transformer(spatial_feats)
        
        # 3. Decision (Take the last timestamp's hidden state)
        last_state = temporal_feats[:, -1, :]
        logits = self.classifier(last_state)
        
        return F.softmax(logits, dim=1)

def save_model_metadata(model, path):
    """
    Saves model weights and necessary metadata for inference.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'arch': 'cnn_transformer_v1',
        'features': 'order_book_10_levels'
    }, path)
