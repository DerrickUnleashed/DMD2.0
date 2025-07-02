import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, jaccard_score, precision_recall_curve, 
                           matthews_corrcoef, cohen_kappa_score, log_loss, auc)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check for MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

def create_patient_similarity_graph(X, k=10, similarity_threshold=0.5):
    """
    Create a patient similarity graph based on gene expression patterns
    
    Args:
        X: Feature matrix (patients x genes)
        k: Number of nearest neighbors to connect
        similarity_threshold: Minimum similarity to create an edge
    
    Returns:
        edge_index: Graph edges in COO format
    """
    # Calculate cosine similarity between patients
    similarity_matrix = cosine_similarity(X)
    
    # Create edges based on k-nearest neighbors and similarity threshold
    edges = []
    n_patients = X.shape[0]
    
    for i in range(n_patients):
        # Get k most similar patients (excluding self)
        similarities = similarity_matrix[i]
        similar_indices = np.argsort(similarities)[::-1][1:k+1]  # Exclude self (index 0)
        
        for j in similar_indices:
            if similarities[j] > similarity_threshold:
                edges.append([i, j])
    
    # Convert to undirected graph
    edge_set = set()
    for edge in edges:
        edge_set.add((min(edge), max(edge)))
    
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    
    print(f"Created graph with {n_patients} nodes and {edge_index.shape[1]} edges")
    print(f"Average degree: {2 * edge_index.shape[1] / n_patients:.2f}")
    
    return edge_index

class GNNClassifier(nn.Module):
    """
    Graph Neural Network for patient classification based on gene expression
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.5):
        super(GNNClassifier, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolution layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling (if batch is provided for batch processing)
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x

def create_graph_data(X, y, edge_index, train_mask, test_mask):
    """Create PyTorch Geometric Data objects"""
    # Convert to tensors
    x = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    # Create data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    
    return data

def train_gnn(model, data, optimizer, criterion, device):
    """Train the GNN for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Calculate loss only on training nodes
    loss = criterion(out[data.train_mask].squeeze(), data.y[data.train_mask].float())
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate_gnn(model, data, device):
    """Evaluate the GNN"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        
        # Get predictions for test set
        test_logits = out[data.test_mask].squeeze()
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs > 0.5).long()
        
        # Convert to numpy for sklearn metrics
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = test_preds.cpu().numpy()
        y_prob = test_probs.cpu().numpy()
        
        return y_true, y_pred, y_prob

# Load and preprocess data
print("Loading data...")
df = pd.read_csv("DMD_combined_dataset.csv", index_col=False)

# Prepare features and labels
X = df.iloc[:, :-1]
X = X.drop(columns=["Unnamed: 0"], axis=1)
y = df.iloc[:, -1]

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("\nLabel distribution:\n", y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Combine for graph creation
X_combined = np.vstack([X_train_scaled, X_test_scaled])
y_combined = np.hstack([y_train, y_test])

print(f"\nCombined dataset shape: {X_combined.shape}")

# Create patient similarity graph
print("\nCreating patient similarity graph...")
edge_index = create_patient_similarity_graph(X_combined, k=15, similarity_threshold=0.3)

# Create train/test masks
n_train = len(X_train_scaled)
n_total = len(X_combined)

train_mask = torch.zeros(n_total, dtype=torch.bool)
test_mask = torch.zeros(n_total, dtype=torch.bool)

train_mask[:n_train] = True
test_mask[n_train:] = True

print(f"Train mask sum: {train_mask.sum()}")
print(f"Test mask sum: {test_mask.sum()}")

# Create graph data
data = create_graph_data(X_combined, y_combined, edge_index, train_mask, test_mask)
data = data.to(device)

print(f"\nGraph data created:")
print(f"  Nodes: {data.x.shape[0]}")
print(f"  Node features: {data.x.shape[1]}")
print(f"  Edges: {data.edge_index.shape[1]}")

# Initialize model
model = GNNClassifier(
    input_dim=data.x.shape[1],
    hidden_dim=128,
    num_layers=3,
    dropout=0.5
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.BCEWithLogitsLoss()

# Training loop
print("\nTraining GNN...")
model.train()
losses = []

for epoch in tqdm(range(200)):
    loss = train_gnn(model, data, optimizer, criterion, device)
    losses.append(loss)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluation
print("\nEvaluating model...")
y_true, y_pred, y_prob = evaluate_gnn(model, data, device)

# Calculate metrics
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["NORMAL", "DMD"])
cm = confusion_matrix(y_true, y_pred)
auc_score = roc_auc_score(y_true, y_prob)
jaccard = jaccard_score(y_true, y_pred)
precision, recall, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recall, precision)
mcc = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
loss_eval = log_loss(y_true, y_prob)

print("\n" + "="*50)
print("GNN Classification Results:")
print("="*50)
print(f"Accuracy: {acc:.4f}")
print(f"AUC-ROC Score: {auc_score:.4f}")
print(f"Jaccard Score: {jaccard:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Log Loss: {loss_eval:.4f}")

print(f"\nClassification Report:\n{report}")
print(f"Confusion Matrix:\n{cm}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Feature importance approximation using gradient-based method
print("\nCalculating feature importance...")
model.eval()
data.x.requires_grad_(True)

# Forward pass
out = model(data.x, data.edge_index)
test_out = out[data.test_mask]

# Calculate gradients
loss_importance = test_out.sum()
loss_importance.backward()

# Get feature importance as absolute gradients
feature_importance = torch.abs(data.x.grad).mean(dim=0).cpu().numpy()
feature_names = X.columns

print("\nTop 10 Most Important Features (by gradient magnitude):")
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10).to_string(index=False))