import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split

def load_data(data_file):
    """
    Load CSV data and split into features and labels.
    """
    data = np.genfromtxt(data_file, delimiter=",", skip_header=1)
    labels = data[:, -1].astype(int)
    features = data[:, :-1]
    return features, labels

class ForestDNN(nn.Module):
    """
    Deep Neural Network that uses random forest predictions (transformed via one-hot encoding)
    as input. This integrated model is named ForestDNN.
    """
    def __init__(self, input_features, n_hidden_1=256, n_hidden_2=128, n_hidden_3=64, 
                 n_classes=2, dropout_prob=0.3):
        super(ForestDNN, self).__init__()
        self.layer1 = nn.Linear(input_features, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.out = nn.Linear(n_hidden_3, n_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout(x)
        return self.out(x)

def transform_rf_predictions(rf, X):
    """
    Transforms random forest tree predictions into one-hot encoded features.
    """
    # Get predictions from each tree (shape: [num_samples, n_trees])
    tree_preds = np.array([tree.predict(X) for tree in rf.estimators_]).T
    # One-hot encode: 0 -> [1, 0], 1 -> [0, 1]
    onehot = np.stack([1 - tree_preds, tree_preds], axis=2)
    # Reshape to a feature vector for each sample
    transformed_features = onehot.reshape(X.shape[0], -1).astype(np.float32)
    return transformed_features

def train_dnn(model, optimizer, criterion, X_train, y_train, X_val, y_val, device, 
              batch_size=8, epochs=200):
    """
    Train the DNN model using mini-batch gradient descent.
    """
    num_samples = X_train.shape[0]
    total_batches = num_samples // batch_size
    loss_history = []

    for epoch in range(epochs):
        model.train()
        permutation = np.random.permutation(num_samples)
        X_train_shuffled = torch.from_numpy(X_train[permutation]).float().to(device)
        y_train_shuffled = torch.from_numpy(y_train[permutation]).long().to(device)
        
        epoch_loss = 0.0
        y_true_epoch, y_pred_epoch = [], []
        
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            xb = X_train_shuffled[start_idx:end_idx]
            yb = y_train_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item() / total_batches
            y_true_epoch.extend(yb.cpu().numpy())
            y_pred_epoch.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
        f1_epoch = f1_score(y_true_epoch, y_pred_epoch, average='weighted')
        loss_history.append(epoch_loss)
        
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val).float().to(device)
            y_val_tensor = torch.from_numpy(y_val).long().to(device)
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        print(f"Epoch {epoch+1:03d}: Train Loss = {epoch_loss:.6f}, "
              f"Train F1 = {f1_epoch:.4f}, Val Loss = {val_loss:.6f}")
        
    return loss_history

def plot_loss_curve(loss_history, title="Training Loss Trend"):
    """
    Plot the training loss over epochs.
    """
    plt.figure()
    plt.plot(range(len(loss_history)), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_roc_curves(y_true, forest_dnn_probs, rf_baseline_probs, raw_dnn_probs):
    """
    Plot ROC curves for ForestDNN, RF Baseline, and Raw DNN models.
    """
    fpr_fdnn, tpr_fdnn, _ = roc_curve(y_true, forest_dnn_probs)
    roc_auc_fdnn = auc(fpr_fdnn, tpr_fdnn)
    
    fpr_rf_baseline, tpr_rf_baseline, _ = roc_curve(y_true, rf_baseline_probs)
    roc_auc_rf_baseline = auc(fpr_rf_baseline, tpr_rf_baseline)
    
    fpr_raw, tpr_raw, _ = roc_curve(y_true, raw_dnn_probs)
    roc_auc_raw = auc(fpr_raw, tpr_raw)
    
    plt.figure()
    plt.plot(fpr_fdnn, tpr_fdnn, label=f'ForestDNN ROC (AUC = {roc_auc_fdnn:.4f})')
    plt.plot(fpr_rf_baseline, tpr_rf_baseline, label=f'RF Baseline ROC (AUC = {roc_auc_rf_baseline:.4f})')
    plt.plot(fpr_raw, tpr_raw, label=f'Raw DNN ROC (AUC = {roc_auc_raw:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.show()

def main():
    data_file = "dataset.csv"
    start_time = time.time()
    
    # Load data and split into train, validation, and test datasets
    X, y = load_data(data_file)
    # First, split out test set (10% of data)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    # Then, split train+validation set into training (90%) and validation (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )
    
    # Train full RF model on training data for transformation
    n_trees = 100
    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    # Train RF baseline model on training data
    rf_baseline = RandomForestClassifier(n_estimators=5, n_jobs=-1, random_state=43)
    rf_baseline.fit(X_train, y_train)
    
    # Evaluate RF baseline on test set
    rf_baseline_preds = rf_baseline.predict(X_test)
    acc_rf_baseline = metrics.accuracy_score(y_test, rf_baseline_preds)
    auc_rf_baseline = roc_auc_score(y_test, rf_baseline.predict_proba(X_test)[:, 1])
    print(f"[INFO] RF Baseline Test Accuracy: {acc_rf_baseline:.4f}")
    print(f"[INFO] RF Baseline Test AUC: {auc_rf_baseline:.4f}")
    print("\nRF Baseline Classification Report:")
    print(classification_report(y_test, rf_baseline_preds))
    
    # Transform data for ForestDNN using full RF model predictions
    X_train_transformed = transform_rf_predictions(rf, X_train)
    X_val_transformed = transform_rf_predictions(rf, X_val)
    X_test_transformed = transform_rf_predictions(rf, X_test)
    
    # -------------------------
    # Train ForestDNN (DNN with RF integration) on transformed training data
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim_fdnn = n_trees * 2  # each tree produces 2 one-hot features
    forest_dnn = ForestDNN(input_features=input_dim_fdnn, n_hidden_1=256, n_hidden_2=128, n_hidden_3=64,
                           dropout_prob=0.3).to(device)
    optimizer_fdnn = optim.Adam(forest_dnn.parameters(), lr=1e-6, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("\nTraining ForestDNN (DNN with RF integration)...")
    loss_history_fdnn = train_dnn(forest_dnn, optimizer_fdnn, criterion, 
                                  X_train_transformed, y_train, X_val_transformed, y_val, device, 
                                  batch_size=8, epochs=100)
    plot_loss_curve(loss_history_fdnn, title="ForestDNN Training Loss Trend")
    
    # Evaluate ForestDNN on test set (transformed data)
    forest_dnn.eval()
    with torch.no_grad():
        X_test_fdnn_tensor = torch.from_numpy(X_test_transformed).float().to(device)
        outputs_fdnn = forest_dnn(X_test_fdnn_tensor)
        forest_dnn_preds = torch.argmax(outputs_fdnn, dim=1).cpu().numpy()
        forest_dnn_accuracy = metrics.accuracy_score(y_test, forest_dnn_preds)
        forest_dnn_auc = roc_auc_score(y_test, outputs_fdnn[:, 1].cpu().numpy())
    
    print(f"ForestDNN Test Accuracy: {forest_dnn_accuracy:.4f}")
    print(f"ForestDNN Test AUC: {forest_dnn_auc:.4f}")
    print("\nForestDNN Classification Report:")
    print(classification_report(y_test, forest_dnn_preds))
    
    # -------------------------
    # Train Raw DNN on original features
    # -------------------------
    input_dim_raw = X_train.shape[1]
    raw_dnn = ForestDNN(input_features=input_dim_raw, n_hidden_1=256, n_hidden_2=128, n_hidden_3=64,
                        dropout_prob=0.3).to(device)
    optimizer_raw = optim.Adam(raw_dnn.parameters(), lr=1e-4, weight_decay=1e-4)
    
    print("\nTraining Raw DNN (on original features)...")
    loss_history_raw = train_dnn(raw_dnn, optimizer_raw, criterion, 
                                 X_train, y_train, X_val, y_val, device, 
                                 batch_size=8, epochs=100)
    plot_loss_curve(loss_history_raw, title="Raw DNN Training Loss Trend")
    
    # Evaluate Raw DNN on test set (original features)
    raw_dnn.eval()
    with torch.no_grad():
        X_test_raw_tensor = torch.from_numpy(X_test).float().to(device)
        outputs_raw = raw_dnn(X_test_raw_tensor)
        raw_dnn_preds = torch.argmax(outputs_raw, dim=1).cpu().numpy()
        raw_dnn_accuracy = metrics.accuracy_score(y_test, raw_dnn_preds)
        raw_dnn_auc = roc_auc_score(y_test, outputs_raw[:, 1].cpu().numpy())
    
    print(f"[INFO] Raw DNN Test Accuracy: {raw_dnn_accuracy:.4f}")
    print(f"[INFO] Raw DNN Test AUC: {raw_dnn_auc:.4f}")
    print("\nRaw DNN Classification Report:")
    print(classification_report(y_test, raw_dnn_preds))
    
    # -------------------------
    # Final Model Comparison
    # -------------------------
    print("\n=== Final Model Comparison ===")
    print(f"ForestDNN (DNN with RF integration) - Accuracy: {forest_dnn_accuracy:.4f}, AUC: {forest_dnn_auc:.4f}")
    print(f"RF Baseline                         - Accuracy: {acc_rf_baseline:.4f}, AUC: {auc_rf_baseline:.4f}")
    print(f"Raw DNN                             - Accuracy: {raw_dnn_accuracy:.4f}, AUC: {raw_dnn_auc:.4f}")
    
    # Get prediction probabilities for ROC curves
    forest_dnn_probs = outputs_fdnn[:, 1].cpu().numpy()
    rf_baseline_probs = rf_baseline.predict_proba(X_test)[:, 1]
    raw_dnn_probs = outputs_raw[:, 1].cpu().numpy()
    
    plot_roc_curves(y_test, forest_dnn_probs, rf_baseline_probs, raw_dnn_probs)
if __name__ == "__main__":
    main()