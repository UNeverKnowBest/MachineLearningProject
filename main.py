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

class DataLoader:
    """Data loading and preprocessing class"""
    
    @staticmethod
    def load_data(data_file):
        """
        Load CSV data and split into features and labels.
        
        Args:
            data_file: Path to CSV file
            
        Returns:
            features: Feature matrix
            labels: Label vector
        """
        data = np.genfromtxt(data_file, delimiter=",", skip_header=1)
        labels = data[:, -1].astype(int)
        features = data[:, :-1]
        return features, labels
    
    @staticmethod
    def split_data(X, y, test_size=0.1, val_size=0.1, random_state=42):
        """
        Split data into training, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Label vector
            test_size: Proportion of test set
            val_size: Proportion of validation set
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test: Split feature sets
            y_train, y_val, y_test: Split label sets
        """
        # First, split out test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        # Then, split train+validation set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class ForestDNN(nn.Module):
    """
    Deep Neural Network that uses random forest predictions (transformed via one-hot encoding)
    as input. This integrated model is named ForestDNN.
    """
    def __init__(self, input_features, n_hidden_1=256, n_hidden_2=128, n_hidden_3=64, 
                 n_classes=2, dropout_prob=0.3):
        """
        Initialize the ForestDNN model.
        
        Args:
            input_features: Number of input features
            n_hidden_1: Number of neurons in first hidden layer
            n_hidden_2: Number of neurons in second hidden layer
            n_hidden_3: Number of neurons in third hidden layer
            n_classes: Number of output classes
            dropout_prob: Dropout probability
        """
        super(ForestDNN, self).__init__()
        self.layer1 = nn.Linear(input_features, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.out = nn.Linear(n_hidden_3, n_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout(x)
        return self.out(x)


class RandomForestTransformer:
    """Random Forest feature transformer"""
    
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1):
        """
        Initialize the Random Forest transformer.
        
        Args:
            n_estimators: Number of trees
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=n_jobs
        )
        
    def fit(self, X, y):
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Label vector
        """
        self.rf.fit(X, y)
        
    def transform(self, X):
        """
        Transform Random Forest tree predictions into one-hot encoded features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed features
        """
        # Get predictions from each tree (shape: [num_samples, n_trees])
        tree_preds = np.array([tree.predict(X) for tree in self.rf.estimators_]).T
        # One-hot encode: 0 -> [1, 0], 1 -> [0, 1]
        onehot = np.stack([1 - tree_preds, tree_preds], axis=2)
        # Reshape to a feature vector for each sample
        transformed_features = onehot.reshape(X.shape[0], -1).astype(np.float32)
        return transformed_features


class ModelTrainer:
    """Model training class"""
    
    def __init__(self, device=None):
        """
        Initialize the model trainer.
        
        Args:
            device: Computation device (CPU/GPU)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
    
    def train_dnn(self, model, optimizer, X_train, y_train, X_val, y_val, 
                  batch_size=8, epochs=200):
        """
        Train the DNN model using mini-batch gradient descent.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size
            epochs: Number of training epochs
            
        Returns:
            loss_history: History of loss values
        """
        num_samples = X_train.shape[0]
        total_batches = num_samples // batch_size
        loss_history = []

        for epoch in range(epochs):
            model.train()
            permutation = np.random.permutation(num_samples)
            X_train_shuffled = torch.from_numpy(X_train[permutation]).float().to(self.device)
            y_train_shuffled = torch.from_numpy(y_train[permutation]).long().to(self.device)
            
            epoch_loss = 0.0
            y_true_epoch, y_pred_epoch = [], []
            
            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                xb = X_train_shuffled[start_idx:end_idx]
                yb = y_train_shuffled[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = model(xb)
                loss = self.criterion(outputs, yb)
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
                X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
                y_val_tensor = torch.from_numpy(y_val).long().to(self.device)
                val_outputs = model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
            
            print(f"Epoch {epoch+1:03d}: Train Loss = {epoch_loss:.6f}, "
                  f"Train F1 = {f1_epoch:.4f}, Val Loss = {val_loss:.6f}")
            
        return loss_history


class ModelEvaluator:
    """Model evaluation class"""
    
    def __init__(self, device=None):
        """
        Initialize the model evaluator.
        
        Args:
            device: Computation device (CPU/GPU)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate_dnn(self, model, X_test, y_test):
        """
        Evaluate the DNN model.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            metrics_dict: Dictionary of evaluation metrics
            predictions: Predicted classes
            probabilities: Predicted probabilities
        """
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).float().to(self.device)
            outputs = model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            probabilities = outputs[:, 1].cpu().numpy()
            
            accuracy = metrics.accuracy_score(y_test, predictions)
            auc_score = roc_auc_score(y_test, probabilities)
            
            metrics_dict = {
                'accuracy': accuracy,
                'auc': auc_score,
                'classification_report': classification_report(y_test, predictions)
            }
            
        return metrics_dict, predictions, probabilities
    
    def evaluate_rf(self, model, X_test, y_test):
        """
        Evaluate the Random Forest model.
        
        Args:
            model: Random Forest model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            metrics_dict: Dictionary of evaluation metrics
            predictions: Predicted classes
            probabilities: Predicted probabilities
        """
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        accuracy = metrics.accuracy_score(y_test, predictions)
        auc_score = roc_auc_score(y_test, probabilities)
        
        metrics_dict = {
            'accuracy': accuracy,
            'auc': auc_score,
            'classification_report': classification_report(y_test, predictions)
        }
        
        return metrics_dict, predictions, probabilities


class Visualizer:
    """Visualization class"""
    
    @staticmethod
    def plot_loss_curve(loss_history, title="Training Loss Trend"):
        """
        Plot the training loss curve.
        
        Args:
            loss_history: History of loss values
            title: Chart title
        """
        plt.figure()
        plt.plot(range(len(loss_history)), loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.close()
    
    @staticmethod
    def plot_roc_curves(y_true, model_probs_dict, title="ROC Curve Comparison"):
        """
        Plot ROC curves for multiple models.
        
        Args:
            y_true: True labels
            model_probs_dict: Dictionary mapping model names to predicted probabilities
            title: Chart title
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, probs in model_probs_dict.items():
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.close()


def main():
    """Main function"""
    data_file = "dataset.csv"
    start_time = time.time()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and split data
    data_loader = DataLoader()
    X, y = data_loader.load_data(data_file)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
    
    print(f"Dataset shapes: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    
    # 1. Train full RF model for transformation
    print("\n[Step 1] Training Random Forest for feature transformation...")
    transformer = RandomForestTransformer(n_estimators=100, random_state=42)
    transformer.fit(X_train, y_train)
    
    # 2. Train RF baseline model
    print("\n[Step 2] Training Random Forest baseline model...")
    rf_baseline = RandomForestClassifier(n_estimators=5, n_jobs=-1, random_state=43)
    rf_baseline.fit(X_train, y_train)
    
    # Evaluate RF baseline model
    evaluator = ModelEvaluator(device)
    rf_metrics, rf_preds, rf_probs = evaluator.evaluate_rf(rf_baseline, X_test, y_test)
    
    print(f"[INFO] RF Baseline Test Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"[INFO] RF Baseline Test AUC: {rf_metrics['auc']:.4f}")
    print("\nRF Baseline Classification Report:")
    print(rf_metrics['classification_report'])
    
    # 3. Transform data for ForestDNN
    print("\n[Step 3] Transforming data using Random Forest...")
    X_train_transformed = transformer.transform(X_train)
    X_val_transformed = transformer.transform(X_val)
    X_test_transformed = transformer.transform(X_test)
    
    # 4. Train ForestDNN
    print("\n[Step 4] Training ForestDNN (DNN with RF integration)...")
    n_trees = 100
    input_dim_fdnn = n_trees * 2  # each tree produces 2 one-hot features
    forest_dnn = ForestDNN(
        input_features=input_dim_fdnn, 
        n_hidden_1=256, 
        n_hidden_2=128, 
        n_hidden_3=64,
        dropout_prob=0.3
    ).to(device)
    
    optimizer_fdnn = optim.Adam(forest_dnn.parameters(), lr=1e-6, weight_decay=1e-4)
    trainer = ModelTrainer(device)
    
    loss_history_fdnn = trainer.train_dnn(
        forest_dnn, 
        optimizer_fdnn, 
        X_train_transformed, 
        y_train, 
        X_val_transformed, 
        y_val, 
        batch_size=8, 
        epochs=110
    )
    
    # Visualize ForestDNN loss
    visualizer = Visualizer()
    visualizer.plot_loss_curve(loss_history_fdnn, title="ForestDNN Training Loss Trend")
    
    # Evaluate ForestDNN
    fdnn_metrics, fdnn_preds, fdnn_probs = evaluator.evaluate_dnn(
        forest_dnn, X_test_transformed, y_test
    )
    
    print(f"ForestDNN Test Accuracy: {fdnn_metrics['accuracy']:.4f}")
    print(f"ForestDNN Test AUC: {fdnn_metrics['auc']:.4f}")
    print("\nForestDNN Classification Report:")
    print(fdnn_metrics['classification_report'])
    
    # 5. Train Raw DNN (on original features)
    print("\n[Step 5] Training Raw DNN (on original features)...")
    input_dim_raw = X_train.shape[1]
    raw_dnn = ForestDNN(
        input_features=input_dim_raw, 
        n_hidden_1=256, 
        n_hidden_2=128, 
        n_hidden_3=64,
        dropout_prob=0.3
    ).to(device)
    
    optimizer_raw = optim.Adam(raw_dnn.parameters(), lr=1e-5, weight_decay=1e-4)
    
    loss_history_raw = trainer.train_dnn(
        raw_dnn, 
        optimizer_raw, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        batch_size=8, 
        epochs=200
    )
    
    # Visualize Raw DNN loss
    visualizer.plot_loss_curve(loss_history_raw, title="Raw DNN Training Loss Trend")
    
    # Evaluate Raw DNN
    raw_metrics, raw_preds, raw_probs = evaluator.evaluate_dnn(raw_dnn, X_test, y_test)
    
    print(f"[INFO] Raw DNN Test Accuracy: {raw_metrics['accuracy']:.4f}")
    print(f"[INFO] Raw DNN Test AUC: {raw_metrics['auc']:.4f}")
    print("\nRaw DNN Classification Report:")
    print(raw_metrics['classification_report'])
    
    # 6. Final Model Comparison
    print("\n=== Final Model Comparison ===")
    print(f"ForestDNN (DNN with RF integration) - Accuracy: {fdnn_metrics['accuracy']:.4f}, AUC: {fdnn_metrics['auc']:.4f}")
    print(f"RF Baseline                         - Accuracy: {rf_metrics['accuracy']:.4f}, AUC: {rf_metrics['auc']:.4f}")
    print(f"Raw DNN                             - Accuracy: {raw_metrics['accuracy']:.4f}, AUC: {raw_metrics['auc']:.4f}")
    
    # Plot ROC curves comparison
    model_probs_dict = {
        'ForestDNN': fdnn_probs,
        'RF Baseline': rf_probs,
        'Raw DNN': raw_probs
    }
    
    visualizer.plot_roc_curves(y_test, model_probs_dict, title="ROC Curve Comparison")
    
    # Calculate runtime
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()