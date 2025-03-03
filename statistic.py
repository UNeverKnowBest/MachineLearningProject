import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score
from main import DataLoader, RandomForestTransformer, ModelTrainer, ModelEvaluator, ForestDNN
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.optim as optim

# Assuming we have all the classes defined in the previous code
# We're extending the code to add statistical analysis functionality

class StatisticalAnalyzer:
    """Statistical analysis class for comparing model performance significance"""
    
    @staticmethod
    def bootstrap_performance(y_true, y_pred_probs_dict, metric_func, n_bootstrap=1000, 
                            random_state=42):
        """
        Use bootstrap method to evaluate model performance confidence intervals and significance
        
        Args:
            y_true: Ground truth labels
            y_pred_probs_dict: Dictionary mapping model names to prediction probabilities
            metric_func: Performance metric function like accuracy_score or roc_auc_score
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed
            
        Returns:
            bootstrap_results: Dictionary containing bootstrap performance values for each model
            ci_results: Dictionary containing 95% confidence intervals for each model
        """
        np.random.seed(random_state)
        n_samples = len(y_true)
        bootstrap_results = {model: [] for model in y_pred_probs_dict}
        
        for i in range(n_bootstrap):
            # Generate bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_bootstrap = y_true[indices]
            
            # Calculate performance for each model on the bootstrap sample
            for model, probs in y_pred_probs_dict.items():
                probs_bootstrap = probs[indices]
                
                # Different calculation methods for different metrics
                if metric_func == accuracy_score:
                    y_pred = (probs_bootstrap > 0.5).astype(int)
                    score = metric_func(y_bootstrap, y_pred)
                else:  # Assuming roc_auc_score
                    score = metric_func(y_bootstrap, probs_bootstrap)
                    
                bootstrap_results[model].append(score)
        
        # Calculate 95% confidence intervals
        ci_results = {}
        for model, scores in bootstrap_results.items():
            ci_results[model] = (
                np.percentile(scores, 2.5),  # Lower bound
                np.percentile(scores, 97.5)  # Upper bound
            )
            
        return bootstrap_results, ci_results
    
    @staticmethod
    def paired_bootstrap_test(bootstrap_results, model_a, model_b, alpha=0.05):
        """
        Perform paired bootstrap test for model comparison significance
        
        Args:
            bootstrap_results: Bootstrap sampling results
            model_a: First model name
            model_b: Second model name
            alpha: Significance level
            
        Returns:
            p_value: Significance p-value
            is_significant: Whether the difference is significant
            better_model: Which model performs better
        """
        scores_a = np.array(bootstrap_results[model_a])
        scores_b = np.array(bootstrap_results[model_b])
        
        # Calculate performance differences
        diffs = scores_a - scores_b
        
        # Calculate p-value (two-sided test)
        p_value = np.sum(diffs <= 0) / len(diffs) if np.mean(diffs) > 0 else np.sum(diffs >= 0) / len(diffs)
        p_value = min(p_value, 1 - p_value) * 2  # Two-sided test
        
        # Determine which model is better
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        better_model = model_a if mean_a > mean_b else model_b
        
        return p_value, p_value < alpha, better_model
    
    @staticmethod
    def mcnemar_test(y_true, y_pred_a, y_pred_b):
        """
        Use McNemar's test to compare two classification models
        
        Args:
            y_true: Ground truth labels
            y_pred_a: Model A predicted labels
            y_pred_b: Model B predicted labels
            
        Returns:
            chi2: Chi-square statistic
            p_value: Significance p-value
        """
        # Create contingency table
        # b01: Number of samples where model A is wrong but model B is correct
        # b10: Number of samples where model A is correct but model B is wrong
        b01 = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
        b10 = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
        
        # Calculate chi-square statistic (with continuity correction)
        chi2 = (abs(b01 - b10) - 1)**2 / (b01 + b10)
        
        # Calculate p-value
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        
        return chi2, p_value

    @staticmethod
    def plot_bootstrap_distributions(bootstrap_results, title="Bootstrap Performance Distributions"):
        """
        Plot performance distributions from bootstrap sampling
        
        Args:
            bootstrap_results: Bootstrap sampling results
            title: Chart title
        """
        plt.figure(figsize=(12, 8))
        
        models = list(bootstrap_results.keys())
        colors = ['blue', 'green', 'red', 'orange', 'purple'][:len(models)]
        
        for i, (model, scores) in enumerate(bootstrap_results.items()):
            plt.hist(scores, bins=30, alpha=0.6, color=colors[i], label=f'{model} (Mean: {np.mean(scores):.4f})')
            
        plt.axvline(0.5, color='black', linestyle='--', label='Random Classifier')
        plt.xlabel('Performance Metric')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.close()


def extend_main():
    """Extend the main function to add statistical comparison analysis"""
    # Assuming we've already run the original main() and have the following data:
    # y_test: Test set true labels
    # rf_preds, fdnn_preds, raw_preds: Model predictions
    # rf_probs, fdnn_probs, raw_probs: Model prediction probabilities
    
    # 1. Create data loader and model evaluator
    data_loader = DataLoader()
    X, y = data_loader.load_data("dataset.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Train RF transformer and baseline model
    print("\n[Statistical Analysis] Training models for statistical analysis...")
    transformer = RandomForestTransformer(n_estimators=100, random_state=42)
    transformer.fit(X_train, y_train)
    
    rf_baseline = RandomForestClassifier(n_estimators=5, n_jobs=-1, random_state=43)
    rf_baseline.fit(X_train, y_train)
    
    # 3. Prepare ForestDNN model
    n_trees = 100
    input_dim_fdnn = n_trees * 2
    forest_dnn = ForestDNN(
        input_features=input_dim_fdnn, 
        n_hidden_1=256, 
        n_hidden_2=128, 
        n_hidden_3=64,
        dropout_prob=0.3
    ).to(device)
    
    optimizer_fdnn = optim.Adam(forest_dnn.parameters(), lr=1e-6, weight_decay=1e-4)
    trainer = ModelTrainer(device)
    
    # Transform training data
    X_train_transformed = transformer.transform(X_train)
    X_val_transformed = transformer.transform(X_val)
    X_test_transformed = transformer.transform(X_test)
    
    # Train ForestDNN
    loss_history_fdnn = trainer.train_dnn(
        forest_dnn, 
        optimizer_fdnn, 
        X_train_transformed, 
        y_train, 
        X_val_transformed, 
        y_val, 
        batch_size=8, 
        epochs=100
    )
    
    # 4. Prepare Raw DNN model
    input_dim_raw = X_train.shape[1]
    raw_dnn = ForestDNN(
        input_features=input_dim_raw, 
        n_hidden_1=256, 
        n_hidden_2=128, 
        n_hidden_3=64,
        dropout_prob=0.3
    ).to(device)
    
    optimizer_raw = optim.Adam(raw_dnn.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Train Raw DNN
    loss_history_raw = trainer.train_dnn(
        raw_dnn, 
        optimizer_raw, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        batch_size=8, 
        epochs=100
    )
    
    # 5. Get predictions for each model on the test set
    evaluator = ModelEvaluator(device)
    
    # Evaluate RF baseline model
    rf_metrics, rf_preds, rf_probs = evaluator.evaluate_rf(rf_baseline, X_test, y_test)
    
    # Evaluate ForestDNN
    fdnn_metrics, fdnn_preds, fdnn_probs = evaluator.evaluate_dnn(
        forest_dnn, X_test_transformed, y_test
    )
    
    # Evaluate Raw DNN
    raw_metrics, raw_preds, raw_probs = evaluator.evaluate_dnn(raw_dnn, X_test, y_test)
    
    # 6. Statistical analysis
    print("\n=== Statistical Significance Analysis ===")
    analyzer = StatisticalAnalyzer()
    
    # Define prediction probability dictionary
    model_probs_dict = {
        'ForestDNN': fdnn_probs,
        'RF Baseline': rf_probs,
        'Raw DNN': raw_probs
    }
    
    # Define prediction label dictionary
    model_preds_dict = {
        'ForestDNN': (fdnn_probs > 0.5).astype(int),
        'RF Baseline': rf_preds,
        'Raw DNN': (raw_probs > 0.5).astype(int)
    }
    
    # 7. Perform Bootstrap analysis (AUC)
    print("\n[Analysis] Performing bootstrap analysis for AUC scores...")
    bootstrap_auc, ci_auc = analyzer.bootstrap_performance(
        y_test, model_probs_dict, metric_func=roc_auc_score, n_bootstrap=1000
    )
    
    # Plot AUC bootstrap distributions
    analyzer.plot_bootstrap_distributions(bootstrap_auc, "Bootstrap AUC Distributions")
    
    # 8. Perform Bootstrap analysis (Accuracy)
    print("\n[Analysis] Performing bootstrap analysis for Accuracy scores...")
    # Convert probabilities to classification predictions
    accuracy_probs_dict = {model: (probs > 0.5).astype(float) for model, probs in model_probs_dict.items()}
    bootstrap_acc, ci_acc = analyzer.bootstrap_performance(
        y_test, accuracy_probs_dict, metric_func=accuracy_score, n_bootstrap=1000
    )
    
    # Plot Accuracy bootstrap distributions
    analyzer.plot_bootstrap_distributions(bootstrap_acc, "Bootstrap Accuracy Distributions")
    
    # 9. Perform paired Bootstrap tests (AUC)
    print("\n[Analysis] Performing paired bootstrap tests for AUC...")
    model_pairs = [
        ('ForestDNN', 'RF Baseline'),
        ('ForestDNN', 'Raw DNN'),
        ('RF Baseline', 'Raw DNN')
    ]
    
    for model_a, model_b in model_pairs:
        p_value, is_significant, better_model = analyzer.paired_bootstrap_test(
            bootstrap_auc, model_a, model_b
        )
        print(f"{model_a} vs {model_b}: p-value = {p_value:.4f}, Significant? {is_significant}, Better: {better_model}")
        
        # Print confidence intervals
        print(f"  {model_a} AUC CI: [{ci_auc[model_a][0]:.4f}, {ci_auc[model_a][1]:.4f}]")
        print(f"  {model_b} AUC CI: [{ci_auc[model_b][0]:.4f}, {ci_auc[model_b][1]:.4f}]")
    
    # 10. Perform McNemar tests (classification accuracy)
    print("\n[Analysis] Performing McNemar tests for classification accuracy...")
    for i, (model_a, model_b) in enumerate(model_pairs):
        chi2, p_value = analyzer.mcnemar_test(
            y_test, model_preds_dict[model_a], model_preds_dict[model_b]
        )
        is_significant = p_value < 0.05
        print(f"{model_a} vs {model_b}: Chi2 = {chi2:.4f}, p-value = {p_value:.4f}, Significant? {is_significant}")
        
        # Print accuracy and confidence intervals
        print(f"  {model_a} Acc CI: [{ci_acc[model_a][0]:.4f}, {ci_acc[model_a][1]:.4f}]")
        print(f"  {model_b} Acc CI: [{ci_acc[model_b][0]:.4f}, {ci_acc[model_b][1]:.4f}]")
    
    # 11. Comprehensive results analysis
    print("\n=== Comprehensive Analysis Results ===")
    print("AUC Performance Comparison:")
    for model, ci in ci_auc.items():
        print(f"  {model}: {np.mean(bootstrap_auc[model]):.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    print("\nAccuracy Performance Comparison:")
    for model, ci in ci_acc.items():
        print(f"  {model}: {np.mean(bootstrap_acc[model]):.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # 12. Output conclusive recommendations
    print("\n=== Conclusive Recommendations ===")
    # This would summarize based on bootstrap and McNemar test results
    # Actual recommendations would be based on the test results


if __name__ == "__main__":
    extend_main()