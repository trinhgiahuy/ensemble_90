import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

# Load data
data_dir = './'  # Update this path if needed
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Split features and labels
features = train_df.drop(columns=['ID', 'Label']).values
labels = train_df['Label'].values  # Use labels directly (not one-hot encoded)
test_features = test_df.drop(columns=['ID']).values

# Center the data (remove mean)
mean = np.mean(features, axis=0)
features_centered = features - mean
test_features_centered = test_features - mean

# Apply PCA to retain 90% variance
pca = PCA(n_components=1075)  # Assuming 1075 components retain 90% variance
features_pca = pca.fit_transform(features_centered)
test_features_pca = pca.transform(test_features_centered)

# Initialize the TabPFN model
model = TabPFNClassifier(device='cpu')  # Use 'cuda' if you have a compatible GPU

# Train and predict using StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
predictions = np.zeros((test_features_pca.shape[0], len(np.unique(labels))))

for train_idx, val_idx in skf.split(features_pca, labels):
    X_train, y_train = features_pca[train_idx], labels[train_idx]
    X_val, y_val = features_pca[val_idx], labels[val_idx]

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the validation and test sets
    val_pred = model.predict_proba(X_val)
    test_pred = model.predict_proba(test_features_pca)

    # Aggregate the test predictions
    predictions += test_pred / n_splits

# Determine the predicted label by taking the argmax of the averaged predictions
final_labels = np.argmax(predictions, axis=1)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'Label': final_labels
})

# Save the final submission file
submission_file_path = os.path.join(data_dir, "tabpfn_submission.csv")
submission_df.to_csv(submission_file_path, index=False)

print("Final predictions have been saved.")
