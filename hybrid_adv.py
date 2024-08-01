import os
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


# Load data
data_dir = './'  # Update this path if needed
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

features = train_df.drop(columns=['ID', 'Label']).values
labels = train_df['Label'].values
test_features = test_df.drop(columns=['ID']).values

# Remove mean
mean = np.mean(features, axis=0)
features_centered = features - mean
test_features_centered = test_features - mean

# Apply PCA to retain 90% variance
n_components = 1075
pca = PCA(n_components=n_components)
features_pca = pca.fit_transform(features_centered)
test_features_pca = pca.transform(test_features_centered)

# Stratified K-Fold split
seed_value = np.random.randint(1, 1000)
print(f"Random seed value: {seed_value}")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)

# Define optimal number of neighbors for KNN
optimal_n_neighbors = 16

# Placeholder for predictions
nn_predictions = []
logistic_predictions = []

# Build the neural network model
def build_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Iterate over each fold for cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(features_pca, labels)):
    print(f"Processing fold {fold+1}/30...")

    X_train, y_train = features_pca[train_idx], labels[train_idx]
    X_val, y_val = features_pca[val_idx], labels[val_idx]

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=optimal_n_neighbors)
    knn.fit(X_train, y_train)

    # Calculate KNN distances and indices for validation set
    val_distances, val_indices = knn.kneighbors(X_val)
    X_val_knn = np.hstack((val_distances, y_train[val_indices]))

    # Train Logistic Regression on KNN features
    logistic = LogisticRegression()
    logistic.fit(X_val_knn, y_val)
    
    # Calculate KNN distances and indices for test set
    test_knn_distances, test_knn_indices = knn.kneighbors(test_features_pca)
    test_knn_features = np.hstack((test_knn_distances, y_train[test_knn_indices]))

    # Predict on test set using Logistic Regression
    test_logistic_pred = logistic.predict_proba(test_knn_features)
    logistic_predictions.append(test_logistic_pred)

    # Train Neural Network on KNN features
    nn_model = build_nn_model(X_val_knn.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    nn_model.fit(X_val_knn, y_val, epochs=50, batch_size=32, verbose=0, validation_data=(X_val_knn, y_val), callbacks=[early_stopping])

    # Predict on test set using Neural Network
    test_nn_pred = nn_model.predict(test_knn_features)
    nn_predictions.append(test_nn_pred)

# Average the predictions from the neural network models
average_nn_pred = np.mean(nn_predictions, axis=0)
average_logistic_pred = np.mean(logistic_predictions, axis=0)

# Combine predictions from NN and Logistic Regression
combined_predictions = (average_nn_pred + average_logistic_pred) / 2

# Determine the predicted label by taking the argmax of the combined predictions
final_labels = np.argmax(combined_predictions, axis=1)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'Label': final_labels
})

# Save the final submission file
submission_file_path = os.path.join(data_dir, "hybrid_ensemble_submission.csv")
submission_df.to_csv(submission_file_path, index=False)

print("Final predictions have been saved.")
