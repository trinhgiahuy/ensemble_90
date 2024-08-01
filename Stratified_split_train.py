import os
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
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

# Build the model function
def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
        Dropout(0.5),  # Add dropout layer for regularization
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(3, activation='softmax')  # Use softmax for multi-class classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

results_dir = 'results_90_s'
os.makedirs(results_dir, exist_ok=True)

# Number of models in the ensemble
n_models = 30

# Placeholder for model predictions
all_predictions = []

# Stratified K-Fold split

skf = StratifiedKFold(n_splits=n_models, shuffle=True, random_state=seed_value)
for fold, (train_idx, val_idx) in enumerate(skf.split(features_pca, labels)):
    print(f"Training model {fold+1}/{n_models}...")
    # Set different seed values for each iteration
    # seed_value = np.random.randint(20, 1000) + np.random.randint(20, 30) - 2 * fold
    seed_value = np.random.randint(1, 9999) + fold
    

    print(f"Random seed value: {seed_value}")
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Split the transformed features into training and validation sets
    X_train, X_val = features_pca[train_idx], features_pca[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # Further split the features to create diversity in columns consistently
    cols = np.random.choice(features_pca.shape[1], int(features_pca.shape[1] * 0.9), replace=False)
    X_train = X_train[:, cols]
    X_val = X_val[:, cols]
    test_features_subset = test_features_pca[:, cols]

    # Build and train the model
    model = build_model(X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    # Predict on the test data subset
    test_predictions = model.predict(test_features_subset)
    all_predictions.append(test_predictions)
    
    # Save the predictions to a CSV file for this model
    output = np.hstack((test_df['ID'].values.reshape(-1, 1), test_predictions))
    np.savetxt(os.path.join(results_dir, f"test_predictions_model_{seed_value}.csv"), output, delimiter=",", header="ID,Class_0,Class_1,Class_2", comments='', fmt='%.6f')

print("All models have been trained and predictions saved.")

# Ensemble: Average the predictions from all models
ensemble_predictions = np.mean(all_predictions, axis=0)

# Determine the predicted label by taking the argmax of the averaged predictions
final_labels = np.argmax(ensemble_predictions, axis=1)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'Label': final_labels
})

# Save the final submission file
submission_file_path = os.path.join(data_dir, "ensemble_submission.csv")
submission_df.to_csv(submission_file_path, index=False)

print("Final predictions have been saved.")
