import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2


# Set random seed for reproducibility
seed_value = 25
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


data_dir = '../data'
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Split features and labels
features = train_df.drop(columns=['ID', 'Label']).values
labels = pd.get_dummies(train_df['Label']).values  # One-hot encoding for multi-class classification
test_features = test_df.drop(columns=['ID']).values

# Center the data (remove mean)
mean = np.mean(features, axis=0)
features_centered = features - mean
test_features_centered = test_features - mean
#
# 1075 for 90% variance
n_components = 1075 
pca = PCA(n_components=n_components)
features_pca = pca.fit_transform(features_centered)
test_features_pca = pca.transform(test_features_centered)

def build_model(input_dim):
  model =  Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(0.001)),
    Dropout(0.5),  # Add dropout layer for regularization
      # Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
      # Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Use softmax for multi-class classification
  ])
  model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

  return model

results_dir = 'results_90'
os.makedirs(results_dir, exist_ok=True)

# Number of models in the ensemble
n_models = 20

for i in range(n_models):
    print(f"Training model {i+1}/{n_models}...")
    # Set different seed values for each iteration
    seed_value = np.random.randint(20, 1000)
    print(f"Random seed value: {seed_value}")
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Split the transformed features into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features_pca, labels, test_size=0.2, random_state=seed_value)

    # Further split the features to create diversity in columns consistently
    cols = np.random.choice(features_pca.shape[1], int(features_pca.shape[1] * 0.7), replace=False)
    # cols = np.random.choice(features_pca.shape[1], int(features_pca.shape[1] * 0.9), replace=False)

    X_train = X_train[:, cols]
    X_val = X_val[:, cols]
    test_features_subset = test_features_pca[:, cols]

    # Build and train the model
    model = build_model(X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Predict on the consistent test data subset
    test_predictions = model.predict(test_features_subset)

    # Save the predictions to a CSV file for this model
    output = np.hstack((test_df['ID'].values.reshape(-1, 1), test_predictions))
    np.savetxt(os.path.join(results_dir, f"test_predictions_model_{seed_value}.csv"), output, delimiter=",", header="ID,Class_0,Class_1,Class_2", comments='', fmt='%.6f')

print("All models have been trained and predictions saved.")
