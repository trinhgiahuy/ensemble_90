from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2

data_dir = '../data'
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

features = train_df.drop(columns=['ID', 'Label']).values
labels = train_df['Label'].values
test_features = test_df.drop(columns=['ID']).values

# remove mean
mean = np.mean(features, axis=0)
features_centered = features - mean
test_features_centered = test_features - mean

seed_value = np.random.randint(1, 1000)
print(f"Random seed value: {seed_value}")

# 1075 for 90%
# 1969 for 97%
n_components = 1075
pca = PCA(n_components=n_components)
features_pca = pca.fit_transform(features_centered)
test_features_pca = pca.transform(test_features_centered)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed_value)

# optimal n can be 16, 42,
optimal_n_neighbors = 16

model_predictions = []
logistic_predictions = []
nn_predictions = []

# Storage for KNN distances and labels
knn_distances = []
knn_labels = []

# Placeholder for neural network predictions
nn_predictions = []

# Calculate KNN distances and labels for each fold
for fold, (train_idx, test_idx) in enumerate(skf.split(features_pca, labels)):
    print(f"Processing fold {fold+1}/4...")

    X_train, y_train = features_pca[train_idx], labels[train_idx]
    X_test, y_test = features_pca[test_idx], labels[test_idx]

    knn = KNeighborsClassifier(n_neighbors=optimal_n_neighbors)
    knn.fit(X_train, y_train)

    distances, indices = knn.kneighbors(X_test)
    X_test_knn = np.hstack((distances, y_train[indices]))

    input_dim = X_test_knn.shape[1]
    # nn = Sequential([
    #     Input(shape=(input_dim,)),
    #     Dense(64, activation='relu'),
    #     Dense(3, activation='softmax')
    # ])
    nn = Sequential([
      Dense(128, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(0.001)),
      Dropout(0.5),  # Add dropout layer for regularization
        # Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        # Dropout(0.5),
      Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
      Dropout(0.5),
      Dense(3, activation='softmax')  # Use softmax for multi-class classification
    ])
    nn.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    nn.fit(X_test_knn, y_test, epochs=50, batch_size=32, verbose=0)

    # Store validation set predictions
    val_predictions = nn.predict(X_test_knn)
    print(f"Validation predictions shape: {val_predictions.shape}")

    # Store test set predictions
    test_knn_distances, test_knn_indices = knn.kneighbors(test_features_pca)
    test_knn_features = np.hstack((test_knn_distances, y_train[test_knn_indices]))
    test_nn_pred = nn.predict(test_knn_features)
    print(f"Test predictions shape: {test_nn_pred.shape}")

    nn_predictions.append(test_nn_pred)

# Average the predictions from the neural network models
average_nn_pred = np.mean(nn_predictions, axis=0)
print(f"average_nn_pred: {average_nn_pred}")

# Determine the predicted label by taking the argmax of the averaged predictions
final_labels = np.argmax(average_nn_pred, axis=1)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'Label': final_labels
})

# Save the final submission file
submission_file_path = os.path.join(data_dir, "ensemble_submission_nn.csv")
submission_df.to_csv(submission_file_path, index=False)

print("Final predictions have been saved.")
