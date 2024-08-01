import os
import numpy as np
import pandas as pd

results_dir = 'results_90_s'
prediction_files = [f for f in os.listdir(results_dir) if f.startswith('test_predictions_model_')]

# Initialize an array to store cumulative predictions
cumulative_predictions = None

# Read each prediction file and accumulate the predictions
for file in prediction_files:
    filepath = os.path.join(results_dir, file)
    print(filepath)
    data = pd.read_csv(filepath)
    predictions = data.iloc[:, 1:].values  # Exclude the ID column
    if cumulative_predictions is None:
        cumulative_predictions = predictions
    else:
        cumulative_predictions += predictions

# Average the predictions
average_predictions = cumulative_predictions / len(prediction_files)

# Determine the predicted label by taking the argmax of the averaged predictions
final_labels = np.argmax(average_predictions, axis=1)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'ID': data['ID'],
    'Label': final_labels
})

# Save the final submission file
submission_file_path = os.path.join(results_dir, "ensemble_submission.csv")
submission_df.to_csv(submission_file_path, index=False)

print("Averaged predictions and final labels saved.")
