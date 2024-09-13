import matplotlib.pyplot as plt
import numpy as np

# Misinformation scores for each model based on manual assessment from specialist
model_scores = {
    "Model 1 (Base)": [-1, -1, 0, -1, -1, 0],
    "Model 2 (Medium)": [0, -1, 0, 0, -1, 0],
    "Model 3 (Large)": [0, 0, 1, 0, 0, 1],
}

# Calculate the average misinformation score for each model
average_scores = {model: np.mean(scores) for model, scores in model_scores.items()}

# Plotting the results
models = list(average_scores.keys())
scores = list(average_scores.values())

plt.figure(figsize=(10, 6))
plt.barh(models, scores, color=['red', 'orange', 'green'])
plt.xlabel('Average Misinformation Score')
plt.title('Degree of Misinformation in GPT-2 Levels')
plt.xlim([-1, 1])  # Scores range from -1 (high misinformation) to +1 (accurate)
plt.axvline(x=0, color='black', linestyle='--')  # Line at 0 to differentiate between negative and positive scores
plt.show()
