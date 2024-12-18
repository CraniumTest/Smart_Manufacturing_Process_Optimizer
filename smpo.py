import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from transformers import pipeline
import numpy as np

class SMPO:
    def __init__(self, machine_data_path):
        self.machine_data = pd.read_csv(machine_data_path)
        self.rf_model = RandomForestClassifier()
        self.nlp = pipeline('summarization', model='facebook/bart-large-cnn')

    def train_predictive_model(self):
        # Preprocess data
        features = self.machine_data.drop('maintenance_needed', axis=1)
        target = self.machine_data['maintenance_needed']
        
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        accuracy = self.rf_model.score(X_test, y_test)
        print(f"Model trained with an accuracy of: {accuracy * 100:.2f}%")

    def predict_maintenance(self, machine_stats):
        prediction = self.rf_model.predict([machine_stats])
        if prediction == 1:
            return "Maintenance Needed"
        return "No Maintenance Needed"

    def explain_maintenance(self, machine_stats):
        predicted_class = self.predict_maintenance(machine_stats)
        explanation = self.nlp(f"The prediction is '{predicted_class}'. This means that {predicted_class.lower()} because the machine stats show values indicating potential issues.")
        return explanation[0]['summary_text']

    def optimize_process(self):
        # Placeholder for optimization logic
        bottlenecks = "Detected a bottleneck in the assembly line due to unbalanced workload."
        improvements = "Reallocate resources from idle stations to bottleneck stations to optimize the process."
        return bottlenecks, improvements

if __name__ == '__main__':
    # Assuming a CSV path for sample implementation
    s = SMPO(machine_data_path='machine_data.csv')
    s.train_predictive_model()

    # Example machine stats for prediction
    machine_stats = np.random.random((5))  # Random data for 5 features
    maintenance_prediction = s.predict_maintenance(machine_stats)
    print(f"Maintenance Prediction: {maintenance_prediction}")

    # Get natural language explanation
    explanation = s.explain_maintenance(machine_stats)
    print(f"Explanation: {explanation}")

    # Process Optimization
    bottlenecks, improvements = s.optimize_process()
    print(f"Bottlenecks: {bottlenecks}")
    print(f"Improvement Suggestions: {improvements}")
