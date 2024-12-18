# README for Smart Manufacturing Process Optimizer (SMPO) Prototype

## Overview

The Smart Manufacturing Process Optimizer (SMPO) is a prototype Python application designed to demonstrate core functionalities of predictive maintenance and process optimization in manufacturing settings. This prototype focuses specifically on providing maintenance predictions and natural language explanations with minimal setup, leveraging pre-existing Python libraries for machine learning and natural language processing.

## Key Features

1. **Predictive Maintenance Advisor**:
   - Utilizes machine learning models (Random Forest Classifier) to predict if a machine needs maintenance.
   - Trained on simulated machine operation data to identify potential maintenance needs.

2. **Natural Language Explanations**:
   - Employs a summarization pipeline from the `transformers` library to generate human-readable explanations for maintenance predictions. This assists users in understanding the rationale behind predictions.

3. **Process Optimization Consultant**:
   - Provides a basic framework for identifying bottlenecks in operations and suggesting improvements.
   - Currently placeholders are used; further development is needed for full functionality.

## Installation and Dependencies

To set up the SMPO prototype, you will need the following Python libraries, which are specified in the `requirements.txt` file:

- `pandas`: For data manipulation.
- `scikit-learn`: For training the Random Forest model.
- `transformers`: For NLP tasks using pre-trained models.
- `torch`: Required by the transformers library for deep learning model execution.

### Installation Steps

1. **Clone the Repository**: Start by cloning the repository or creating the project directory where the script will reside.

2. **Install Required Libraries**: Navigate into the project directory and run the following command to install necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Machine Data**: The script expects a CSV file named `machine_data.csv` in the root of the project directory with appropriate features and target column (`maintenance_needed`). Customize this file to suit your simulation or testing needs.

## Execution

Run the Python script `smpo.py` to:

1. Train a predictive model on your machine data.
2. Make maintenance predictions on sample machine statistics.
3. Obtain natural language summaries explaining predictions.
4. Get basic process optimization suggestions.

## Notes

- **Sample Data**: Modify the `machine_data.csv` with real machine operational data to enhance the predictive capabilities.
- **Model Customization**: The logic can be adapted to integrate more sophisticated predictive models and include additional features for a more robust analysis.
- **Natural Language Model**: The default BART model is used here for natural language generation; you can adjust the model choice based on specific domain requirements.

Through this prototype, the SMPO seeks to leverage machine learning and natural language processing to improve manufacturing processes, focusing on operational efficiency and proactive maintenance strategies. This provides a stepping stone towards more comprehensive solutions tailored to specific manufacturing environments.
