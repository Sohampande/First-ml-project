# Student Math Score Prediction – Machine Learning Project

This project is an end-to-end Machine Learning web application that predicts a student’s math score based on their personal and academic attributes.
A user inputs relevant details through a simple interface, and the trained ML model returns the predicted performance score.

The project includes:

A complete ML pipeline (data ingestion → transformation → model training → evaluation)

A Flask web application for real-time prediction

Saved artifacts such as preprocessing objects, trained models, logs, and configuration files

A modular, production-ready project structure

## Features

User-friendly prediction interface (Flask + HTML templates)

Modular ML pipeline following best practices

Automatic logging & exception handling

Training artifacts saved for reproducibility

Easily deployable (AWS Elastic Beanstalk or Azure App Service — optional for future work)

📂 Project Structure
the project structure is available above. Have a look !

## How the Model Works

The model is trained on a dataset of student performance and uses inputs such as:

Gender

Parental education

Lunch type

Test preparation course

Reading score

Writing score

…and other relevant features

After preprocessing and feature engineering, the model predicts a math score.

## Installation
1. Clone the repository
git clone https://github.com/<your-username>/First-ml-project.git
cd First-ml-project

2. Create a virtual environment
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

3. Install dependencies
pip install -r requirements.txt

4. Running the Web Application Locally

Simply run:

python application.py


The Flask server will start, usually at:

http://127.0.0.1:5000/


Open it in your browser to use the prediction interface.

## Training the Model (optional)

If you want to retrain the model, you can run the training pipeline:

python -m src.pipeline.train_pipeline


The newly trained model and transformer will be stored in the artifact/ directory.

## Making Predictions Programmatically

Example usage:

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

data = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=72,
    writing_score=74
)

df = data.get_data_as_data_frame()
predict_pipeline = PredictPipeline()
result = predict_pipeline.predict(df)

print("Predicted Math Score:", result)

## Logging

All pipeline and app activity is logged automatically inside the logs/ folder.
Each run generates a new timestamped log file.

## Future Work

Deployment using AWS Elastic Beanstalk or Azure App Service

Docker containerization

Hyperparameter tuning for better accuracy

UI enhancements

Add additional input validations

## License

This project is for educational use. You may reuse the code with attribution.

🙌 Acknowledgements

This project follows industry-standard ML pipeline structuring inspired by popular tutorials and production ML workflow best practices.
