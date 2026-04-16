📊 Student Performance Prediction – End-to-End Machine Learning Project
An end-to-end machine learning system that predicts student performance based on various input features. The project follows a modular ML pipeline architecture, including data ingestion, transformation, model training, and deployment through a web interface.
🚀 Project Overview
This project demonstrates a complete production-style ML workflow:
Data ingestion and preprocessing
Feature engineering and transformation
Model training and evaluation
Pipeline-based prediction system
Web deployment using Flask
Logging and custom exception handling
The system is structured for scalability, maintainability, and deployment readiness.
🧠 Problem Statement
The goal is to predict student performance scores based on various academic and demographic features.
This helps in understanding key factors influencing academic success and can be used for educational analytics.
🏗️ Project Architecture
The project follows a modular pipeline structure:
tree src


📂 Repository Structure

├── artifacts/                   # Saved models & preprocessing objects
│   ├── model.pkl
│   ├── preprocessor.pkl
│
├── data/                        # Raw dataset
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│
├── notebook/                    # Jupyter notebooks (EDA & experiments)
│   ├── Student_performance_evaluation.ipynb
│
├── src/                         # Source code (ML pipeline)
│
├── templates/                   # Flask HTML templates
│   ├── home.html
│   ├── index.html
│   ├── about.html
│
├── application.py               # Flask web application
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
├── readme.md                    # Project documentation



⚙️ Tech Stack
Programming Language: Python 🐍
Machine Learning: Scikit-learn, Pandas, NumPy
Model Serialization: Pickle
Web Framework: Flask
Frontend: HTML (Jinja templates)
Logging & Debugging: Custom logger + exception handling

🔄 ML Pipeline Workflow
Data Ingestion
Load dataset
Train-test split
Store processed data in data/
Data Transformation
Handle missing values
Feature encoding
Scaling & preprocessing
Save preprocessor as preprocessor.pkl
Model Training
Train ML models
Evaluate performance
Select best model
Save model as model.pkl
Prediction Pipeline
Load trained model & preprocessor
Accept user input
Generate predictions


🌐 Web Application
A Flask-based web app provides a simple UI where users can input student-related data and get predicted performance scores.
Run locally:

python application.py

Then open:

http://127.0.0.1:5000/


📦 Installation
Clone the repository:

git clone https://github.com/your-username/student-performance-prediction.git
cd student-performance-prediction



Create virtual environment:

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows


pip install -r requirements.txt

▶️ Run the Project
1. Train the Model
python src/pipeline/train_pipeline.py
2. Run Web App
   python application.py

📊 Key Features
End-to-end ML pipeline architecture
Modular and reusable codebase
Production-style project structure
Flask deployment for real-time predictions
Logging and exception tracking system


📁 Saved Artifacts
After training, the following files are generated:
model.pkl → Trained ML model
preprocessor.pkl → Data preprocessing pipeline


🧪 Future Improvements
Replace Flask with FastAPI for better scalability
Deploy on AWS / Azure / Render
Add CI/CD pipeline
Add Docker support
Improve model performance with advanced algorithms
Add experiment tracking (MLflow)


👨‍💻 Author
Abhijeet Malik
M.Sc. Data Science & AI (Final Semester)
University of the Saarland, Germany


📜 License
This project is for educational purposes.
