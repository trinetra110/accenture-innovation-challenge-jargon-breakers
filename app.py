from flask import Flask, request, render_template, redirect, url_for, flash, session
import pandas as pd
import os
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
import logging
import openai

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set OpenAI API key from environment variable
openai.api_key = "####"

# Route for the homepage to upload dataset
@app.route('/')
def index():
    return render_template('index.html')

# Function to load dataset based on file extension
def load_dataset(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

# Route to analyze uploaded dataset and display column headers
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        # Redirect to the index page if the method is GET
        return redirect(url_for('index'))
    elif request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            logging.debug("No file part in request.")
            return redirect(url_for('index'))
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            logging.debug("No file selected for upload.")
            return redirect(url_for('index'))
        if file:
            try:
                # Save the uploaded file to a temporary location
                temp_dir = tempfile.gettempdir()
                temp_file_path = os.path.join(temp_dir, file.filename)
                file.save(temp_file_path)
                logging.debug(f"File saved to temporary location: {temp_file_path}")
                df = load_dataset(temp_file_path)
                logging.debug("Dataset loaded successfully.")
                # Save the dataset to a temporary file for later use
                temp_dataset_path = os.path.join(temp_dir, 'dataset.csv')
                df.to_csv(temp_dataset_path, index=False)
                session['temp_dataset_path'] = temp_dataset_path  # Save path to session
                columns = df.columns.tolist()
                # Render the analyze.html template with dataset preview and columns
                return render_template('analyze.html', tables=[df.head().to_html(classes='data')], columns=columns)
            except Exception as e:
                flash(f'Error: {e}')
                logging.error(f"An error occurred: {e}")
                return redirect(url_for('index'))

# Route to perform prediction and generate insights
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Redirect to the index page if the method is GET
        return redirect(url_for('index'))
    elif request.method == 'POST':
        target_column = request.form.get('target_column')
        if 'temp_dataset_path' not in session:
            flash('No dataset found. Please upload again.')
            return redirect(url_for('index'))
        temp_dataset_path = session['temp_dataset_path']
        df = pd.read_csv(temp_dataset_path)
        try:
            # Preprocess data
            df.fillna(df.mean(numeric_only=True), inplace=True)
            df = pd.get_dummies(df, drop_first=True)

            if target_column not in df.columns:
                flash('Invalid target column selected.')
                return redirect(url_for('analyze'))

            # Split the data into features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Determine if classification or regression
            if y.nunique() <= 10 and y.dtype.kind in 'biu':
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                score_text = f'Accuracy Score: {score:.2f}'
            else:
                model = DecisionTreeRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = mean_absolute_error(y_test, y_pred)
                score_text = f'Mean Absolute Error: {score:.2f}'

            # Generate insights using OpenAI ChatCompletion API
            summary = df.describe(include='all').to_string()
            prompt = f"Based on the following data summary, generate insights for business decision making:\n{summary}"

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
            )
            insights = response['choices'][0]['message']['content'].strip()

            return render_template('result.html', target_column=target_column, score_text=score_text, insights=insights)
        except Exception as e:
            flash(f'Error: {e}')
            logging.error(f"An error occurred: {e}")
            return redirect(url_for('analyze'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)