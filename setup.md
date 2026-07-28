# Setting Up "The Predictor" Environment

This guide will walk you through setting up your local Python environment to run the Streamlit application and the MLflow training script.

## 1. Prerequisites

Ensure you have **Python 3.8 or higher** installed on your system. You can check your Python version by running:

```bash
python --version
```

## 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to keep this project's dependencies isolated from the rest of your system.

Open your terminal or command prompt, navigate to the project folder, and run:

```bash
python -m venv .venv
```

This will create a new folder named `venv` containing your isolated Python environment.

## 3. Activate the Virtual Environment

Before installing any packages or running the code, you must activate the environment.

**On macOS / Linux:**

```bash
source .venv/bin/activate
```

**On Windows:**

```bash
.\.venv\Scripts\activate
```

*(You should now see `(venv)` at the beginning of your terminal prompt, indicating the environment is active).*

## 4. Install Dependencies

Next, install all the required Python libraries. Ensure you are in the same directory as the `requirements.txt` file and run:

```bash
pip install -r requirements.txt
```

### Adding MLflow Support

If your `requirements.txt` does not already include `mlflow`, you will need to install it manually to use the training tracking features:

```bash
pip install mlflow
```

## 5. Running the Application

Once the installation is complete, you can launch the Streamlit web application by running:

```bash
streamlit run app.py
```

A browser window should automatically open pointing to `http://localhost:8501`.

## 6. Running the MLflow Training Script

To see how the model was trained and tracked using MLflow, run the training script:

```bash
python train_mlflow.py
```

Once the script finishes, you can view the logged metrics and the saved model by starting the MLflow UI:

```bash
mlflow ui
```

Then, navigate to `http://127.0.0.1:5000` in your web browser.

---

**Deactivating the Environment:**
When you are done working on the project, you can exit the virtual environment by simply typing:

```bash
deactivate
```
