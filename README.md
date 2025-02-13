### Requirement

- **Python Version 3.12** for **zenml** installation

### **Install Dependencies**

Make virtual environment for dependencies:

```markdown
python -m venv (your-environment)
```

Install dependencies packages from requirements.txt :

```bash
pip install -r requirements.txt
```

Intstall zenml:

```bash
pip install "zenml[server]"

```

### Setup Integration with MLflow Example

Install **integration mlflow**:

```bash
zenml integration install mlflow -y
```

Register your experiment tracker for MLflow integration:

```bash
zenml experiment-tracker register <YOUR_NAME_EXPERIMENT> --flavor=mlflow
```

example:

```bash
# Register the MLflow experiment tracker
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow
```

### Running Project in Local

Execute pipeline:

```bash
python run_pipeline.py
```

Open Zenml Dashboard:

```bash
zenml login --local --blocking
```

Open MLflow Dashboard:

```bash
mlflow ui
```
