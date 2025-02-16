# Gunakan image resmi Python sebagai base image
FROM python:3.12-slim

# Setel direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt .

# Instal dependensi yang diperlukan
RUN pip install --no-cache-dir -r requirements.txt

# Instal ZenML dengan dukungan server
RUN pip install "zenml[server]"

# Instal integrasi MLflow untuk ZenML
RUN zenml integration install mlflow -y

# Salin semua file proyek ke dalam container
COPY . .

# Jalankan perintah untuk mendaftarkan experiment tracker MLflow
RUN zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow

# Tentukan perintah default saat container dijalankan
CMD ["python", "run_pipeline.py"]
