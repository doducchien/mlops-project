# Run Mlflow UI:
    mlflow ui
# Fetch data, processed data and Fine tune model:
    python src/mlops_pipeline.py

# run api:
    python src/app.py
# run prefect server:
    prefect server start

# run unit test:
    PYTHONPATH=src pytest test/

# start prefect server:
    prefect server start

# push data to dvc remote:
    dvc add models/fine_tuned_gpt2
    dvc push 

