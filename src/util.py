import subprocess
from prefect import task

@task
def track_with_dvc(data_path, model_path):
    print("Tracking files with DVC...")
    subprocess.run(["dvc", "add", data_path], check=True)
    subprocess.run(["dvc", "add", model_path], check=True)
    subprocess.run(["git", "add", f"{data_path}.dvc", f"{model_path}.dvc", ".gitignore"], check=True)
    subprocess.run(["git", "commit", "-m", "Track data and model with DVC"], check=True)
    subprocess.run(["dvc", "push"], check=True)
    print("DVC tracking and push complete.")
