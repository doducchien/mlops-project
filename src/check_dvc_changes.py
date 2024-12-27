import os

def check_data_and_push():
    status = os.popen("dvc status").read().strip()
    if "up to date" in status:
        print("✅ Data and pipelines are up to date. No action needed.")
    else:
        print("⚠️ Data or pipelines have changed. Proceeding to push changes.")
        # DVC add, commit, và push
        os.system("dvc add data/processed_data.json")
        os.system("git add data/processed_data.json.dvc")
        os.system('git commit -m "Data changes detected, updated processed data."')
        os.system("dvc push")
        print("✅ Data and pipelines have been pushed to DVC remote.")

if __name__ == "__main__":
    check_data_and_push()
