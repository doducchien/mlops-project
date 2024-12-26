import subprocess

def should_fine_tune():
    result = subprocess.run("dvc status", shell=True, capture_output=True, text=True)
    if "changed outs" in result.stdout:
        return True
    else:
        return False
