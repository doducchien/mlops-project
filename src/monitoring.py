from time import sleep

def monitor_model():
    print("Monitoring model...")
    # Giả định kiểm tra log hiệu suất hoặc endpoint
    while True:
        print("Checking model health...")
        # Thực hiện kiểm tra tại đây
        sleep(60)

if __name__ == "__main__":
    monitor_model()
