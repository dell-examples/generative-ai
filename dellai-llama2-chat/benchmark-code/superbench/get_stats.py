import csv
import datetime

def gpu_usage_worker(filename, start, end):
    total = 0
    with open(filename, newline='') as f:
        reader = csv.DictReader(f, delimiter=',')
        i = 0
        for row in reader:
            #print(row["_start"], row["_stop"], row["_time"])
            try:
                now = datetime.datetime.strptime(row["_time"], "%Y-%m-%dT%H:%M:%SZ")
                if (now >= start and now <= end and float(row["_value"]) > 30):
                    total += float(row["_value"])
                    i += 1
            except:
                pass
        print(f"{filename}: {total / i}")

def other_worker(filename, start, end):
    total = 0
    with open(filename, newline='') as f:
        reader = csv.DictReader(f, delimiter=',')
        i = 0
        for row in reader:
            #print(row["_start"], row["_stop"], row["_time"])
            try:
                now = datetime.datetime.strptime(row["_time"], "%Y-%m-%dT%H:%M:%SZ")
                if (now >= start and now <= end):
                    total += float(row["_value"])
                    i += 1
            except:
                pass
        print(f"{filename}: {total / i}")
        

if __name__ == "__main__":
    start = datetime.datetime.strptime("2023-04-20 11:28:05", "%Y-%m-%d %H:%M:%S")
    end = datetime.datetime.strptime("2023-04-20 11:31:36", "%Y-%m-%d %H:%M:%S")
    other_worker('memory_usage.csv', start, end)
    other_worker('cpu_usage.csv', start, end)
    gpu_usage_worker('gpu_memory_usage.csv', start, end)
    #worker('gpu_computational_usage.csv')
