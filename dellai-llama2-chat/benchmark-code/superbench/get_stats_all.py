import csv
import datetime
import json

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
        #print(f"{filename}: {total / i}")
    return total / i

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
        #print(f"{filename}: {total / i}")
    return total / i
        
def get_data(main_dir, folders):
    retval = []
    for folder in folders:
        with open(f"{main_dir}{folder}/nodes/scalers/benchmarks/gpt_models/rank0/results.json") as f:
            loaded_json = json.load(f)[0]
            if (loaded_json["result"]["return_code"][0] != 0):
                print(f"{folder} has non zero return code")
                continue
            #print(list(loaded_json["result"].keys())[1])
            result_keys = list(loaded_json["result"].keys())
            #print(result_keys)
            retval.append((loaded_json["start_time"], loaded_json["end_time"], loaded_json["result"][result_keys[1]][0], loaded_json["result"][result_keys[2]][0]))
    return retval

if __name__ == "__main__":
    folders = [
        "gpt2-small_float16_inference_1400",
        "gpt2-small_fp8_hybrid_inference_682",
        "gpt2-small_fp8_e4m3_inference_682",
        "gpt2-small_fp8_hybrid_inference_1400",
        "gpt2-small_fp8_e4m3_inference_1400",
        "gpt2-medium_float32_inference_314",
        "gpt2-medium_float16_inference_630",
        "gpt2-medium_fp8_hybrid_inference_314",
        "gpt2-medium_fp8_e4m3_inference_314",
        "gpt2-medium_fp8_hybrid_inference_630",
        "gpt2-medium_fp8_e4m3_inference_630",
        "gpt2-large_float32_inference_174",
        "gpt2-large_float16_inference_356",
        "gpt2-large_fp8_hybrid_inference_174",
        "gpt2-large_fp8_e4m3_inference_174",
        "gpt2-large_fp8_hybrid_inference_356",
        "gpt2-large_fp8_e4m3_inference_356",
        "gpt2-xl_float32_inference_104",
        "gpt2-xl_float16_inference_218",
        "gpt2-xl_fp8_hybrid_inference_104",
        "gpt2-xl_fp8_e4m3_inference_104",
        "gpt2-xl_fp8_hybrid_inference_218",
        "gpt2-xl_fp8_e4m3_inference_218",
    ]
    folders = [ "gpt2-medium_float16_inference_314", "gpt2-xl_float16_inference_104" ]
    folders = [
        "gpt2-small_float16_train_80",
        "gpt2-small_fp8_hybrid_train_80",
        "gpt2-small_fp8_e4m3_train_80",
        "gpt2-medium_float16_train_28",
        "gpt2-medium_fp8_hybrid_train_28",
        "gpt2-medium_fp8_e4m3_train_28",
        "gpt2-large_float16_train_14",
        "gpt2-large_fp8_hybrid_train_14",
        "gpt2-large_fp8_e4m3_train_14",
        "gpt2-xl_float16_train_6",
        "gpt2-xl_fp8_hybrid_train_6",
        "gpt2-xl_fp8_e4m3_train_6",
    ]
    folders = [ "gpt2-xl_float32_train_6_10000_steps" ]
    tuples = get_data("/mnt/scalers/exps/outputs/", folders)
    for i, tup in enumerate(tuples):
        #print(tup)
        start = datetime.datetime.strptime(f"{tup[0]}", "%Y-%m-%d %H:%M:%S")
        end = datetime.datetime.strptime(f"{tup[1]}", "%Y-%m-%d %H:%M:%S")
        print(f"{folders[i]},{gpu_usage_worker('gpu_memory_usage.csv', start, end)},{other_worker('cpu_usage.csv', start, end)},{other_worker('memory_usage.csv', start, end)},{tup[2]},{tup[3]}")
    #start = datetime.datetime.strptime("2023-04-20 11:28:05", "%Y-%m-%d %H:%M:%S")
    #end = datetime.datetime.strptime("2023-04-20 11:31:36", "%Y-%m-%d %H:%M:%S")
    #other_worker('memory_usage.csv', start, end)
    #other_worker('cpu_usage.csv', start, end)
    #gpu_usage_worker('gpu_memory_usage.csv', start, end)
    #worker('gpu_computational_usage.csv')
