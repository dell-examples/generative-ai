import time
import nvgpu

#Logging file to get the gpu data

while True:

    current_time = int(time.time() * 1000000000)
    gpu_list = nvgpu.gpu_info()

    for i in range(0,98):

        for gpu_dict in gpu_list:
                line = (f"gpu_stats,device={gpu_dict['index']} "
                        f"comp_util={gpu_dict['utilization']},"
                        f"mem_util={gpu_dict['mem_used_percent']},"
                        f"temp={gpu_dict['temp']},"
                        f"power_used={gpu_dict['power_used']},"
                        f"power_percent={gpu_dict['power_used']/gpu_dict['power_limit']}"
                        f" {current_time} \n")
                print(line)


        time.sleep(0.01)

        #so the thing about telegraf is that it collects metrics after 98 units- this solution will be much healthier for performance

# gpu_list = nvgpu.gpu_info()
# print(gpu_list[0]['index'])