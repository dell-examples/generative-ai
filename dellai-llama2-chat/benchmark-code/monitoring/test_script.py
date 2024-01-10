import pynvml as nv

nv.nvmlInit()

handle = nv.nvmlDeviceGetHandleByIndex(0)

power = nv.nvmlDeviceGetPowerUsage(handle)

power_limit = nv.nvmlDeviceGetEnforcedPowerLimit(handle)

fan_speed = nv.nvmlDeviceGetFanSpeed(handle)

print(power)

print(power_limit)

print(fan_speed)