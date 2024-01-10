import yaml
import subprocess
import time 

def run_benchmark():
    deploy_str = "sb deploy --docker-image superbench/superbench:v0.8.0-cuda12.1 -f /mnt/scalers/exps/new_sb/local.ini"
    subprocess.run(deploy_str, shell=True)

    tests = [# (model_name, data_type, train/inference, batch_size)
            ("gpt2-small", "float16", "train", "80"),
            ("gpt2-small", "fp8_hybrid", "train", "80"),
            ("gpt2-small", "fp8_e4m3", "train", "80"),
            ("gpt2-medium", "float16", "train", "28"),
            ("gpt2-medium", "fp8_hybrid", "train", "28"),
            ("gpt2-medium", "fp8_e4m3", "train", "28"),
            ("gpt2-large", "float16", "train", "14"),
            ("gpt2-large", "fp8_hybrid", "train", "14"),
            ("gpt2-large", "fp8_e4m3", "train", "14"),
            ("gpt2-xl", "float16", "train", "6"),
            ("gpt2-xl", "fp8_hybrid", "train", "6"),
            ("gpt2-xl", "fp8_e4m3", "train", "6"),

            #("gpt2-small", "float32", "train", "80"),
            #("gpt2-small", "float32", "inference", "682"),
            #("gpt2-small", "float16", "train", "145"),
            #("gpt2-small", "float16", "inference", "682"),
            #("gpt2-medium", "float32", "train", "28"),
            #("gpt2-medium", "float32", "inference", "330"),
            #("gpt2-medium", "float16", "train", "56"),
            #("gpt2-medium", "float16", "inference", "512"),
            #("gpt2-large", "float32", "train", "14"),
            #("gpt2-large", "float32", "inference", "180"),
            #("gpt2-large", "float16", "train", "28"),
            #("gpt2-large", "float16", "inference", "364"),
            #("gpt2-xl", "float32", "train", "6"),
            #("gpt2-xl", "float32", "inference", "108"),
            #("gpt2-xl", "float16", "train", "15"),
            #("gpt2-xl", "float16", "inference", "228"),
        ]
    tests = [ ("gpt2-xl", "float32", "train", "6") ]
    tests = [
            ("resnet50", "float32", "train", "850"),
            ("resnet50", "float16", "train", "1600"),
            ("resnet50", "float32", "inference", "5500"),
            ("resnet50", "float16", "inference", "5500"),
            ("mobilenet_v2", "float32", "train", "850"),
            ("mobilenet_v2", "float16", "train", "1600"),
            ("mobilenet_v2", "float32", "inference", "1700"),
            ("mobilenet_v2", "float16", "inference", "1700"),
    ]

    tests = [
            ("resnet152", "float32", "train", "350"),
            ("resnet152", "float16", "train", "750"),
            ("resnet152", "float32", "inference", "5200"),
            ("resnet152", "float16", "inference", "10500"),
    ]

    tests = [
            ("vgg16", "float32", "train", "650"),
            ("vgg16", "float16", "train", "1400"),
            ("vgg16", "float32", "inference", "2300"),
            ("vgg16", "float16", "train", "6000"),
    ]



    """
    tests = [# (model_name, data_type, train/inference, batch_size)
            ("gpt2-medium", "float32", "inference", "314"),
            ("gpt2-medium", "float16", "inference", "630"),
            ("gpt2-medium", "fp8_hybrid", "inference", "314"),
            ("gpt2-medium", "fp8_e4m3", "inference", "314"),
            ("gpt2-medium", "fp8_hybrid", "inference", "630"),
            ("gpt2-medium", "fp8_e4m3", "inference", "630"),
    ]
    latest_tests = [# (model_name, data_type, train/inference, batch_size)
            ("gpt2-small", "float32", "inference", "682"),
            #("gpt2-small", "float16", "inference", "1400"),
            #("gpt2-small", "fp8_hybrid", "inference", "682"),
            #("gpt2-small", "fp8_e4m3", "inference", "682"),
            #("gpt2-small", "fp8_hybrid", "inference", "1400"),
            #("gpt2-small", "fp8_e4m3", "inference", "1400"),
            #("gpt2-medium", "float32", "inference", "314"),
            #("gpt2-medium", "float16", "inference", "630"),
            #("gpt2-medium", "fp8_hybrid", "inference", "314"),
            #("gpt2-medium", "fp8_e4m3", "inference", "314"),
            #("gpt2-medium", "fp8_hybrid", "inference", "630"),
            #("gpt2-medium", "fp8_e4m3", "inference", "630"),
            #("gpt2-large", "float32", "inference", "174"),
            #("gpt2-large", "float16", "inference", "356"),
            #("gpt2-large", "fp8_hybrid", "inference", "174"),
            #("gpt2-large", "fp8_e4m3", "inference", "174"),
            #("gpt2-large", "fp8_hybrid", "inference", "356"),
            #("gpt2-large", "fp8_e4m3", "inference", "356"),
            #("gpt2-xl", "float32", "inference", "104"),
            #("gpt2-xl", "float16", "inference", "218"),
            #("gpt2-xl", "fp8_hybrid", "inference", "104"),
            #("gpt2-xl", "fp8_e4m3", "inference", "104"),
            #("gpt2-xl", "fp8_hybrid", "inference", "218"),
            #("gpt2-xl", "fp8_e4m3", "inference", "218"),
    ]
    fp16_tests = [# (model_name, data_type, train/inference, batch_size)
            ("gpt2-medium", "float16", "inference", "314"),
            ("gpt2-xl", "float16", "inference", "104"),
    ]

    tests = [# (model_name, data_type, train/inference, batch_size)
            #("gpt2-small", "float32", "train", "70"),
            #("gpt2-small", "float32", "inference", "682"),
            #("gpt2-small", "float16", "train", "130"),
            ("gpt2-small", "float16", "inference", "1400"),
            ("gpt2-small", "fp8_hybrid", "inference", "682"),
            ("gpt2-small", "fp8_e4m3", "inference", "682"),
            ("gpt2-small", "fp8_hybrid", "inference", "1400"),
            ("gpt2-small", "fp8_e4m3", "inference", "1400"),
            #("gpt2-small", "fp8_hybrid", "inference", "682"),
            #("gpt2-small", "fp8_e4m3", "inference", "682"),
            #("gpt2-medium", "float32", "train", "24"),
            #("gpt2-medium", "float32", "inference", "330"),
            #("gpt2-medium", "float16", "train", "48"),
            #("gpt2-medium", "float16", "inference", "512"),
            #("gpt2-medium", "fp8_hybrid", "inference", "512"),
            #("gpt2-medium", "fp8_e4m3", "inference", "512"),
            #("gpt2-large", "float32", "train", "10"),
            ("gpt2-large", "float32", "inference", "180"),
            #("gpt2-large", "float16", "train", "24"),
            ("gpt2-large", "float16", "inference", "364"),
            ("gpt2-large", "fp8_hybrid", "inference", "364"),
            ("gpt2-large", "fp8_e4m3", "inference", "364"),
            #("gpt2-xl", "float32", "train", "4"),
            ("gpt2-xl", "float32", "inference", "108"),
            #("gpt2-xl", "float16", "train", "12"),
            ("gpt2-xl", "float16", "inference", "228"),
            ("gpt2-xl", "fp8_hybrid", "inference", "228"),
            ("gpt2-xl", "fp8_e4m3", "inference", "228"),
    ]

    fp8_tests = [# (model_name, data_type, train/inference, batch_size)
            ("gpt2-small", "fp8_hybrid", "train", "130"),
            #("gpt2-small", "fp8_hybrid", "inference", "682"),
            ("gpt2-small", "fp8_e4m3", "train", "130"),
            #("gpt2-small", "fp8_e4m3", "inference", "682"),
            ("gpt2-medium", "fp8_hybrid", "train", "48"),
            #("gpt2-medium", "fp8_hybrid", "inference", "330"),
            ("gpt2-medium", "fp8_e4m3", "train", "48"),
            #("gpt2-medium", "fp8_e4m3", "inference", "512"),
            ("gpt2-large", "fp8_hybrid", "train", "24"),
            #("gpt2-large", "fp8_hybrid", "inference", "180"),
            ("gpt2-large", "fp8_e4m3", "train", "24"),
            #("gpt2-large", "fp8_e4m3", "inference", "364"),
            ("gpt2-xl", "fp8_hybrid", "train", "12"),
            #("gpt2-xl", "fp8_hybrid", "inference", "108"),
            ("gpt2-xl", "fp8_e4m3", "train", "12"),
            #("gpt2-xl", "fp8_e4m3", "inference", "228"),
    ]
    """
    for test_case in tests:
        subprocess.run("rm /mnt/scalers/exps/new_sb/gpt.yaml", shell=True)
        config = f"""
    # SuperBench Config
    version: v0.8
    superbench:
      enable: null
      monitor:
        enable: true
        sample_duration: 1
        sample_interval: 10
      benchmarks:
        gpt_models:
          enable: true
          modes:
            - name: torch.distributed
              proc_num: 8
              node_num: 1
          frameworks: ['pytorch']
          models: ['{test_case[0]}']
          parameters:
            duration: 0
            num_warmup: 16
            num_steps: 100
            batch_size: {int(test_case[3])}
            precision: ['{test_case[1]}']
            model_action: ['{test_case[2]}']
        """
        subprocess.run(f"echo \"{config}\"> /mnt/scalers/exps/new_sb/gpt.yaml", shell=True)
        run_str = f"sb run --docker-image superbench/superbench:v0.8.0-cuda12.1 -f /mnt/scalers/exps/new_sb/local.ini -c /mnt/scalers/exps/new_sb/gpt.yaml --output-dir outputs/{test_case[0]}_{test_case[1]}_{test_case[2]}_{test_case[3]}_10000_steps"
        subprocess.run(run_str, shell=True)
        time.sleep(60)
    return tests

if __name__ == "__main__":
    tests = run_benchmark()
