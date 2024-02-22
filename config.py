LAYER_CONFIG = {
    "opt-30b": {
        "weight_dim": 7168,
        "num_heads": 56,
        "num_layers": 48
    },
    "opt-13b": {
       "weight_dim": 5120,
       "num_heads": 40,
       "num_layers": 40 
    }
}

HARDWARE_CONFIG = {
    "a100": {
        "compute": 19.5, # TFLOPS
        "mem_bw": 1400 # GFLOPS
    },
    "v100": {
        "compute": 7, # TFLOPS
        "mem_bw": 700
    }
}