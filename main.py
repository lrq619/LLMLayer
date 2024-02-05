import utils
import re
import config

def filter_metric_table(out):
    regex = r"\s+(\w+\.sum)\s+(\w+)\s+([\d.]+)"
    matches = re.findall(regex, out)
    kernel_metrics = []
    for match in matches:
        bytes_accessed = float(match[2])
        unit = 1
        if match[1] == "Kbyte":
            unit = 1024 
        elif match[1] == "Mbyte":
            unit = 1024**2
        elif match[1] == "Gbyte":
            unit = 1024**3
        elif match[1] == "Tbyte":
            unit = 1024**4

        bytes_accessed *= unit
        if match[0] == "dram__bytes_read.sum":
            kernel_metrics.append({"read":bytes_accessed})
        else:
            kernel_metrics[-1]["write"] = bytes_accessed
    return kernel_metrics

def filter_flops_latency(out):
    total_flops_pattern = r"###total flops: ([\d.]+) ops"
    latency_pattern = r"###latency: ([\d.]+) ms"
    latency_match = re.findall(latency_pattern, out)    
    total_flops_match = re.findall(total_flops_pattern, out)
    latency_value = latency_match[-1] if latency_match else None
    total_flops_value = total_flops_match[-1] if total_flops_match else None

    return float(latency_value), float(total_flops_value)

def total_mem_accessed(kernel_metrics):
    kernels = kernel_metrics[-7:]
    total_mem_read = 0
    total_mem_write = 0
    for k in kernels:
        total_mem_read += k["read"]
        total_mem_write += k["write"]

    return total_mem_read + total_mem_write


        

def profile(token_len, hardware="a100"):
    print(f"---token length: {token_len}---")
    python_path = utils.exec("which python")
    pwd = utils.exec("pwd")

    cmd = f"{python_path} {pwd}/decoder.py --sqt {token_len}"
    out = utils.exec(cmd)

    practical_latency, total_flops = filter_flops_latency(out)
    # practical latency is in ms

    cmd = f"ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum {python_path} {pwd}/decoder.py --sqt {token_len}"
    out = utils.exec(cmd)
    kernel_metrics = filter_metric_table(out)
    print(f"Collected {len(kernel_metrics)} kernels")
    total_mem = total_mem_accessed(kernel_metrics)
    print(f"practical_latency: {practical_latency:.1f} ms. total flops: {total_flops/(1024**3):.1f} Gops. total memory accessed: {total_mem/(1024**3):.1f} GB")

    compute = config.HARDWARE_CONFIG[hardware]["compute"] * (1024**4)
    mem_bw = config.HARDWARE_CONFIG[hardware]["mem_bw"] * (1024**3)
    estimated_latency = (total_flops / compute + total_mem / mem_bw) * 1000 #ms
    print(f"estimated_latency: {estimated_latency:.1f} ms under {compute/(1024**4):.1f} TFLOPS and {mem_bw/(1024**3):.1f} GB/s")
    error = (practical_latency - estimated_latency) / (practical_latency)
    print(f"error: {error*100:.1f}%")
    return practical_latency, estimated_latency, error
    
def main():
    import ctp
    run = ctp.append_run("decoder_latency")
    hardware = "v100"
    for token_len in range(1,257,256):
        p_latency, e_latency, error = profile(token_len,hardware)
        run.collect("p_latency", p_latency)
        run.collect("e_latency", e_latency)
        run.collect("error", error)
    run.stop_collect()


if __name__ == '__main__':
    main()