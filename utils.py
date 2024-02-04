import torch
import torch.nn as nn
import subprocess
def linear_layer_memory_usage(linear_layer):
    """
    Calculates the memory usage of a PyTorch linear layer.
    
    Args:
    linear_layer (nn.Linear): A PyTorch linear layer.

    Returns:
    float: Memory usage of the layer in megabytes.
    """
    if not isinstance(linear_layer, nn.Linear):
        raise ValueError("The input layer must be an instance of nn.Linear")

    # Memory for weights
    weight_memory = linear_layer.weight.nelement() * linear_layer.weight.element_size()

    # Memory for bias, if it exists
    bias_memory = 0
    if linear_layer.bias is not None:
        bias_memory = linear_layer.bias.nelement() * linear_layer.bias.element_size()

    # Total memory in bytes
    total_memory_bytes = weight_memory + bias_memory

    # Convert bytes to megabytes
    total_memory_mb = total_memory_bytes / (1024 * 1024)

    return total_memory_mb

def total_tensors_size_mb(tensors):
    """
    Calculates the total memory size of a list of tensors in megabytes.

    Args:
    tensors (list of torch.Tensor): List of PyTorch tensors.

    Returns:
    float: Total memory size of the tensors in megabytes.
    """
    total_size_bytes = sum(tensor.nelement() * tensor.element_size() for tensor in tensors)
    total_size_mb = total_size_bytes / (1024 * 1024)
    return total_size_mb

def tensor_size_bytes(tensor):
    return tensor.nelement() * tensor.element_size()

def calculate_matmul_flops(tensor_a, tensor_b):
    if len(tensor_a.shape) not in [2,3]:
        raise ValueError("Tensor 'a' must be 2 or 3-dimensional.")
    if len(tensor_b.shape) not in [2, 3]:
        raise ValueError("Tensor 'b' must be 2 or 3-dimensional.")
    
    # Extracting the dimensions
    if len(tensor_a.shape) == 2:
        m, k = tensor_a.shape
        batch_size = 1
    elif len(tensor_a.shape) == 3:
        batch_size, m, k = tensor_a.shape
    n = tensor_b.shape[-1]

    if len(tensor_b.shape) == 2 and k != tensor_b.shape[0]:
        raise ValueError("Incompatible dimensions for matrix multiplication.")
    if len(tensor_b.shape) == 3:
        if tensor_b.shape[0] != batch_size or k != tensor_b.shape[1]:
            raise ValueError("Incompatible dimensions for matrix multiplication.")

    # Calculate FLOPs for one matrix multiplication and multiply by batch size
    single_matmul_flops =   m * n * (2*k-1)  # 2*m*n multiplications and 2*m*n-1 additions
    total_flops = single_matmul_flops * batch_size


    return total_flops

def exec(cmd):
    """
    Executes a shell command and returns the stdout.
    If an error occurs, it raises an error and prints the stderr.

    Args:
    cmd (str): The shell command to be executed.

    Returns:
    str: The stdout from the shell command.

    Raises:
    Exception: If the shell command execution fails.
    """
    try:
        # Execute the command and capture the output
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        if result.stdout[-1] == '\n':
            result.stdout = result.stdout[:-1]
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Print the stderr and raise an error
        print(f"Error occurred during executing cmd: {cmd}\nError msg: {e.stderr}")
        exit(-1)