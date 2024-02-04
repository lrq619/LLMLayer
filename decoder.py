import torch
import torch.nn as nn
import time
import utils
import argparse
from config import LAYER_CONFIG

# customizable parameters
model_name = "opt-30b"
bsz = 16

try:
    weight_dim = LAYER_CONFIG.get(model_name).get("weight_dim")
    num_heads = LAYER_CONFIG.get(model_name).get("num_heads")
except:
    print(f"failed to get {model_name} in the LAYER_CONFIG")

print(f"get model: {model_name} weight_dim: {weight_dim}, num_heads: {num_heads}")
head_dim = weight_dim // num_heads

device = "cuda:0"
hidden_states = torch.randn(bsz,1,weight_dim).to(device)
_, tgt_len, _ = hidden_states.size()

def _shape(tensor: torch.Tensor, seq_len: int, bsz: int):
    return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

# Define three linear layers
q_proj = nn.Linear(weight_dim, weight_dim)
k_proj = nn.Linear(weight_dim, weight_dim)
v_proj = nn.Linear(weight_dim, weight_dim)

# Manually initialize the weights and biases with random values
# For q_proj
q_proj.weight = nn.Parameter(torch.randn(weight_dim, weight_dim))
q_proj.bias = nn.Parameter(torch.randn(weight_dim))

# For k_proj
k_proj.weight = nn.Parameter(torch.randn(weight_dim, weight_dim))
k_proj.bias = nn.Parameter(torch.randn(weight_dim))

# For v_proj
v_proj.weight = nn.Parameter(torch.randn(weight_dim, weight_dim))
v_proj.bias = nn.Parameter(torch.randn(weight_dim))

q_proj = q_proj.to(device)
k_proj = k_proj.to(device)
v_proj = v_proj.to(device)

weight_size = utils.linear_layer_memory_usage(q_proj) + utils.linear_layer_memory_usage(k_proj) + utils.linear_layer_memory_usage(v_proj)
print(f"model weight size: {weight_size:.1f} MB")
def func(sqt_len):
    flops = 0
    past_key = torch.randn(bsz,num_heads,sqt_len,head_dim, device=device)
    past_val = torch.randn(bsz,num_heads,sqt_len,head_dim, device=device)
    print(f"past_key size: {utils.tensor_size_bytes(past_key)/(1024**2):.1f} MB")
    print(f"past_val size: {utils.tensor_size_bytes(past_val)/(1024**2):.1f} MB")

    start = time.time()
    query_states = q_proj(hidden_states)
    q_flops = utils.calculate_matmul_flops(hidden_states, q_proj.weight) 
    # print(f"flops for q_proj: {q_flops}, mem for q_proj: {q_mem}")
    flops += q_flops

    key_states = _shape(k_proj(hidden_states), -1, bsz)
    k_flops = utils.calculate_matmul_flops(hidden_states, k_proj.weight)
    # print(f"flops for k_proj: {k_flops}, mem for k_proj: {k_mem}")
    flops += k_flops

    value_states = _shape(v_proj(hidden_states), -1, bsz)
    v_flops = utils.calculate_matmul_flops(hidden_states, v_proj.weight)
    # print(f"flops for v_proj: {v_flops}, mem for v_proj: {v_mem}")
    flops += v_flops

    key_states = torch.cat([past_key, key_states], dim=2)
    value_states = torch.cat([past_key, value_states], dim=2)
    

    proj_shape = (bsz * num_heads, -1, head_dim)
    query_states = _shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)
    # calculate attn_weights
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    aw_flops = utils.calculate_matmul_flops(query_states, key_states.transpose(1,2))
    # print(f"flops for attn_weights: {aw_flops}, mem for attn_weights: {aw_mem}")
    flops += aw_flops



    attn_output = torch.bmm(attn_weights, value_states)
    ao_flops = utils.calculate_matmul_flops(attn_weights, value_states)
    # print(f"flops for attn_output: {ao_flops}, mem for attn_output: {ao_mem}")
    flops += ao_flops
    torch.cuda.synchronize()

    theoretical_comp = 19.5 * (10**12)#TFLOPS

    latency = time.time() - start
    print(f"sequence length: {sqt_len}, ###latency: {latency*1000:.1f} ms. ###total flops: {flops:.1f} ops, theoretical time spent on compute: {flops*1000/theoretical_comp:.1f} ms.  tensor type: {hidden_states.dtype}")
    compute_tgt = flops / latency
    print(f"compute throughput: {compute_tgt:.1f} FLOPS")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sqt', type=int ,default=512, help="token length")
    args = parser.parse_args()
    for i in range(2):
        func(args.sqt)

if __name__ == '__main__':
    main()
