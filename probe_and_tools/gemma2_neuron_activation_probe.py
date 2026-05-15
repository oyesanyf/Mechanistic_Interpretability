import torch
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-2-2b-it" 

print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.bfloat16, 
    device_map="auto" 
)

prompt = "Identify the core security threat."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

layer_activations = {}

# We use a PRE-hook on the down_proj layer. 
# The input to down_proj is the exact 9,216-dimensional intermediate activation.
def get_intermediate_activation(layer_index):
    def hook(module, input):
        # input is a tuple. The first element is the tensor we want.
        layer_activations[layer_index] = input[0].detach()
    return hook

handles = []
num_layers = len(model.model.layers)

print(f"Registering pre-hooks on the internal intermediate layers...")
for i in range(num_layers):
    # Hooking the input of the final down-projection
    handle = model.model.layers[i].mlp.down_proj.register_forward_pre_hook(get_intermediate_activation(i))
    handles.append(handle)

print("Running forward pass...")
with torch.no_grad():
    model(**inputs)

# Clean up hooks
for handle in handles:
    handle.remove()

csv_filename = "all_intermediate_neurons.csv"
print(f"\nSaving ALL internal neuron activations to {csv_filename}...")

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Layer", "Neuron Index", "Activation Magnitude"])
    
    total_rows = 0
    for i in range(num_layers):
        last_token_activations = layer_activations[i][0, -1, :]
        
        # Iterate through all 9,216 neurons in this layer
        for neuron_idx, activation_val in enumerate(last_token_activations):
            writer.writerow([i, neuron_idx, f"{activation_val.item():.6f}"])
            total_rows += 1

print(f"Done! Successfully saved {total_rows} rows to {csv_filename}.")