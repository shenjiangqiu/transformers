# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

chat = [
    r'In Java, I want to replace string like "This is a new {object} at {place}" with a Map, {object: "student", "point 3, 4"}, and get a result "This is a new student at point 3, 4". How can I do?'
]

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch_device)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

# add the EOS token as PAD token to avoid warnings
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", pad_token_id=tokenizer.eos_token_id).to(
    torch_device
)

# encode context the generation is conditioned on
model_inputs = tokenizer(f"User: {chat[0]}", return_tensors="pt").to(torch_device)

# generate 40 new tokens
beam_outputs = model.generate(
    **model_inputs,
    max_new_tokens=128,
    num_beams=5,
    early_stopping=True,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
    return_dict_in_generate=True,
    output_scores=True,
    # output_hidden_states=True,
    # output_attentions=True,
)

# %%
beam_outputs.keys()

# %%
len(beam_outputs["beam_indices"][1])


# %%
beam_outputs["beam_indices"][4]

# %%
# now we have 3 output sequences
print("Output:\n" + 100 * "-")
for i, beam_output in enumerate(beam_outputs["sequences"]):
    print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

# %%
beam_indices = beam_outputs["beam_indices"]
for i, beam_output in enumerate(beam_indices):
    print("{}: {}".format(i, beam_output))

import print_score_graph

print_score_graph.plot_source_tree(beam_indices.cpu(), "beam.pdf")

# %%
full_indices = beam_outputs["full_indices"]
full_indices_tensor = torch.stack(full_indices)
print(full_indices_tensor)

# %%
full_indices_tensor_permuted = full_indices_tensor.permute(1, 0)
print(full_indices_tensor_permuted)

# %%
import print_score_graph

print_score_graph.plot_tree(full_indices_tensor_permuted.cpu(), "full.pdf")

# %%
