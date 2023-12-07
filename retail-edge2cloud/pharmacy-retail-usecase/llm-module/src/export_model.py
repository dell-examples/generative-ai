# Created by Scalers AI for Dell Inc.
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

# Export HF Llama 2 7b Chat model into OpenVINO format.
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = OVModelForCausalLM.from_pretrained(model_id, export=True)

# Saves exported model and tokenizer into a local folder.
model.save_pretrained("llama-2-7b-chat-ov")
tokenizer.save_pretrained("llama-2-7b-chat-ov")
