from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load model and tokenizer
model_name = "microsoft/phi-2"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  
    llm_int8_enable_fp32_cpu_offload=True  
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define prompt
prompt = "Once upon a time,"

# Tokenize input
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# Generate text
output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:", generated_text)
