from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "microsoft/phi-2"

# Define 8-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Load model in 8-bit
    llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading for FP32 layers
)

# Load model with optimized settings
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",  # Automatically selects GPU/CPU
    quantization_config=quantization_config,  # Use defined quantization config
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
