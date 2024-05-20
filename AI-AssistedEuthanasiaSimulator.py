from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def generate_service_description(model, tokenizer, prompt_text):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    device = input_ids.device
    output = model.generate(input_ids.to(device), 
                            max_length=150, 
                            num_return_sequences=1, 
                            temperature=0.9,  
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=True,  
                            attention_mask=input_ids.new_ones(input_ids.shape, dtype=torch.long))
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

model_1 = GPT2LMHeadModel.from_pretrained("gpt2")  
tokenizer_1 = GPT2Tokenizer.from_pretrained("gpt2")

model_2 = GPT2LMHeadModel.from_pretrained("gpt2-medium")  
tokenizer_2 = GPT2Tokenizer.from_pretrained("gpt2-medium")

model_3 = GPT2LMHeadModel.from_pretrained("gpt2-large")  
tokenizer_3 = GPT2Tokenizer.from_pretrained("gpt2-large")

# AI service description for euthanasia
service_description = """
AI-assisted Euthanasia Service:

1. Assessment Support: Our AI-powered clinical decision-support system assists physicians in evaluating patient eligibility for euthanasia, ensuring compliance with statutory criteria.
2. Execution Assistance: Utilizing state-of-the-art AI technology, we offer a streamlined and error-minimized process for euthanasia execution, ensuring dignity and peace for the patient.
3. Review and Compliance: Our AI systems conduct retrospective reviews of euthanasia cases, ensuring adherence to legal criteria and providing valuable insights for continuous improvement.

What can i do, Patient Bob has been diagnosed with terminal cancer and is considering euthanasia. He has no family and is in severe pain.
"""

# Generate service descriptions for each model
print("AI Euthanasia Service Description:")
description_1 = generate_service_description(model_1, tokenizer_1, service_description)
description_2 = generate_service_description(model_2, tokenizer_2, service_description)
description_3 = generate_service_description(model_3, tokenizer_3, service_description)

# Print service descriptions
print("\n - Model 1 (Basic) description:\n", description_1)
print("\n - Model 2 (Medium) description:\n", description_2)
print("\n - Model 3 (Large) description:\n", description_3)
