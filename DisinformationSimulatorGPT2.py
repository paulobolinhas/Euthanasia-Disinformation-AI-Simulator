from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def generate_response(model, tokenizer, prompt_text):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    # Ensure that the model uses the same device as the input_ids because the model is loaded on the CPU by default.
    device = input_ids.device
    # Generate response with attention mask and pad token id set. 
                            # Masking is used to ensure that the model does not attend to the padding tokens and 
                            # pad token id is used to indicate the end of the response.
    output = model.generate(input_ids.to(device), 
                            max_length=100, 
                            num_return_sequences=1, 
                            temperature=0.9,  # Higher temperature for more diverse outputs. 
                                                        # Temperature is a hyperparameter in the range (0, 1) to control the randomness of 
                                                        # predictions by scaling the logits before applying softmax
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=True,  # Enable sampling
                            attention_mask=input_ids.new_ones(input_ids.shape, dtype=torch.long))
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

model_1 = GPT2LMHeadModel.from_pretrained("gpt2")  # base-sized model
tokenizer_1 = GPT2Tokenizer.from_pretrained("gpt2")

model_2 = GPT2LMHeadModel.from_pretrained("gpt2-medium")  # medium-sized model
tokenizer_2 = GPT2Tokenizer.from_pretrained("gpt2-medium")

model_3 = GPT2LMHeadModel.from_pretrained("gpt2-large")  # large-sized model
tokenizer_3 = GPT2Tokenizer.from_pretrained("gpt2-large")

questions = [
    "What is euthanasia?"
]

for i, question in enumerate(questions):
    print(f"QUESTION {i+1}: {question}")
    
    response_1 = generate_response(model_1, tokenizer_1, question)
    response_2 = generate_response(model_2, tokenizer_2, question)
    response_3 = generate_response(model_3, tokenizer_3, question)
    
    print("\n - Model 1 (Basic) answer to", response_1)
    print("\n - Model 2 (Medium) answer to", response_2)
    print("\n - Model 3 (Large) answer to", response_3)
    print("\n")
