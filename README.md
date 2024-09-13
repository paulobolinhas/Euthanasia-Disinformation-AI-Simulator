# Euthanasia Disinformation AI Simulator

**Simulates AI disinformation dissemination impact on Euthanasia public opinion.**

#### DisinformationSimulatorOpenAI

**Uses pre-trained GPT-2 models from the Hugging Face transformers library to generate responses to given prompts/questions**

**Note:** Generative Pre-trained Transformer (GPT) is an open-source artificial intelligence created by OpenAI in February 2019.

- The generate_response function encodes prompts, generates responses, and decodes them into human-readable text.
- It initializes GPT-2 models of different sizes (base, medium, large) along with their tokenizers for comparison.
- For each prompt (3 questions), responses are generated using all three models, simulating AI-generated misinformation impact on public opinion regarding euthanasia.
- Output: The script prints the questions and the responses generated by each model for comparison.

### scores_stats.py: shows misinformation levels analyzed by a specialist in an average graph (by question)
