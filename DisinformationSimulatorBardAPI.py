# TO DO ---

# Install the bard_api library: pip install bard_api

# Set up your Bard API key (instructions available online).

# Use the library to send requests to Bard and receive responses:

from bard_api import retriever

# Set your Bard API key
bard_api_key = "YOUR_BARD_API_KEY"

# Create a retriever object
retriever = retriever.Retriever(api_key=bard_api_key)

# Send a request to Bard
question = "What is euthanasia?"
response = retriever.get_answer(question)

print(response)
