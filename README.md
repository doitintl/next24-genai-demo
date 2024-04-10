# next24-genai-demo
Gen AI Demo for Google Next'24. This is a RAG system with Agents to retrieve project specific cost and combine it with DoiT blog posts to deliver a analysis on how to reduce cost.


### cost_nalyzer.py

This is the most comprehensive example of using Agents to retrieve data from a AlloyDB (vector database), and customer data from AlloyDB (relational database). The model powering the analysis is Gemini Pro. The code also uses Natural Language API to evaluate the sentiment of the responses. The entire script was based on Vertex AI SDK

### cost_analyzer_ui.py

The application leverages streamlit to create a UI and using Gemini Pro to create recomendations based on the content retrieved from the databases.

### langchain.ipynb

This is a small script that uses langchain to create a very strict flow for the llm to produce results.

### load_data.ipynb

Helper script to read text files from Google Cloud Storage, embed the text and save the vector into AlloyDB

### testing_pipeline.ipynb

Extra script to showcase a pipeline that will read a set of defined prompts to evaluate the output based on a specific model. The results are saved in google cloud storage.
