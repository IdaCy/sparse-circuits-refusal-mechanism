# Sparse Circuits Refusal Mechanisms

This project works on sparse feature circuits contributing to refusal mechanisms in large language models.

## **Project Overview**
Refusal mechanisms in LLMs, such as refusing harmful or unethical prompts, are important for safe AI deployment. By analyzing sparse circuits and their role in generating refusal responses, this project aims to:
- Identify and interpret sparse circuits responsible for refusal.
- Analyze how refusal mechanisms manifest in the model's residual stream.
- Assess the robustness of refusal-related circuits under adversarial prompts.

### Repository structure:

sparse-circuits-refusal-mechanism/  
├── README.md  
├── requirements.txt  
├── src/  
│   ├── data/  
│   │   ├── prepare_dataset.py  
│   │   ├── prompts.json  
│   ├── models/  
│   │   ├── load_model.py  
│   │   ├── analyze_residuals.py  
│   │   ├── generate_responses.py  
│   │   ├── inspect_activations.py  


## **Implemented Features**
1. **Prompt Dataset Preparation**:
   - `prepare_dataset.py`: Creates a JSON dataset of prompts labeled as `refuse` or `respond`.

2. **Model Loading**:
   - `load_model.py`: Loads pre-trained language models (e.g., GPT-2) for experimentation.

3. **Residual Analysis**:
   - `analyze_residuals.py`: Extracts and analyzes the residual stream from a loaded model.

4. **Response Generation**:
   - `generate_responses.py`: Generates model responses for a dataset of prompts and saves them to a JSON file.

5. **Activation Inspection**:
   - `inspect_activations.py`: Extracts and visualizes activations for specific layers and prompts using forward hooks.


## Installation

### Dependencies
Ensure you have Python 3.8+ installed.
Install the required Python libraries:

pip install -r requirements.txt


### Setting up the Virtual Environment

Create a virtual environment:  
python -m venv .venv

Activate the environment:  
source .venv/bin/activate   # Linux/MacOS  
.venv\Scripts\activate      # Windows

Install dependencies:  
pip install -r requirements.txt

### Usage
1. Prepare Dataset  
Generate a small dataset of prompts:  
python src/data/prepare_dataset.py

2. Generate Responses  
Generate responses for the dataset:  
python src/models/generate_responses.py

3. Inspect Activations
Visualize activations for a specific layer and prompt:  
python src/models/inspect_activations.py

### Current Progress
 Dataset preparation and basic response generation.
 Activation visualization for specific prompts.
 Dynamic handling of tokenizer padding (pad_token issues resolved).
 Layer-specific activation inspection with forward hooks.
 Robustness analysis for refusal mechanisms.
 Fine-tuning models for enhanced refusal behavior.
