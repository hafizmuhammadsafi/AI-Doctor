import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from unsloth import FastLanguageModel
import re

# =============================================================================
# Model Loading (Cached)
# =============================================================================
@st.cache_resource
def load_models():
    # Load Diagnosis Model
    diagnosis_llm = Llama(
        model_path="/content/unsloth.Q4_K_M.gguf",
        n_ctx=2048,
        n_gpu_layers=15,
        n_threads=8,
        verbose=False
    )

    # Load Prescription Model
    prescription_model, prescription_tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(prescription_model)

    return diagnosis_llm, prescription_model, prescription_tokenizer

# =============================================================================
# Chat Functions
# =============================================================================
def get_diagnosis(symptoms, llm):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    As an AI doctor, analyze these symptoms and respond in this format:

    **Detected Condition**: [Medical Name]
    **Key Symptoms**: [Matching Symptoms]
    **Confidence**: [High/Medium/Low]

    Symptoms: {symptoms}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>"""

    response = llm.create_completion(
        prompt=prompt,
        max_tokens=256,
        temperature=0.2,
        stop=["<|eot_id|>"]
    )
    return response['choices'][0]['text']

def get_treatment(disease, model, tokenizer):
    prompt = f"""<|start_header_id|>system<|end_header_id|>
    As a medical AI, provide treatment guidance for {disease}:
    1. Recommended specialists
    2. Essential tests
    3. Treatment plan
    4. Precautions<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>"""

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# =============================================================================
# Streamlit UI
# =============================================================================
def main():
    st.title("AI Doctor Chatbot 🩺")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Describe your symptoms..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Load models (cached)
        diagnosis_llm, rx_model, rx_tokenizer = load_models()

        with st.spinner("Analyzing symptoms..."):
            try:
                # Get diagnosis
                diagnosis = get_diagnosis(prompt, diagnosis_llm)

                # Get treatment plan
                disease = re.search(r"\*\*Detected Condition\*\*: (.+)", diagnosis)
                disease = disease.group(1) if disease else "Unknown Condition"
                treatment = get_treatment(disease, rx_model, rx_tokenizer)

                # Format response
                response = f"""
                **Diagnosis Report**
                {diagnosis}

                **Treatment Plan**
                {treatment.split('<|eot_id|>')[-1].strip()}
                """

            except Exception as e:
                response = f"Error processing request: {str(e)}"

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
