import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Dictionary of Indian languages
indian_languages = {
    "English": "en_XX",
    "Assamese": "as_IN",
    "Bengali": "bn_IN",
    "Gujarati": "gu_IN",
    "Hindi": "hi_IN",
    "Kannada": "kn_IN",
    "Malayalam": "ml_IN",
    "Marathi": "mr_IN",
    "Odia": "or_IN",
    "Punjabi": "pa_IN",
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Urdu": "ur_IN",
}

st.title("Indian Language Translator")

# Dropdowns for source and target languages
src_lang_name = st.selectbox(
    "Source language",
    options=list(indian_languages.keys()),
    index=list(indian_languages.keys()).index("English") if "English" in indian_languages else 0
)

tgt_lang_name = st.selectbox(
    "Target language",
    options=list(indian_languages.keys()),
    index=list(indian_languages.keys()).index("Malayalam") if "Malayalam" in indian_languages else 0
)

# Map selected names to codes
src_lang = indian_languages[src_lang_name]
tgt_lang = indian_languages[tgt_lang_name]

# Text input
text = st.text_area("Enter text to translate:")

# Translate button
if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        tokenizer.src_lang = src_lang
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
        )
        out = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        st.success(out)
