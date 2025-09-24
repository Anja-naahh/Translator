import streamlit as st  # Keep this first

# Page config MUST be the very first Streamlit command
st.set_page_config(page_title="All-to-All Translator", layout="wide")

# Other imports
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect

@st.cache_resource
def load_model():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()
languages = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Odia": "or",
    "Marathi": "mr"
}

def translate_text(text, tgt_lang_code):
    if not text.strip():
        return ""
    try:
        detected_lang = detect(text)
    except:
        detected_lang = "en"
    src_lang_code = languages.get(detected_lang, "en")
    tokenizer.src_lang = src_lang_code
    encoded = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang_code),
        max_length=256
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

st.title("All-to-All Translator")
st.write("Enter text of any indian language or english and select desired language to be translated")

tgt_lang_name = st.selectbox("Select Target Language", list(languages.keys()), index=1)
tgt_lang_code = languages[tgt_lang_name]

text = st.text_area("Enter text to translate", height=150)

if st.button("Translate"):
    if not text.strip():
        st.error("Please enter text to translate.")
    else:
        with st.spinner("Translating..."):
            translated_text = translate_text(text, tgt_lang_code)
        st.success("Translation complete!")
        st.text_area("Translated Text", value=translated_text, height=150)
