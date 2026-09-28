import os
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import streamlit as st

# ==========================================
# STEP 1: NLTK Data & Setup
# ==========================================

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Preprocessing tools
exclude = string.punctuation

stop_words = set(stopwords.words('english'))

# Keep negation words for correct sentiment prediction
stop_words -= {'not', 'no', 'nor', 'never'}

english_words = set(words.words())
lemmatizer = WordNetLemmatizer()

# ==========================================
# STEP 2: Text Preprocessing Function
# ==========================================

def clean_and_preprocess(text):
    # 1. Lowercase
    text = str(text).lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', exclude))

    # 3. Tokenize words
    tokens = word_tokenize(text)

    # 4. Remove stopwords and apply lemmatization
    cleaned_words = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and w.isalpha()
    ]

    return ' '.join(cleaned_words), tokens

# ==========================================
# STEP 3: Genuine Review Check (Validation)
# ==========================================

def is_genuine_review(raw_text, tokens, input_vector, vectorizer):

    # Rule 1: Minimum character length check
    if len(raw_text.strip()) < 5:
        return False, "Review is too short. Please enter a complete sentence."

    # Rule 2: Check for excessive character repetition
    if re.search(r'(.)\1{4,}', raw_text):
        return False, "Invalid characters detected. Please enter a genuine review."

    # Rule 3: Check if TF-IDF recognizes any word
    if input_vector.nnz == 0:
        return False, "Please enter a genuine review. Unable to recognize any valid words."

    # Rule 4: Match ratio check against English dictionary or vocabulary
    alpha_tokens = [
        w.lower() for w in tokens
        if w.isalpha() and len(w) > 1
    ]

    if not alpha_tokens:
        return False, "Please enter a genuine review."

    valid_words = [
        w for w in alpha_tokens
        if (w in english_words or w in vectorizer.vocabulary_)
    ]

    match_ratio = len(valid_words) / len(alpha_tokens)

    if match_ratio < 0.3:
        return False, "Gibberish or random text detected. Please enter a genuine review."

    return True, ""

# ==========================================
# STEP 4: Load Trained Models
# ==========================================

@st.cache_resource
def load_all_artifacts():

    with open('vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)

    with open('sentiment_model.pkl', 'rb') as f:
        sent_m = pickle.load(f)

    with open('recommendation_model.pkl', 'rb') as f:
        rec_m = pickle.load(f)

    return vec, sent_m, rec_m

# ==========================================
# STEP 5: Streamlit Web UI
# ==========================================

st.set_page_config(
    page_title="Clothing Review Analyzer",
    page_icon="👗",
    layout="centered"
)

st.title("👗 Women Clothing Review Analyzer")

st.write(
    "Enter your clothing review below. "
    "The model will predict Sentiment and Recommendation."
)

# Check if model files exist
if not (
    os.path.exists('vectorizer.pkl')
    and os.path.exists('sentiment_model.pkl')
    and os.path.exists('recommendation_model.pkl')
):
    st.error(
        "Model files (.pkl) not found! "
        "Please run the training notebook to generate them first."
    )
    st.stop()

vectorizer, sentiment_model, rec_model = load_all_artifacts()

# ==========================================
# STEP 6: User Input Box
# ==========================================

user_review = st.text_area(
    "Enter Customer Review:",
    placeholder="Example: I loved this dress, the fabric is soft and fits perfectly!",
    height=120
)

# ==========================================
# STEP 7: Prediction
# ==========================================

if st.button("Analyze & Predict", type="primary"):

    if not user_review.strip():
        st.warning("Please enter a review first!")

    else:
        # Preprocessing & Vectorization
        processed_text, tokens = clean_and_preprocess(user_review)

        input_vector = vectorizer.transform([processed_text])

        # Validation check
        valid, message = is_genuine_review(
            user_review,
            tokens,
            input_vector,
            vectorizer
        )

        if not valid:
            st.error(f"⚠️ {message}")

        else:
            # 1. Sentiment Prediction
            sent_pred = sentiment_model.predict(input_vector)[0]

            sent_conf = (
                max(sentiment_model.predict_proba(input_vector)[0]) * 100
            )

            # 2. Recommendation Prediction
            rec_pred = rec_model.predict(input_vector)[0]

            rec_conf = (
                rec_model.predict_proba(input_vector)[0][1] * 100
            )

            # ==========================================
            # STEP 8: Display Results
            # ==========================================

            st.markdown("---")

            col1, col2 = st.columns(2)

            # Sentiment Analysis
            with col1:

                st.subheader("Sentiment Analysis")

                if sent_pred == 1:
                    st.success(
                        f"😊 **Positive**\n\nConfidence: {sent_conf:.1f}%"
                    )

                elif sent_pred == 2:
                    st.info(
                        f"😐 **Neutral**\n\nConfidence: {sent_conf:.1f}%"
                    )

                else:
                    st.error(
                        f"😞 **Negative**\n\nConfidence: {sent_conf:.1f}%"
                    )

            # Recommendation Status
            with col2:

                st.subheader("Recommendation Status")

                if rec_pred == 1:
                    st.success(
                        f"👍 **Recommended (YES)**\n\n"
                        f"Likelihood: {rec_conf:.1f}%"
                    )

                else:
                    st.error(
                        f"👎 **Not Recommended (NO)**\n\n"
                        f"Likelihood: {100 - rec_conf:.1f}%"
                    )