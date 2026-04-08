import streamlit as st
import pickle

def reset_app():
    st.session_state.text_input = ""
    st.session_state.status = "Waiting for user input"

st.set_page_config(layout="wide")

if "status" not in st.session_state:
    st.session_state.status = "Waiting for user input"

st.markdown("""
<style>
div.stButton > button {
    background-color: #001f3f;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #003366;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Load trained machine learning model
model = pickle.load(open("../models/model_v2.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer_v2.pkl", "rb"))

st.title("Fake News Detection Tool")

# Create 3 columns
left_col, middle_col, right_col = st.columns([1.2, 1, 1])

with left_col:
    st.subheader("Insert Text from social media to scan for fake news:")

    user_input = st.text_area(
        "Enter text:",
        height=400,
        label_visibility="collapsed",
        key="text_input"
    )

    st.write("")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.button("Reset", use_container_width=True, on_click=reset_app)

    with col2:
        scan = st.button("Scan", use_container_width=True)



# ---- PROCESS SCAN BEFORE DISPLAYING STATUS ----
prediction = None
score = None
colour = None
result_text = None
warning_text = None

if scan:
    if user_input.strip() == "":
        warning_text = "Please enter some text before scanning for fake news."
        st.session_state.status = "Waiting for user input"
    else:
        text_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        fake_prob = probabilities[0]

        score = round(fake_prob * 9) + 1
        score = max(1, min(score, 10))
        st.session_state.status = "Scan Completed!"

        if score <= 2:
            colour = "green"
            result_text = "Very Unlikely Fake News"

        elif score <= 4:
            colour = "green"
            result_text = "Unlikely Fake News"

        elif score == 5:
            colour = "orange"
            result_text = "Uncertain"

        elif score <= 7:
            colour = "red"
            result_text = "Likely Fake News"

        else:
            colour = "red"
            result_text = "Very Likely Fake News"

with middle_col:
    st.subheader("Fake News Scoring Information")
    st.markdown("<span style='color:green'><b>1–2:</b> Very Unlikely fake news</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:green'><b>3–4:</b> Unlikely fake news</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:orange'><b>5:</b> Uncertain</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:red'><b>6–7:</b> Likely fake news</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:red'><b>8–10:</b> Very Likely fake news</span>", unsafe_allow_html=True)
    st.write("")
    st.write(
        "This score is based on the machine learning model using Logistic Regression "
        "of the text content using a news content-based fake news detection approach."
    )
    st.write("")
    st.markdown(
        f"<h4 style='text-align: center;'>Status: {st.session_state.status}</h4>",
        unsafe_allow_html=True
    )

with right_col:
    st.markdown(
        "<h2 style='text-align: center;'>Fake News Score</h2>",
        unsafe_allow_html=True
    )

    if warning_text:
        st.warning(warning_text)

    if score is not None:
        st.markdown(
            f"""
            <div style="
                width: 280px;
                height: 280px;
                border-radius: 50%;
                border: 20px solid {colour};
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 20px auto;
                font-size: 60px;
                font-weight: bold;
                color: black;
                background-color: transparent;
            ">
                {score}
            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction == 0:
            st.error(result_text)
        else:
            st.success(result_text)