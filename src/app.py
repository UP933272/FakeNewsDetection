import streamlit as st
import pickle

st.set_page_config(layout="wide")
st.markdown("""
<style>
div.stButton > button {
    background-color: #001f3f;  /* navy blue */
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #003366;  /* lighter navy on hover */
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Load trained model and vectorizer
model = pickle.load(open("../models/model.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))

st.title("Fake News Detection Tool")

# Create 3 columns
left_col, middle_col, right_col = st.columns([1.2, 1, 1])

with left_col:
    st.subheader("Insert Text")

    user_input = st.text_area(
        "Enter text:",
        height=400
    )

    st.write("")  # spacing

    # NEW: row for button alignment
    col1, col2 = st.columns([5, 1])

    with col2:
        scan = st.button("Scan", use_container_width=True)

with middle_col:
    st.subheader("Scoring Information")
    st.write("**0–2:** Highly likely fake news")
    st.write("**3–4:** Likely fake news")
    st.write("**5:** Uncertain")
    st.write("**6–7:** Likely real news")
    st.write("**8–10:** Highly likely real news")
    st.write("")
    st.write(
        "This score is based on the machine learning model using Logical Regression "
        "of the text content using a news content based fake news detection approach."
    )

with right_col:
    st.markdown(
    "<h2 style='text-align: center;'>Fake News Score</h2>",
    unsafe_allow_html=True
)

    # Placeholder until scan runs
    score_placeholder = st.empty()
    result_placeholder = st.empty()

    if scan:
        if user_input.strip() == "":
            result_placeholder.warning("Please enter some text before scanning.")
        else:
            text_vectorized = vectorizer.transform([user_input])

            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]

            # probability of class 1 (real)
            real_prob = probabilities[1]
            score = round(real_prob * 10)

            if score <= 3:
                colour = "red"
            elif score <= 6:
                colour = "orange"
            else:
                colour = "green"

            score_placeholder.markdown(
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
                result_placeholder.error("Result: Fake News ❌")
            else:
                result_placeholder.success("Result: Real News ✅")