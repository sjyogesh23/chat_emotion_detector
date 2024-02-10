import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import time

pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    probabilities = pipe_lr.decision_function([docx])[0]
    max_prob_index = np.argmax(probabilities)
    predicted_emotion = results[0]
    confidence = probabilities[max_prob_index]
    return predicted_emotion, confidence


def main():
    st.title("Chat emotion detection")
    st.markdown('Done By: [Yogesh SJ](https://www.linkedin.com/in/yogeshsj/)')

    with st.form(key='my_form'):
        raw_text = st.text_area("Enter Here")
        submit_text = st.form_submit_button(label='Submit')

    
    if submit_text:
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i+1)
        st.balloons()
        
        prediction, confidence = predict_emotions(raw_text)
        confidence_percentage = max(0, min(100, confidence * 100))

        
        emoji_icon = emotions_emoji_dict[prediction]
        st.info("{}:{}".format(prediction, emoji_icon))
        st.success("Confidence: {:.2f}%".format(confidence_percentage))
        
        probabilities = pipe_lr.decision_function([raw_text])[0]
        scaled_probabilities = (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities)) * 100
        proba_data = [{'emotions': emotion, 'probability': probability} for emotion, probability in zip(emotions_emoji_dict.keys(), scaled_probabilities)]
        proba_df_clean = pd.DataFrame(proba_data)
        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
        st.altair_chart(fig, use_container_width=True)

        progress.empty()

        
if __name__ == '__main__':
    main()
