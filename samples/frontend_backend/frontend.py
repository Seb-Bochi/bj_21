import os
import pandas as pd
import requests
import streamlit as st


def get_backend_url() -> str:
    """Get the URL of the backend service."""
    return os.environ.get("BACKEND", "http://127.0.0.1:8000")


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend.rstrip('/')}/classify/"
    response = requests.post(
        predict_url,
        files={"file": ("image.jpg", image, "image/jpeg")},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()

def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"][0]
            top_indices = sorted(
                range(len(probabilities)),
                key=probabilities.__getitem__,
                reverse=True,
            )[:10]

            data = {
                "Class": [f"Class {index}" for index in top_indices],
                "Probability": [probabilities[index] for index in top_indices],
            }
            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", prediction)

            # make a nice bar chart
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()