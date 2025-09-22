import gradio as gr
from fastai.vision.all import *

# Load trained model
model = load_learner("models/vgg19_bn_v2.pkl")

def recognize_landmark(image):
    pred, idx, probs = model.predict(image)
    
    # Sort probabilities (descending) and take top 5
    top_probs = sorted(
        zip(model.dls.vocab, map(float, probs)),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    # Convert to dict for Gradio Label
    label_dict = {cls: float(prob) for cls, prob in top_probs}
    top_pred = pred
    return label_dict, top_pred


with gr.Blocks() as demo:
    
    gr.Markdown(
        """
        # 🏛️ Popular Bangladeshi Landmarks Recognizer  
        Upload an image of a **Bangladeshi landmark** and get the **Top 5 predictions**.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📸 Upload Landmark Image")
            predict_btn = gr.Button("🔍 Classify")
        
        with gr.Column():
            gr.Markdown("### 🔮 Prediction Results")
            label_output = gr.Label(label="Top 5 Class Probabilities")
            top_pred_text = gr.Textbox(label="Most Likely Landmark", interactive=False)

    # Example Images
    examples = [
        ["assets/images/test_image1.jpg"],
        ["assets/images/test_image2.jpg"],
        ["assets/images/test_image3.jpg"],
        ["assets/images/test_image4.jpg"],
        ["assets/images/test_image5.jpg"],
        ["assets/images/test_image6.jpg"],
        ["assets/images/test_image7.jpg"],
        ["assets/images/test_image8.jpg"],
        ["assets/images/test_image9.jpg"],
        ["assets/images/test_image10.jpg"],
        ["assets/images/test_image11.jpg"],
        ["assets/images/test_image12.jpg"],
        ["assets/images/test_image13.jpg"],
        ["assets/images/test_image14.jpg"],
        ["assets/images/test_image15.jpeg"],
        ["assets/images/test_image16.jpg"]
    ]
    gr.Examples(examples=examples, inputs=image_input)

    # Button Action
    predict_btn.click(
        fn=recognize_landmark,
        inputs=image_input,
        outputs=[label_output, top_pred_text]
    )

demo.launch(share=True)
