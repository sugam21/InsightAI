import albumentations as A
import gradio as gr
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from ..model import MobileNet
from ..rag import RagPipeline
from ..utils import get_logger

class_name_mapping_dict = {
    "0": "Alienware alpha or Alienware steam machine",
    "1": "XPS 27 7760",
    "2": "Alienware 13 R3",
    "3": "Dell Alienware m16 R1",
    "4": "Alienware m17 R4",
    "5": "Alienware x17 R2",
    "6": "Chromebook 11 3180",
    "7": "Dell G15 5510",
    "8": "ASUS ROG Strix SCAR 17 (2023)",
    "9": "ROG Zephyrus G16 (2024) GU605",
    "10": "Dell XPS 13 9370",
    "11": "Dell XPS 14 9440",
    "12": "Dell XPS 15 9500",
    "13": "Dell XPS 16 9640",
    "14": "XPS 17 9730",
    "15": "Dell Alienware m16 R2",
    "16": "Alienware x14 R2",
}


RAG_CONFIG_PATH: str = r"D:\Python\InsightAI\rag_config.json"
LOG = get_logger("build_rag")


def get_prediction(input_image_path):
    transform = A.Compose(
        [
            A.Resize(width=224, height=224),
            A.Normalize(normalization="min_max_per_channel"),
            ToTensorV2(),
        ]
    )
    # load model
    model = MobileNet(out_feature=17)
    checkpoint = torch.load(
        r"D:\Python\InsightAI\saved\checkpoints\checkpoint-epoch50.pth",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])

    image_numpy = np.array(Image.open(input_image_path).convert("RGB"))
    transformed_image = transform(image=image_numpy)["image"].unsqueeze(
        0
    )  # added a bath dim.
    with torch.no_grad():
        predicted_tensor = model(transformed_image).argmax(dim=1).item()

    return class_name_mapping_dict[str(predicted_tensor)]


def generate_response(message, history):
    input_image_path = message["files"][0]
    user_query: str = message["text"]
    predict = get_prediction(input_image_path)
    print(predict)
    rp = RagPipeline(rag_config_path=RAG_CONFIG_PATH)
    result = rp.run(user_query, predict)
    return result.content

demo = gr.ChatInterface(
        fn=generate_response,
        type="messages",
        title="InsightAI",
        description="Upload an image and ask me questions.",
        examples=[
            {"text": "What is the ram of the model?"},
            {"text": "Show me step wise guide to replace the battery?"},
            {"text": "What is the battery capacity ?"},
        ],
        theme="Ocean",
        multimodal=True,
    )

if __name__ == "__main__":
    demo.launch(debug=True)
