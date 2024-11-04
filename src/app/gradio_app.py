from pathlib import Path

import gradio as gr
import torch
from loguru import logger
from PIL import Image
from torchvision import transforms

from src.model import SqueezeNetModel
from src.rag import RagPipeline

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


RAG_CONFIG_PATH: str = Path("rag_config.json").resolve()


def get_prediction(input_image_path):
    transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # load model
    path_to_checkpoint = str(
        Path(r"saved/checkpoints/epoch=14-step=7725_squeeze_net.ckpt").resolve(),
    )

    model = SqueezeNetModel.load_from_checkpoint(path_to_checkpoint, num_classes=17)
    model.eval()

    image_numpy = Image.open(input_image_path).convert("RGB")
    transformed_image = transform(image_numpy).unsqueeze(0)  # added a bath dim.
    logits = model(transformed_image)
    preds = torch.argmax(logits, -1).item()

    return class_name_mapping_dict[str(preds)]


last_uploaded_image_path: str | None = None


def generate_response(message, history):
    global last_uploaded_image_path
    try:
        if "files" in message and message["files"]:
            input_image_path = message["files"][0]
            last_uploaded_image_path = input_image_path
        elif last_uploaded_image_path:
            input_image_path = last_uploaded_image_path
        else:
            return "Please Upload an image!!"
    except Exception as e:
        return f"An error occured, {e}"

    user_query: str = message.get("text", "")
    if not user_query:
        return "User qurey is empty!!"

    predict = get_prediction(input_image_path)
    logger.info(f"Predicted Image class {predict}")
    logger.debug(predict)
    logger.debug(user_query)
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
