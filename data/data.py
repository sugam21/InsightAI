import os
from typing import List


def main() -> None:
    path_to_images: str = r"D:\Python\InsightAI\src\insightai\data\images"
    image_class: List[str] = os.listdir(path_to_images)
    total_images: int = 0

    for class_name in image_class:
        img_class_dir: str = os.path.join(path_to_images, class_name)
        total_images += len(os.listdir(img_class_dir))

    print("Toal Images are", total_images)


if __name__ == "__main__":
    main()
