import cv2
def load_and_preprocess_image(image_path):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    resized_image = cv2.resize(grayscale_image, (28, 28), interpolation=cv2.INTER_AREA)

    return resized_image / 255.0