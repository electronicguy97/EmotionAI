import os, cv2, torch
from tqdm import tqdm
from IPython.display import clear_output
from ultralytics import YOLO
import argparse

def main(args):
    # Set device
    device = torch.device(args.device)

    # Load YOLO model
    yolo_model = YOLO(args.weights).to(device)

    # Output directories for cropped images
    output_dir_train = os.path.join(args.data_dir, 'train_{args.output_dir}')
    output_dir_val = os.path.join(args.data_dir, 'val_{args.output_dir}')

    # Define function to get image files from a directory
    def get_image_files(directory):
        image_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        return image_files

    # Define function to crop faces from images
    def crop_faces(image_path, output_dir):
        image = cv2.imread(image_path)
        results = yolo_model(image)

        for i, result in enumerate(results):
            try:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = box.tolist()
                    face = image[int(y1):int(y2), int(x1):int(x2)]
                    filename = os.path.basename(image_path).split('.')[0]
                    class_name = os.path.basename(os.path.dirname(image_path))
                    output_class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(output_class_dir, exist_ok=True)
                    output_face_path = os.path.join(output_class_dir, f"{filename}_face_{i}.jpg")
                    cv2.imwrite(output_face_path, face)
            except IndexError:
                print(f"No face detected in the image: {image_path}")

    # Create output directories
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_val, exist_ok=True)

    # Process train images
    train_dir = os.path.join(args.data_dir, 'train')
    image_files_train = get_image_files(train_dir)
    for image_path in tqdm(image_files_train, desc='Processing train images'):
        crop_faces(image_path, output_dir_train)
        clear_output(wait=True)

    # Process val images
    val_dir = os.path.join(args.data_dir, 'val')
    image_files_val = get_image_files(val_dir)
    for image_path in tqdm(image_files_val, desc='Processing val images'):
        crop_faces(image_path, output_dir_val)
        clear_output(wait=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop faces from images using YOLO model")
    parser.add_argument('-d',"--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use (cuda or cpu)")
    parser.add_argument('-w',"--weights", type=str, default="../weights/yolov8n-face.pt", help="Path to YOLO weights file")
    parser.add_argument("--data-dir", type=str, default="../../../data/image", help="Data directory path")
    parser.add_argument('-o',"--output-dir", type=str, default="crop", help="Output directory path for cropped images")
    args = parser.parse_args()
    main(args)

