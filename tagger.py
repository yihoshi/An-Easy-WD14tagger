import os
import glob
import csv
import argparse
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 定义常量
IMAGE_SIZE = 448 
CSV_FILE = "selected_tags.csv"  

def load_onnx_model(model_path):
    return ort.InferenceSession(model_path)

def preprocess_image(image):
    #预处理图像
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB -> BGR
    size = max(image.shape[0:2])
    if size > IMAGE_SIZE:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LANCZOS4
    
    if size != IMAGE_SIZE:
        pad_x = size - image.shape[1]
        pad_y = size - image.shape[0]
        pad_l = pad_x // 2
        pad_t = pad_y // 2
        image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode='constant', constant_values=255)
    
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)
    image = image.astype(np.float32)
    return image

class ImageLoadingPrepDataset(Dataset):
    #自定义数据集类，用于加载和预处理图像
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image)
        except Exception as e:
            print(f"Could not load image path: {img_path}, error: {e}")
            return None
        return (image, img_path)

def collate_fn_remove_corrupted(batch):
    #过滤掉加载失败的样本 
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    images, paths = zip(*batch)
    return list(images), list(paths)

def run_batch(path_imgs, model, tags, args):
    if path_imgs is None:
        return

    images, image_paths = path_imgs
    images = np.array(images)

    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: images})[0]
    probs = outputs

    for image_path, prob in zip(image_paths, probs):
        tag_probs = prob[4:]
        valid_indices = [i for i, p in enumerate(tag_probs) if p >= args.thresh and i < len(tags)]
        
        tag_text = ", ".join([
            tags[i].replace("_", " ") if args.replace_underscores else tags[i]
            for i in valid_indices
        ])

        if tag_text:
            base_name = os.path.basename(image_path)
            caption_file_path = os.path.join(os.path.dirname(image_path), os.path.splitext(base_name)[0] + args.caption_extension)
            with open(caption_file_path, "wt", encoding='utf-8') as f:
                f.write(tag_text + '\n')
                if args.debug:
                    print(image_path, tag_text)

        if isinstance(image_path, str):
            # 使用 image_path 的目录作为基础路径
            base_name = os.path.basename(image_path)
            caption_file_path = os.path.join(os.path.dirname(image_path), os.path.splitext(base_name)[0] + args.caption_extension)
            with open(caption_file_path, "wt", encoding='utf-8') as f:
                f.write(tag_text + '\n')
                if args.debug:
                    print(image_path, tag_text)
        else:
            print(f"Unexpected type for image_path: {type(image_path)}")

def validate_arguments(args):
   #验证命令行参数
    if not os.path.isdir(args.train_data_dir):
        raise ValueError(f"Train data directory does not exist: {args.train_data_dir}")
    if not os.path.isdir(args.model_dir):
        raise ValueError(f"Model directory does not exist: {args.model_dir}")

def load_tags(csv_file_path):
    #读取标签文件
    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tags = [row['name'] for row in reader if row['category'] == '0']
    return tags

def main(args):
    image_paths = glob.glob(os.path.join(args.train_data_dir, "*.jpg")) + \
                  glob.glob(os.path.join(args.train_data_dir, "*.png")) + \
                  glob.glob(os.path.join(args.train_data_dir, "*.webp"))
    print(f"Found {len(image_paths)} images.")

    print("Loading model and labels")
    model = load_onnx_model(os.path.join(args.model_dir, "model.onnx"))

    tags = load_tags(os.path.join(args.model_dir, CSV_FILE))

    dataset = ImageLoadingPrepDataset(image_paths)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.max_data_loader_n_workers, collate_fn=collate_fn_remove_corrupted)

    for batch in tqdm(data_loader, desc="Processing images"):
        if batch is None:
            continue
        run_batch(batch, model, tags, args)

    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run image tagging using an ONNX model")
    parser.add_argument("train_data_dir", type=str, help="Directory for training images")
    parser.add_argument("--model_dir", type=str, default="wd14_tagger_model", help="Directory to store the ONNX model")
    parser.add_argument("--thresh", type=float, default=0.35, help="Confidence threshold to add a tag")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=4, help="Number of worker threads for data loading")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="Extension of the caption file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--replace_underscores", action='store_true', help="Replace underscores with spaces in tags")

    args = parser.parse_args()

    validate_arguments(args)

    main(args)