import os
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms.v2 as T
from pycocotools.coco import COCO

# === CONFIG ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'best_model.pth'
img_dir = 'dataset/images'
ann_file = 'dataset/_annotations.coco.json'
output_dir = 'result/detected_images'
threshold = 0.5

os.makedirs(output_dir, exist_ok=True)

# === LOAD MODEL ===
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === LOAD CLASS NAMES ===
coco = COCO(ann_file)
id2name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

# === TRANSFORMS ===
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True)
])

# === RUN INFERENCE ON ALL IMAGES ===
image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
font = ImageFont.load_default()

for img_file in image_files:
    img_path = os.path.join(img_dir, img_file)
    img = Image.open(img_path).convert('RGB')
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
        if score >= threshold:
            x1, y1, x2, y2 = box.cpu().numpy()
            class_name = id2name.get(label.item(), f"ID {label.item()}")
            label_text = f"{class_name} {score:.2f}"

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

            # Draw label background
            text_size = draw.textbbox((x1, y1), label_text, font=font)
            draw.rectangle([text_size[0], text_size[1], text_size[2], text_size[3]], fill='red')

            # Draw label text
            draw.text((x1, y1), label_text, fill='white', font=font)

    # Save image
    save_path = os.path.join(output_dir, img_file)
    draw_img.save(save_path)

print(f"✅ All detections saved to '{output_dir}'")
