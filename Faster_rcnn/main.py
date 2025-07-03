import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4
num_epochs = 100
learning_rate = 0.005
patience = 25  # early stopping patience

img_dir = 'dataset/images'
ann_file = 'dataset/_annotations.coco.json'
model_save_path = 'fasterrcnn_model.pth'
best_model_path = 'best_model.pth'

# === CREATE RESULT DIRECTORY ===
os.makedirs("results", exist_ok=True)

# === CUSTOM DATASET CLASS ===
class CocoDetectionTransform(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.ids = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        image_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([image_id])
        }

        if self._transforms:
            img = self._transforms(img)

        return img, target

# === TRANSFORMS ===
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True)
])

# === LOAD DATASET AND SPLIT ===
full_dataset = CocoDetectionTransform(img_dir, ann_file, transform)
image_ids = full_dataset.ids
train_ids, temp_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

train_indices = [full_dataset.ids.index(id) for id in train_ids]
val_indices = [full_dataset.ids.index(id) for id in val_ids]
test_indices = [full_dataset.ids.index(id) for id in test_ids]

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# === MODEL SETUP ===
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# === COCO OBJECT FOR VALIDATION SET ===
coco_val = COCO(ann_file)
coco_val.dataset['images'] = [img for img in coco_val.dataset['images'] if img['id'] in set(val_ids)]
coco_val.createIndex()

# === EVALUATION FUNCTION ===
def evaluate(model, data_loader, coco_gt, device):
    model.eval()
    coco_results = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                image_id = int(targets[i]['image_id'].item())
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        'score': float(score)
                    })

    if not coco_results:
        print("No predictions to evaluate.")
        return 0.0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # mAP@[IoU=0.50:0.95]

# === TRAINING LOOP WITH TRACKING AND EARLY STOPPING ===
best_map = 0.0
epochs_without_improvement = 0
train_loss_per_epoch = []
val_map_per_epoch = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    start = time.time()

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
    print("Evaluating on validation set...")
    map_score = evaluate(model, val_loader, coco_val, device)
    print(f"Validation mAP@[IoU=0.50:0.95]: {map_score:.4f}")

    # Save tracking
    train_loss_per_epoch.append(epoch_loss)
    val_map_per_epoch.append(map_score)

    # Early stopping logic
    if map_score > best_map:
        best_map = map_score
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ New best model saved with mAP: {best_map:.4f}")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

    if epochs_without_improvement >= patience:
        print("⏹️ Early stopping triggered.")
        break

# === FINAL SAVE ===
torch.save(model.state_dict(), model_save_path)
print(f"\nTraining complete. Final model saved to {model_save_path}")
print(f"Best model saved to {best_model_path} with mAP: {best_map:.4f}")

# === PLOT LOSS AND MAP ===
plt.figure()
plt.plot(train_loss_per_epoch, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('results/loss_curve.png')
plt.show() 

plt.figure()
plt.plot(val_map_per_epoch, label='Validation mAP@[0.5:0.95]')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('Validation mAP per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('results/map_curve.png')
plt.show()

# === EVALUATE ON TEST SET ===
print("\nEvaluating on test set...")
coco_test = COCO(ann_file)
coco_test.dataset['images'] = [img for img in coco_test.dataset['images'] if img['id'] in set(test_ids)]
coco_test.createIndex()
evaluate(model, test_loader, coco_test, device)
