import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import os

# Load SAM2 model
sam_checkpoint = "/media/harry7557558/New Volume/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
# sam_checkpoint = "/media/harry7557558/New Volume/sam/xl0.pt"
# model_type = "xl0"
# sam_checkpoint = "/media/harry7557558/New Volume/sam/sam_vit_b_01ec64.pth"
# model_type = "vit_b"


#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:0"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Initialize variables
mask = None
is_mouse_down = False
display_scale = 2

all_points = {}
all_labels = {}
current_image_index = 1

def load_image(index):
    global image, image_rgb, current_image_index
    filename = f"images/frame_{index:05d}.jpg"
    if os.path.exists(filename):
        print(f"Load image {filename}")
        image = cv2.imread(filename)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        current_image_index = index
        return True
    return False

def on_mouse(event, x, y, flags, param):
    x, y = x * display_scale, y * display_scale
    global mask, is_mouse_down
    label = 0 if flags & cv2.EVENT_FLAG_ALTKEY else 1
    
    if event == cv2.EVENT_MOUSEMOVE:
        update_mask(x, y, label)
    elif event == cv2.EVENT_LBUTTONDOWN:
        all_points[current_image_index].append([x, y])
        all_labels[current_image_index].append(label)
        is_mouse_down = True
        update_mask(x, y, label)
    elif event == cv2.EVENT_LBUTTONUP:
        is_mouse_down = False

def update_mask(x, y, label):
    global mask
    if label is None:
        if len(all_points[current_image_index]) == 0:
            mask = None
            return
        input_point = np.array(all_points[current_image_index])
        input_label = np.array(all_labels[current_image_index])
    else:
        input_point = np.array(all_points[current_image_index] + [[x, y]])
        input_label = np.array(all_labels[current_image_index] + [label])
    mask, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

def save_mask():
    if mask is not None:
        filename = f"masks/frame_{current_image_index:05d}.png"
        cv2.imwrite(filename, (mask[0] * 255).astype(np.uint8))
        print(f"Mask saved as {filename}")

# Load initial image
load_image(26)

# Create window
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", on_mouse)

while True:
    display_image = image.copy()

    if current_image_index not in all_points:
        all_points[current_image_index] = []
    if current_image_index not in all_labels:
        all_labels[current_image_index] = []
    
    if mask is not None:
        # Overlay mask on image
        mask_overlay = (mask[0, :, :, None] * [255, 0, 0]).astype(np.uint8)
        cv2.addWeighted(display_image, 1, mask_overlay, 0.5, 0, display_image)
    
    # Draw selection points
    for point, label in zip(all_points[current_image_index], all_labels[current_image_index]):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(display_image, (int(point[0]), int(point[1])), 5, color, -1)
    
    h, w = np.array(display_image.shape)[:2] / display_scale
    display_image = cv2.resize(display_image, (int(w),int(h)))
    cv2.imshow("Image", display_image)
    
    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        print("Key down:", key)
    if key == 27:  # ESC
        break
    if key == 115 and mask is not None:  # S
        save_mask()
    elif key == 8:  # Backspace
        if all_points[current_image_index]:
            all_points[current_image_index].pop()
            all_labels[current_image_index].pop()
            update_mask(-1, -1, None)
    elif key == 91:  # [
        if load_image(current_image_index - 1):
            if current_image_index not in all_points:
                all_points[current_image_index] = []
                all_labels[current_image_index] = []
            update_mask(-1, -1, None)
    elif key == 93:  # ]
        if load_image(current_image_index + 1):
            if current_image_index not in all_points:
                all_points[current_image_index] = []
                all_labels[current_image_index] = []
            update_mask(-1, -1, None)

cv2.destroyAllWindows()