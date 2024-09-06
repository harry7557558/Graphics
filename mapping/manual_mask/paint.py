import cv2
import numpy as np
import os


# Initialize variables
mask = None
is_mouse_down = False
display_scale = 2

all_points = {}
all_labels = {}
current_image_index = 1

def load_image(index):
    global image, current_image_index
    filename = f"masks/frame_{index:05d}.png"
    if os.path.exists(filename):
        print(f"Load image {filename}")
        image = cv2.imread(filename)
        current_image_index = index
        return True
    return False

def on_mouse(event, x, y, flags, param):
    x, y = x * display_scale, y * display_scale
    global mask, is_mouse_down
    label = 0 if flags & cv2.EVENT_FLAG_ALTKEY else 255
    
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
    mask = image.copy()
    for (x, y), label in zip(input_point, input_label):
        s = 32
        mask[y-s:y+s+1, x-s:x+s+1] = label


def save_mask():
    if mask is not None:
        filename = f"masks_painted/frame_{current_image_index:05d}.png"
        cv2.imwrite(filename, mask.astype(np.uint8))
        print(f"Mask saved as {filename}")

# Load initial image
load_image(1)

# Create window
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", on_mouse)

while True:
    if mask is None:
        display_image = image.copy()
    else:
        display_image = mask.copy()

    if current_image_index not in all_points:
        all_points[current_image_index] = []
    if current_image_index not in all_labels:
        all_labels[current_image_index] = []
    
    # Draw selection points
    for point, label in zip(all_points[current_image_index], all_labels[current_image_index]):
        color = (0, 255, 0) if label > 0 else (0, 0, 255)
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
