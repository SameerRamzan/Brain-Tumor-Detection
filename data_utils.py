import os
import cv2
import numpy as np

def letterbox(im, new_shape=(224, 224), color=(0, 0, 0)): #color=(114,114,114), you can use this color for gray scales 

    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int): #Its a check. It allows a user to pass a single integer (e.g., letterbox(im, 224)) for a square output
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

def load_data(dir_path, img_size):

    data = []
    labels = []
    
    # Define paths for YES and NO subdirectories
    yes_path = os.path.join(dir_path, 'YES')
    no_path = os.path.join(dir_path, 'NO')
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG')

    # Determine the target size for resizing
    if isinstance(img_size, int):
        target_size = (img_size, img_size)
    elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        target_size = tuple(map(int, img_size))
    else:
        raise ValueError("img_size must be an integer or a tuple/list of two integers.")

    # Process images in the YES directory
    if os.path.exists(yes_path):
        for filename in os.listdir(yes_path):
            if filename.endswith(image_extensions):
                img_path = os.path.join(yes_path, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}. Skipping.")
                        continue
                    img = letterbox(img, new_shape=target_size)
                    data.append(img)
                    labels.append(1)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

    # Process images in the NO directory
    if os.path.exists(no_path):
        for filename in os.listdir(no_path):
            if filename.endswith(image_extensions):
                img_path = os.path.join(no_path, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}. Skipping.")
                        continue
                    img = letterbox(img, new_shape=target_size)
                    data.append(img)
                    labels.append(0)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Shuffle the data and labels together
    if len(data) > 0:
        indices = np.arange(data.shape[0])
        np.random.seed(42) # for reproducibility
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
    
    print(f"Loaded {len(data)} images from {dir_path}.")
    
    return data, labels
