import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# Task 7: Depth Approximation from Two Images
# =============================================

def compute_depth_map(left_path, right_path):
    # Step 1: Load images
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    # Step 2: Resize to same dimensions
    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    left = cv2.resize(left, (w, h))
    right = cv2.resize(right, (w, h))

    # Step 3: Convert to grayscale
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Step 4: Compute disparity using StereoBM
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(gray_left, gray_right)

    # Step 5: Normalize disparity to 0-255
    disp_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_normalized = np.uint8(disp_normalized)

    # Step 6: Apply colormap (Bright = Closer, Dark = Farther)
    depth_colored = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

    # Step 7: Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    plt.title('Left Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    plt.title('Right Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
    plt.title('Depth Map\n(Bright=Close | Dark=Far)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('depth_map_result.jpg', dpi=150)
    plt.show()

    cv2.imwrite('depth_map.jpg', depth_colored)
    print("Depth map saved as: depth_map.jpg")

# Run
compute_depth_map('left_image.jpg', 'right_image.jpg')
