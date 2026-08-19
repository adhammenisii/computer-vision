import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# Task 8: High Dynamic Range (HDR) Imaging
# =============================================

def create_hdr(image_paths, exposure_times):
    """
    image_paths: list of image file paths (min 3)
    exposure_times: list of exposure times in seconds e.g. [1/30, 1/8, 1/2, 2]
    """

    # Step 1: Read all images
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        images.append(img)
    print(f"Loaded {len(images)} images")

    exposure_times = np.array(exposure_times, dtype=np.float32)

    # Step 2: Estimate Camera Response Function (CRF) using Debevec method
    print("Estimating Camera Response Function (CRF)...")
    calibrate = cv2.createCalibrateDebevec()
    crf = calibrate.process(images, times=exposure_times)

    # Step 3: Merge exposures into HDR radiance map
    print("Merging exposures into HDR...")
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(images, times=exposure_times, response=crf)

    # Step 4: Tone mapping using Reinhard operator
    print("Applying Reinhard tone mapping...")
    tonemap = cv2.createTonemapReinhard(
        gamma=2.2,
        intensity=0,
        light_adapt=0,
        color_adapt=0
    )
    ldr = tonemap.process(hdr)

    # Step 5: Convert to 8-bit
    ldr_8bit = np.clip(ldr * 255, 0, 255).astype(np.uint8)

    # Step 6: Display results
    plt.figure(figsize=(18, 5))

    for i, (img, exp) in enumerate(zip(images, exposure_times)):
        plt.subplot(1, len(images) + 1, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Exposure: {exp:.4f}s')
        plt.axis('off')

    plt.subplot(1, len(images) + 1, len(images) + 1)
    plt.imshow(cv2.cvtColor(ldr_8bit, cv2.COLOR_BGR2RGB))
    plt.title('HDR Result\n(Reinhard Tone Mapping)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('hdr_result.jpg', dpi=150)
    plt.show()

    cv2.imwrite('hdr_output.jpg', ldr_8bit)
    print("HDR image saved as: hdr_output.jpg")

# ---- How to use ----
# Put your images in the same folder and run:

image_paths = ['dark.jpg', 'normal.jpg', 'bright.jpg']  # your images here
exposure_times = [1/30, 1/8, 1/2]  # exposure in seconds

create_hdr(image_paths, exposure_times)
