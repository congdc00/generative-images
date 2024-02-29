import cv2
from skimage import metrics


# Load images
image1 = cv2.imread("/home/congdc/hdd/tool/generative-images/origin_01.png")
image2 = cv2.imread("/home/congdc/hdd/tool/generative-images/edit_3.jpg")
# image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation = cv2.INTER_AREA)

# Convert images to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# Calculate SSIM
ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
print(f"SSIM Score: ", round(ssim_score[0], 2))