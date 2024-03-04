import cv2

# Load the image
image = cv2.imread('your_image_path.jpg')

# Specify the new dimensions (width, height)
new_width = 500
new_height = 300

# Resize the image using cv2.resize()
resized_image = cv2.resize(image, (new_width, new_height))

# Display the resized image (optional)
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
