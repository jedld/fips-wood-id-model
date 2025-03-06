import torchvision.transforms as transforms
from PIL import Image

class ResizeWithPadding:
    def __init__(self, size, fill=0):
        self.size = size  # Desired output size, e.g., 512
        self.fill = fill  # Padding fill value (0 for black)

    def __call__(self, img):
        # Original image size
        w, h = img.size

        # Calculate the scaling factor to fit the image within the desired size
        scale = self.size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image with the calculated dimensions
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create a new image with the desired size and fill color
        new_img = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))

        # Paste the resized image onto the center of the new image
        paste_x = (self.size - new_w) // 2
        paste_y = (self.size - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))

        return new_img