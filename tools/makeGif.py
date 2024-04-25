import os
from PIL import Image

input_folder = 'E:/project/infantVision/data/random cropping/64x64/16963'
output_gif = 'E:/project/infantVision/results/data analysis/randomCropping64.gif'

image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

image_files = image_files[:30]


images = [Image.open(os.path.join(input_folder, image)) for image in image_files]


images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=500,  # 500ms per frame
    loop=0  # infinite loop
)

print(f'Gif saved: {output_gif}')
