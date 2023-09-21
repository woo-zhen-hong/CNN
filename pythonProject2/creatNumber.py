import random
import os
from PIL import Image, ImageDraw, ImageFont

random.seed(3)
path_img = "1to10/"

def mkdir_for_images():
    if not os.path.isdir(path_img):
        os.mkdir(path_img)
    for i in range(1, 10):
        if os.path.isdir(path_img  + str(i)):
            pass
        else:
            print(path_img + "number_" + str(i))
            os.mkdir(path_img + str(i))


mkdir_for_images()

def clear_images():
    for i in range(1, 10):
        dir_nums = os.listdir(path_img + str(i))
        for tmp_img in dir_nums:
            if tmp_img in dir_nums:
                os.remove(path_img  + str(i) + "/" + tmp_img)


clear_images()

# 扭曲 
def generate_single():
    im_50_blank = Image.new('RGB', (50, 50), (255, 255, 255))
    draw = ImageDraw.Draw(im_50_blank)
    num = str(random.randint(1, 9))
    font = ImageFont.truetype('simsun.ttc', 20)

    draw.text(xy=(18, 11), font=font, text=num, fill=(0, 0, 0))

    random_angle = random.randint(-10, 10)
    im_50_rotated = im_50_blank.rotate(random_angle)

    params = [1 - float(random.randint(1, 2)) / 100,
              0,
              0,
              0,
              1 - float(random.randint(1, 10)) / 100,
              float(random.randint(1, 2)) / 500,
              0.001,
              float(random.randint(1, 2)) / 500]

    im_50_transformed = im_50_rotated.transform((50, 50), Image.PERSPECTIVE, params)

    im_30 = im_50_transformed.crop([15, 15, 35, 35])
    return im_30, num

def generate_number_1_to_9(n):
    cnt_num = []

    for i in range(10):
        cnt_num.append(0)

    for m in range(1, n + 1):
        img, generate_num = generate_single()
        img_gray = img.convert('1')

        for j in range(1, 10):
            if generate_num == str(j):
                cnt_num[j] = cnt_num[j] + 1
                img_gray.save(path_img  + str(j) + "/" + str(j) + "_" + str(cnt_num[j]) + ".png")
    print('ok')


def main():
    generate_number_1_to_9(3000)
    

if __name__ == '__main__':
    main()
   