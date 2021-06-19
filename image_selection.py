import os
import random
import re
import shutil

import imageio
import numpy as np
import rawpy
import tensorflow as tf

src_dir = "C:/Users/s4625266/Dropbox/UCE Corals/"
dst_dir = "C:/Users/s4625266/PycharmProjects/coral/processed_image/"

ext = ('.jpg', '.png', '.jpeg')
docx = '.docx'


def select_identical_image():
    for root, _, images in os.walk(src_dir):
        if images:
            _raw = False
            current_directory_path = os.path.abspath(root)
            m = re.search(r'^.+\\([^\\]+)$', current_directory_path)
            black_pixels = []
            for image_file in images:
                if not image_file.lower().endswith(docx):
                    print(f'image: {image_file}')
                    try:
                        image = tf.keras.preprocessing.image.load_img(current_directory_path + '/' + image_file)
                    except:
                        print('raw file found')
                        raw = rawpy.imread(current_directory_path + '/' + image_file)
                        image = raw.postprocess()

                    input_arr = tf.keras.preprocessing.image.img_to_array(image)
                    number_of_black_pix = np.sum(input_arr.ravel() == 0)
                    black_pixels.append(number_of_black_pix)

            try:
                threshold = sum(black_pixels) / float(len(black_pixels))
                selected_image = images[black_pixels.index(min(i for i in black_pixels if i > threshold))]
            except ValueError:
                selected_image = random.choice(images)

            if not selected_image.lower().endswith(ext):
                _raw = True

            print(f'selected_image: {selected_image}')
            directory = dst_dir + m.groups(1)[0]
            source_directory = src_dir + m.groups(1)[0] + '/' + selected_image

            if not os.path.exists(directory):
                print('making directory')
                os.makedirs(directory)

            print('copying file')
            if _raw:
                print('change raw format to jpg')
                raw = rawpy.imread(current_directory_path + '/' + selected_image)
                image = raw.postprocess()
                imageio.imsave(f'{directory + "/" + selected_image.rsplit(".", 1)[0]}.jpg', image)
            else:
                shutil.copy(source_directory, directory + "/" + selected_image)


select_identical_image()
