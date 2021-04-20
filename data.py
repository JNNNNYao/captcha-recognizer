import tensorflow as tf
import collections
import random
import numpy as np
import os

from config import constants

def load_data(PATH):
    with open(PATH + 'spec_train_val.txt', 'r') as fh:
        lines = fh.readlines()

    image_path_to_caption = collections.defaultdict(list)
    for val in lines:
        image, text = val.split(' ')
        text = text[:-1]
        text_with_space = ''
        for a in text:
            text_with_space += a + " "
        caption = f"<start> {text_with_space[:-1]} <end>"
        image_path = PATH + '{}.png'.format(image)
        image_path_to_caption[image_path].append(caption)

    train_image_paths = list(image_path_to_caption.keys())
    # random.shuffle(train_image_paths)

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    return train_captions, img_name_vector



def tokenize_caption(train_captions):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=29, # 26 + 3
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Pad each vector to the max_length of the captions
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    max_length = max(len(t) for t in train_seqs)

    return cap_vector, max_length, tokenizer



def load_image(image_path, caption):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (constants['img_size'], constants['img_size']))
    img = img / 255.0 - 0.5
    return img, caption



def load_image_aug(image_path, caption):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    x_min = tf.random.uniform(shape=[], minval=-0.1, maxval=0.1)
    y_min = tf.random.uniform(shape=[], minval=-0.1, maxval=0.1)
    x_max = tf.random.uniform(shape=[], minval=0.9, maxval=1.1)
    y_max = tf.random.uniform(shape=[], minval=0.9, maxval=1.1)
    img = tf.image.crop_and_resize(img[tf.newaxis, :], [[y_min, x_min, y_max, x_max]], [0], 
            [constants['img_size'], constants['img_size']], extrapolation_value=255)
    img = img[0]
    img = img / 255.0 - 0.5
    return img, caption



def split_dataset(img_name_vector, cap_vector):
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    img_keys = list(img_to_cap_vector.keys())
    # random.shuffle(img_keys)

    train_slice_index = 100000
    img_name_train_keys, img_name_val_keys = img_keys[:train_slice_index], img_keys[train_slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    return img_name_train, cap_train, img_name_val, cap_val



def create_dataset(download=True, PATH='./words_captcha/'):
    train_captions, img_name_vector = load_data(PATH)
    cap_vector, max_length, tokenizer = tokenize_caption(train_captions)
    img_name_train, cap_train, img_name_val, cap_val = split_dataset(img_name_vector, cap_vector)

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    # Use map to load the numpy files in parallel
    dataset = dataset.map(load_image_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Shuffle and batch
    dataset = dataset.shuffle(5000).batch(constants['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
    # Use map to load the numpy files in parallel
    ds_val = ds_val.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Shuffle and batch
    ds_val = ds_val.batch(1, drop_remainder=True)   # batch size = 1 for evaluation
    ds_val = ds_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset, ds_val, tokenizer


