import math
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')


def grayscale():
    img = Image.open("static/img/img_default.jpg")
    img = img.convert("RGBA")

    img_arr = np.asarray(img)
    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    sum_r = np.sum(r)
    sum_g = np.sum(g)
    sum_b = np.sum(b)
    sum_all = sum_r + sum_g + sum_b

    arr_gray = (sum_r / sum_all * r) + \
        (sum_g / sum_all * g) + (sum_b / sum_all * b)

    img_new = Image.fromarray(arr_gray)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/img_grayscaled.jpg")


def zoomin():
    img = Image.open("static/img/img_default.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img)
    new_size = ((img_arr.shape[0] * 2),
                (img_arr.shape[1] * 2), img_arr.shape[2])
    new_arr = np.full(new_size, 255)
    new_arr.setflags(write=1)

    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    new_r = []
    new_g = []
    new_b = []

    for row in range(len(r)):
        temp_r = []
        temp_g = []
        temp_b = []
        for i in r[row]:
            temp_r.extend([i, i])
        for j in g[row]:
            temp_g.extend([j, j])
        for k in b[row]:
            temp_b.extend([k, k])
        for _ in (0, 1):
            new_r.append(temp_r)
            new_g.append(temp_g)
            new_b.append(temp_b)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_arr[i, j, 0] = new_r[i][j]
            new_arr[i, j, 1] = new_g[i][j]
            new_arr[i, j, 2] = new_b[i][j]

    new_arr = np.uint8(new_arr)
    img_new = Image.fromarray(new_arr)
    img_new.save("static/img/img_zoomed_in.jpg")


def zoomout():
    img = Image.open("static/img/img_default.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img)

    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    new_arr_size = ((img_arr.shape[0] // 2),
                    (img_arr.shape[1] // 2), img_arr.shape[2])

    new_arr = np.full(new_arr_size, 255)
    new_arr.setflags(write=1)

    new_r = []
    new_g = []
    new_b = []

    for i in range(0, len(r), 2):
        temp_r = []
        temp_g = []
        temp_b = []
        for j in range(0, len(r[i]), 2):
            temp_r.append(
                math.ceil((r[i][j]+r[i][j+1]+r[i+1][j]+r[i+1][j+1])/4))
            temp_g.append(
                math.ceil((g[i][j]+g[i][j+1]+g[i+1][j]+g[i+1][j+1])/4))
            temp_b.append(
                math.ceil((b[i][j]+b[i][j+1]+b[i+1][j]+b[i+1][j+1])/4))
        new_r.append(temp_r)
        new_g.append(temp_g)
        new_b.append(temp_b)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_arr[i, j, 0] = new_r[i][j]
            new_arr[i, j, 1] = new_g[i][j]
            new_arr[i, j, 2] = new_b[i][j]

    new_arr = np.uint8(new_arr)
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/img_zoomed_out.jpg")


def move_left():
    img = Image.open("static/img/img_default.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
    g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
    b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_moved_left.jpg")


def move_right():
    img = Image.open("static/img/img_default.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
    g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
    b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_moved_right.jpg")


def move_up():
    img = Image.open("static/img/img_default.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_moved_up.jpg")


def move_down():
    img = Image.open("static/img/img_default.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_moved_down.jpg")


def brightness_addition():
    img = Image.open("static/img/img_default.jpg")
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_brightened_addition.jpg")


def brightness_substraction():
    img = Image.open("static/img/img_default.jpg")
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_darkened_substraction.jpg")


def brightness_multiplication():
    img = Image.open("static/img/img_default.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_brightened_multiplication.jpg")


def brightness_division():
    img = Image.open("static/img/img_default.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_darkened_division.jpg")
