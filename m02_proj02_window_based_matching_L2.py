import numpy as np
import cv2

# Using L2 mean square difference - Manthan Distance for windown-based matching


def distance(x, y):
    return (x-y)**2


def window_based_mactching(
        left_image,
        right_image,
        disparity_range,
        kernel_size=5,
):
    # đọc ảnh bên trái và đọc ảnh bên phải
    left_img = cv2.imread(left_image, 0)
    right_img = cv2.imread(right_image, 0)

    # ép kiểu cho ảnh thành float32
    left = left_img.astype(np.float32)
    right = right_img.astype(np.float32)

    # lấy chiều cao và chiều rộng
    height, width = left.shape[:2]

    # khởi tạo disparity map
    disparity_map = np.zeros((height, width), np.uint8)

    kernel_half = kernel_size//2
    scale = 3
    max_value = 255

    # xét tới từng điểm ảnh
    for y in range(kernel_half, (height - kernel_half)):
        for x in range(kernel_half, (width - kernel_half)):
            disparity = 0
            total_min = max_value * (kernel_size**2)
            for j in range(disparity_range):
                total = cal_total(left, right, kernel_half, max_value, y, x, j)
                if total < total_min:
                    total_min = total
                    disparity = j
            disparity_map[y, x] = disparity * scale
    return disparity_map


def cal_total(left, right, kernel_half, max_value, y, x, j):
    total = 0
    for v in range((-kernel_half), (kernel_half + 1)):
        for u in range((-kernel_half), (kernel_half + 1)):
            value = max_value
            if x + u - j >= 0:
                value = distance(
                        int(left[y + v, x + u]), int(right[y + v, (x + u) - j]))
            total += value
    return total


def save_result(disparity_map):
    print('Saving result ... ')
    cv2.imwrite('window_based_l2.png', disparity_map)
    cv2.imwrite('window_based_l2_color.png',
                cv2.applyColorMap(disparity_map,
                                  cv2.COLORMAP_JET))
    print('Done !!!')


if __name__ == '__main__':
    left_image_path = 'Aloe/Aloe_left_1.png'
    right_image_path = 'Aloe/Aloe_right_1.png'
    disparity_range = 64
    kernel_size = 3
    disparity_map = window_based_mactching(
        left_image_path,
        right_image_path,
        disparity_range,
        kernel_size)
    save_result(disparity_map)
