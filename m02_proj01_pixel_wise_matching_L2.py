import numpy as np
import cv2

# Using L2 mean square difference - Euclidean Distance for pixel-wise matching


def distance(x, y):
    return (x-y)**2


def pixel_wise_mactching(left_image, right_image, disparity_range, save_results=True):
    # đọc ảnh bên trái và đọc ảnh bên phải
    left_img = cv2.imread(left_image, 0)
    right_img = cv2.imread(right_image, 0)

    # ép kiểu cho ảnh thành float32
    left_img = left_img.astype(np.float32)
    right_img = right_img.astype(np.float32)

    # lấy chiều cao và chiều rộng
    height, width = left_img.shape[:2]

    # khởi tạo disparity map
    disparity_map = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255

    # xét tới từng điểm ảnh
    for y in range(height):
        for x in range(width):
            disparity = 0
            cost = max_value
            for j in range(disparity_range):
                cost_current = cost
                if x-j >= 0:
                    cost_current = distance(left_img[y, x], right_img[y, x-j])

                if cost_current < cost:
                    cost = cost_current
                    disparity = j
            disparity_map[y, x] = disparity * scale

    if save_results:
        print('Saving result ... ')
        cv2.imwrite('pixel_wise_l2.png', disparity_map)
        cv2.imwrite('pixel_wise_l2_color.png',
                    cv2.applyColorMap(disparity_map,
                                      cv2.COLORMAP_JET))
        print('Done !!!')
    return disparity_map


if __name__ == '__main__':
    left_image_path = 'tsukuba/left.png'
    right_image_path = 'tsukuba/right.png'
    disparity_range = 16
    disparity_map = pixel_wise_mactching(
        left_image_path, right_image_path, disparity_range)
