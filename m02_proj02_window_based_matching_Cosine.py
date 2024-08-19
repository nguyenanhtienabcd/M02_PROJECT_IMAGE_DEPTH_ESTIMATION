import numpy as np
import cv2

# Using L1 absolute difference - Manthan Distance for windown-based matching


def cosine_similarity(x, y):
    numerator = np.dot(x, y)
    denomirator = np.linalg.norm(x)*np.linalg.norm(y)
    return numerator/denomirator


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

    # xét tới từng điểm ảnh
    for y in range(kernel_half, (height - kernel_half)):
        for x in range(kernel_half, (width - kernel_half)):
            disparity = 0
            cost_max = -1

            # xét tới giải đo của disparity
            for j in range(disparity_range):
                d = x - j
                cost = -1
                if d - kernel_half >= 0:
                    wf = left[y - kernel_half: y + kernel_half +
                              1, x - kernel_half: x + kernel_half + 1]
                    wr = right[y - kernel_half: y + kernel_half +
                               1, d - kernel_half: d + kernel_half + 1]
                    wff = wf.flatten()
                    wrf = wr.flatten()
                    cost = cosine_similarity(wff, wrf)

                if cost > cost_max:
                    cost_max = cost
                    disparity = j
            disparity_map[y, x] = disparity * scale
    return disparity_map


def save_result(disparity_map):
    print('Saving result ... ')
    cv2.imwrite('window_based_Cosine.png', disparity_map)
    cv2.imwrite('window_based_Cosine_color.png',
                cv2.applyColorMap(disparity_map,
                                  cv2.COLORMAP_JET))
    print('Done !!!')


if __name__ == '__main__':
    left_image_path = 'Aloe/Aloe_left_1.png'
    right_image_path = 'Aloe/Aloe_right_1.png'
    disparity_range = 64
    kernel_size = 5
    disparity_map = window_based_mactching(
        left_image_path,
        right_image_path,
        disparity_range,
        kernel_size)
    save_result(disparity_map)
