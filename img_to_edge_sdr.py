import os
import cv2
import numpy as np

def apply_sobel_to_nested_folders(input_root, output_root):
    for subdir in os.listdir(input_root):
        sub_input_dir = os.path.join(input_root, subdir)
        sub_output_dir = os.path.join(output_root, subdir)

        # 如果是資料夾才處理
        if os.path.isdir(sub_input_dir):
            os.makedirs(sub_output_dir, exist_ok=True)

            for filename in os.listdir(sub_input_dir):
                if filename.lower().endswith('.png'):
                    input_path = os.path.join(sub_input_dir, filename)

                    # 讀取灰階圖
                    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"⚠️ 無法讀取圖片：{input_path}")
                        continue

                    # 計算 Sobel 邊緣圖
                    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
                    sobel = sobel.astype(np.uint8)

                    # 儲存到對應位置
                    output_path = os.path.join(sub_output_dir, filename)
                    cv2.imwrite(output_path, sobel)

    print(f"✅ Sobel 邊緣處理完成，儲存於：{output_root}")

# 範例呼叫
apply_sobel_to_nested_folders(
    r"dataset\test\sdr",
    r"dataset\test\sdr_edge"
)