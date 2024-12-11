import cv2
import os

# 경로 설정
input_path = os.path.expanduser("~/Desktop/open/bus.jpeg")
output_path = os.path.expanduser("~/Desktop/open/bus.jpeg")

try:
    # 이미지 읽기
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not read image from {input_path}")

    # 경계 감지 적용
    edges = cv2.Canny(image, 100, 200)

    # 결과 저장
    cv2.imwrite(output_path, edges)
    print(f"Edge-detected image saved to {output_path}")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
