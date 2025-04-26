import cv2
import os

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

user_name = input("Nhập tên danh tính: ")
user_folder = os.path.join(dataset_path, user_name)
os.makedirs(user_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không thể mở camera!")
    exit()

count = 0
total_images = 100  # Số ảnh cần thu thập

print("Nhấn SPACE để chụp ảnh, nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc frame từ camera!")
        break

    # Hiển thị khung hình trực tiếp
    cv2.imshow("Chup Anh", frame)

    key = cv2.waitKey(10) & 0xFF  # Tăng thời gian phản hồi tránh giật lag

    if key == 32 and count < total_images:  # SPACE
        img_path = os.path.join(user_folder, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Đã lưu: {img_path}")
        count += 1

    if count >= total_images or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Hoàn tất thu thập ảnh!")
