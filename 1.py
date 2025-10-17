import cv2


def call_ks2a418(cam_id=1):  # 替换成你的摄像头编号
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"ERROR：无法打开编号 {cam_id} 的摄像头")
        return

    # -------------------------- 关键：开启自动曝光 + 优化参数 --------------------------
    # 1. 开启自动曝光（不同摄像头参数ID可能不同，先试这两个常用ID）
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75=开启自动曝光（多数USB摄像头适用）
    # 若上面无效，替换成这个：cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    # 2. 开启自动白平衡（避免画面偏色，辅助改善曝光视觉效果）
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # 1=开启自动白平衡
r
    # 3. （可选）手动限制最大曝光值（若自动曝光仍过曝，可添加这行）
    # cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # 数值越小，曝光越低（范围：-10~0，根据实际调整）
    # -----------------------------------------------------------------------------------

    # 设置分辨率（KS2A418支持1080P，可按需调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 255)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 255)

    print("按 ESC 退出，按 S 保存画面")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR：无法读取画面")
            break

        frame = cv2.flip(frame, 1)  # 水平翻转
        cv2.imshow("KS2A418 Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            save_path = f"ks2a418_capture_{cv2.getTickCount()}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"已保存：{save_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    call_ks2a418(cam_id=1)  # 替换成你的摄像头编号