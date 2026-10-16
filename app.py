import os
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from ultralytics import YOLO

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN & KHỞI TẠO MÔ HÌNH YOLOv8
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "my_chu_best.pt"
EXAMPLES_DIR = BASE_DIR / "examples"

# Tự động chuẩn bị 4 ảnh mẫu vào thư mục examples/ (nếu có thư mục clean & foreign_object)
def prepare_sample_images():
    EXAMPLES_DIR.mkdir(exist_ok=True)
    sample_sources = [
        (BASE_DIR / "clean" / "MC_C_0450.jpg", "sample_clean_1.jpg"),
        (BASE_DIR / "clean" / "MC_C_0490.jpg", "sample_clean_2.jpg"),
        (BASE_DIR / "foreign_object" / "MC_FO_0450.jpg", "sample_foreign_object_1.jpg"),
        (BASE_DIR / "foreign_object" / "MC_FO_0490.jpg", "sample_foreign_object_2.jpg"),
    ]
    for src, dst_name in sample_sources:
        dst = EXAMPLES_DIR / dst_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

prepare_sample_images()

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy trọng số mô hình tại: {MODEL_PATH}. "
        "Vui lòng đặt file 'my_chu_best.pt' cùng thư mục với 'app.py'."
    )

print(f"[INFO] Đang nạp mô hình YOLOv8 từ: {MODEL_PATH} ...")
model = YOLO(str(MODEL_PATH))

CLASS_LABELS_VI = {
    0: "Bó Mỳ Chũ (noodle_bundle)",
    1: "Dị vật bề mặt (foreign_object)"
}


def draw_detections(image_rgb: np.ndarray, boxes_data):
    """Vẽ bounding box với màu sắc trực quan: Xanh lá = Bó mì, Đỏ = Dị vật."""
    canvas = image_rgb.copy()
    h, w = canvas.shape[:2]
    line_thickness = max(3, int(round(min(h, w) * 0.003)))
    font_scale = max(0.65, min(h, w) / 1400.0)

    # Vẽ bó mì (class 0) trước, vẽ dị vật (class 1) đè lên trên để luôn nổi bật
    sorted_boxes = sorted(boxes_data, key=lambda x: x["cls"])

    for item in sorted_boxes:
        x1, y1, x2, y2 = map(int, item["xyxy"])
        cls_id = item["cls"]
        conf = item["conf"]

        if cls_id == 0:
            color = (0, 210, 90)       # Xanh lá (RGB)
            text_color = (0, 0, 0)
            tag = f"noodle_bundle {conf:.2f}"
        else:
            color = (255, 23, 68)      # Đỏ tươi cảnh báo (RGB)
            text_color = (255, 255, 255)
            tag = f"FOREIGN_OBJECT {conf:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=line_thickness)

        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        ty1 = max(0, y1 - th - baseline - 8)
        ty2 = ty1 + th + baseline + 8
        tx2 = min(w, x1 + tw + 12)

        cv2.rectangle(canvas, (x1, ty1), (tx2, ty2), color, thickness=-1)
        cv2.putText(
            canvas,
            tag,
            (x1 + 6, ty2 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

    return canvas


def inspect_noodle_image(input_img: Image.Image, conf_thresh: float, iou_thresh: float):
    if input_img is None:
        empty_html = (
            "<div style='padding:16px;border-radius:10px;background:#f8f9fa;border:1px solid #dee2e6;'>"
            "<b>Chưa có ảnh đầu vào.</b> Vui lòng tải ảnh lên hoặc chọn một ảnh mẫu bên dưới."
            "</div>"
        )
        return None, empty_html, []

    img_rgb = np.array(input_img.convert("RGB"))
    H, W = img_rgb.shape[:2]
    img_area = float(W * H)

    t0 = time.perf_counter()
    results = model.predict(
        source=img_rgb[:, :, ::-1],  # RGB -> BGR cho OpenCV/Ultralytics
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=640,
        verbose=False
    )[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    parsed_boxes = []
    table_rows = []
    noodle_count = 0
    foreign_count = 0

    if results.boxes is not None and len(results.boxes) > 0:
        xyxy_arr = results.boxes.xyxy.cpu().numpy()
        conf_arr = results.boxes.conf.cpu().numpy()
        cls_arr = results.boxes.cls.cpu().numpy().astype(int)

        for idx, (xyxy, conf, cls_id) in enumerate(zip(xyxy_arr, conf_arr, cls_arr), start=1):
            x1, y1, x2, y2 = xyxy.tolist()
            area_pct = ((x2 - x1) * (y2 - y1)) / img_area * 100.0

            if cls_id == 0:
                noodle_count += 1
            elif cls_id == 1:
                foreign_count += 1

            parsed_boxes.append({
                "xyxy": [x1, y1, x2, y2],
                "conf": float(conf),
                "cls": int(cls_id)
            })
            table_rows.append([
                idx,
                CLASS_LABELS_VI.get(cls_id, f"Class {cls_id}"),
                f"{conf * 100:.1f}%",
                f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]",
                f"{area_pct:.2f}%"
            ])

    annotated_rgb = draw_detections(img_rgb, parsed_boxes)

    # Xây dựng thẻ kết luận kiểm định
    if foreign_count > 0:
        status_html = f"""
        <div style="padding:18px;border-radius:12px;background:#fff5f5;border-left:6px solid #e53935;color:#b71c1c;">
            <h3 style="margin:0 0 8px 0;color:#c62828;">CẢNH BÁO: PHÁT HIỆN DỊ VẬT TRÊN BÓ MỲ! (KHÔNG ĐẠT)</h3>
            <p style="margin:4px 0;font-size:15px;">
                Hệ thống phát hiện <b>{foreign_count} dị vật (`foreign_object`)</b> và <b>{noodle_count} bó mì (`noodle_bundle`)</b>.
                Sản phẩm cần được loại bỏ hoặc kiểm tra thủ công trước khi đóng gói.
            </p>
            <p style="margin:4px 0 0 0;font-size:13px;color:#555;">
                Kích thước ảnh: {W}x{H} px | Thời gian suy luận: <b>{elapsed_ms:.1f} ms</b>
            </p>
        </div>
        """
    elif noodle_count > 0 and foreign_count == 0:
        status_html = f"""
        <div style="padding:18px;border-radius:12px;background:#e8f5e9;border-left:6px solid #2e7d32;color:#1b5e20;">
            <h3 style="margin:0 0 8px 0;color:#2e7d32;">ĐẠT CHUẨN: BÓ MỲ CHŨ SẠCH (KHÔNG PHÁT HIỆN DỊ VẬT)</h3>
            <p style="margin:4px 0;font-size:15px;">
                Đã định vị thành công <b>{noodle_count} bó mì (`noodle_bundle`)</b>, không phát hiện dị vật bề mặt.
                Sản phẩm đủ điều kiện đóng gói.
            </p>
            <p style="margin:4px 0 0 0;font-size:13px;color:#555;">
                Kích thước ảnh: {W}x{H} px | Thời gian suy luận: <b>{elapsed_ms:.1f} ms</b>
            </p>
        </div>
        """
    else:
        status_html = f"""
        <div style="padding:18px;border-radius:12px;background:#fff8e1;border-left:6px solid #f9a825;color:#e65100;">
            <h3 style="margin:0 0 8px 0;color:#ef6c00;">KHÔNG TÌM THẤY BÓ MỲ CHŨ TRONG ẢNH</h3>
            <p style="margin:4px 0;font-size:15px;">
                Mô hình không phát hiện được bó mì hay dị vật nào với ngưỡng Confidence hiện tại ({conf_thresh:.2f}).
                Vui lòng chụp rõ sản phẩm Mỳ Chũ hoặc giảm ngưỡng Confidence.
            </p>
            <p style="margin:4px 0 0 0;font-size:13px;color:#555;">
                Kích thước ảnh: {W}x{H} px | Thời gian suy luận: <b>{elapsed_ms:.1f} ms</b>
            </p>
        </div>
        """

    return Image.fromarray(annotated_rgb), status_html, table_rows


# ==============================================================================
# XÂY DỰNG GIAO DIỆN WEB GRADIO
# ==============================================================================
example_files = sorted([str(p) for p in EXAMPLES_DIR.glob("*.jpg")])

with gr.Blocks(title="AI Kiểm Định Dị Vật Mỳ Chũ - YOLOv8") as demo:
    gr.Markdown(
        """
        # Hệ Thống AI Kiểm Tra Dị Vật Trên Bề Mặt Bó Mỳ Chũ (YOLOv8s)
        **Mô tả hệ thống:** Ứng dụng Thị giác máy tính (Computer Vision) tự động định vị sản phẩm và kiểm soát chất lượng bề mặt Mỳ Chũ.
        - **Lớp 0 (`noodle_bundle` - Khung Xanh lá):** Định vị vùng bó Mỳ Chũ.
        - **Lớp 1 (`foreign_object` - Khung Đỏ):** Phát hiện dị vật lạ trên bề mặt (bìa carton, mảnh nhựa, lá khô, xốp, dăm gỗ, sỏi nhỏ...).
        - **Hiệu năng trên tập Test độc lập (150 ảnh):** `mAP@50 = 96.34%` | `Recall (Bó mì) = 100%` | `Recall (Dị vật) = 97.33%`.
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            input_image = gr.Image(
                type="pil",
                label="Ảnh đầu vào (Tải ảnh lên hoặc chọn mẫu bên dưới)",
                height=460
            )
            with gr.Row():
                conf_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.90,
                    value=0.25,
                    step=0.05,
                    label="Ngưỡng độ tin cậy (Confidence Threshold)"
                )
                iou_slider = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.45,
                    step=0.05,
                    label="Ngưỡng khử trùng lặp NMS (IoU Threshold)"
                )
            run_btn = gr.Button("KIỂM TRA CHẤT LƯỢNG SẢN PHẨM", variant="primary")

        with gr.Column(scale=6):
            output_image = gr.Image(
                type="pil",
                label="Kết quả nhận diện Bounding Box",
                height=460
            )
            status_box = gr.HTML(label="Kết luận kiểm định")
            det_table = gr.Dataframe(
                headers=["#", "Đối tượng phát hiện", "Độ tin cậy", "Tọa độ Pixel [x1, y1, x2, y2]", "% Diện tích ảnh"],
                datatype=["number", "str", "str", "str", "str"],
                label="Chi tiết các vùng phát hiện",
                interactive=False
            )

    if example_files:
        gr.Examples(
            examples=[[f, 0.25, 0.45] for f in example_files],
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, status_box, det_table],
            fn=inspect_noodle_image,
            cache_examples=False,
            label="Ảnh mẫu từ Tập Test (Bấm vào ảnh bất kỳ để chạy thử ngay)"
        )

    run_btn.click(
        fn=inspect_noodle_image,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, status_box, det_table]
    )
    input_image.change(
        fn=inspect_noodle_image,
        inputs=[input_image, conf_slider, iou_slider],
        outputs=[output_image, status_box, det_table]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True)
