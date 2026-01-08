import gradio as gr
import os
import shutil
from PIL import Image
import cv2
import glob

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn folder tạm để lưu ảnh upload và kết quả
INPUT_DIR = "gradio_temp_input"
OUTPUT_DIR = "gradio_temp_output"
BBOX_DIR = "gradio_temp_input_bbox" # Detectron2 sẽ lưu bbox tạm vào đây

# Tên cấu hình model (tương ứng với tên folder trong thư mục checkpoints)
# Trong notebook của bạn, model được load từ: checkpoints/coco_mask/latest_net_GF.pth
MODEL_NAME = "coco_mask" 

# Đảm bảo các thư mục tồn tại
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_instcolorization(input_image):
    if input_image is None:
        return None

    # 1. DỌN DẸP THƯ MỤC TẠM (Để tránh lấy nhầm ảnh cũ)
    def clean_folder(folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    clean_folder(INPUT_DIR)
    clean_folder(OUTPUT_DIR)
    # Folder bbox thường được tạo tự động bởi script, nhưng ta clean cho chắc
    clean_folder(BBOX_DIR) 

    # 2. LƯU ẢNH UPLOAD
    # Gradio input là PIL Image, ta lưu thành file jpg
    input_path = os.path.join(INPUT_DIR, "image.jpg")
    input_image.save(input_path)

    print(f"--- Đã nhận ảnh, bắt đầu xử lý... ---")

    # 3. BƯỚC 1: PHÁT HIỆN VẬT THỂ (BBOX)
    # Lệnh dựa trên notebook: !python inference_bbox.py ...
    # Lưu ý: Script inference_bbox.py thường tự tạo folder _bbox dựa trên tên folder input
    # Ta cần chỉnh code hoặc để nó tự chạy. Ở đây ta gọi lệnh chuẩn.
    cmd_bbox = (
        f"python inference_bbox.py "
        f"--test_img_dir {INPUT_DIR} "
        f"--filter_no_obj " # Lọc bỏ ảnh không có object (nhưng ta chỉ có 1 ảnh nên cứ chạy)
    )
    print(f"Running BBox: {cmd_bbox}")
    os.system(cmd_bbox)

    # 4. BƯỚC 2: TÔ MÀU (FUSION)
    # Lệnh dựa trên notebook: !python test_fusion.py ...
    # Tham số --model fusion, --fineSize 256
    cmd_fusion = (
        f"python test_fusion.py "
        f"--name {MODEL_NAME} "         # Tên để load weight trong checkpoints/MODEL_NAME
        f"--sample_p 1.0 "
        f"--model fusion "
        f"--fineSize 256 "
        f"--test_img_dir {INPUT_DIR} "  # Folder chứa ảnh gốc
        f"--results_img_dir {OUTPUT_DIR} " # Folder chứa kết quả
    )
    print(f"Running Fusion: {cmd_fusion}")
    os.system(cmd_fusion)

    # 5. LẤY KẾT QUẢ
    # Kết quả thường là file .png hoặc .jpg trong folder output
    # InstColorization thường ghép ảnh gốc và ảnh màu, hoặc lưu riêng.
    # Ta tìm tất cả file ảnh trong output dir
    output_files = glob.glob(os.path.join(OUTPUT_DIR, "*"))
    
    if not output_files:
        return None
    
    # Lấy file kết quả đầu tiên tìm thấy
    # InstColorization thường đặt tên file output giống file input
    result_path = output_files[0]
    
    # Mở ảnh để trả về cho Gradio
    return Image.open(result_path)

# --- GIAO DIỆN GRADIO ---
with gr.Blocks(title="InstColorization Demo") as demo:
    gr.Markdown("# 🎨 InstColorization Demo")
    gr.Markdown("Upload ảnh đen trắng để tô màu sử dụng model InstColorization.")
    
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Ảnh đầu vào (Input)")
            btn = gr.Button("Tô màu (Colorize)", variant="primary")
        with gr.Column():
            out = gr.Image(type="pil", label="Kết quả (Result)")
    
    btn.click(fn=run_instcolorization, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch(share=True)