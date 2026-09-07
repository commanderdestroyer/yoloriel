import io
import os
import tempfile
import cv2
import subprocess

from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError

from fastapi.middleware.cors import CORSMiddleware


# ==========================================
# 1. KHỞI TẠO FASTAPI
# ==========================================

app = FastAPI(title="YOLO Object Detection API")


# ==========================================
# 2. CẤU HÌNH CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 3. TẢI MODEL YOLO MỘT LẦN
# ==========================================

model = YOLO("yolov8n.pt")


# ==========================================
# 4. CẤU HÌNH FILE
# ==========================================

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
}


# ==========================================
# 5. ENDPOINT KIỂM TRA SERVER
# ==========================================

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "YOLO Server đang hoạt động!"
    }


# ==========================================
# 6. ENDPOINT PREDICT
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # ==========================================
    # BƯỚC 1: KIỂM TRA LOẠI FILE
    # ==========================================

    content_type = file.content_type

    if (
        content_type not in ALLOWED_IMAGE_TYPES
        and content_type not in ALLOWED_VIDEO_TYPES
    ):
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG and MP4 files are allowed."
        )


    # ==========================================
    # BƯỚC 2: ĐỌC FILE
    # ==========================================

    file_bytes = await file.read()


    # ==========================================
    # BƯỚC 3: KIỂM TRA FILE SIZE
    # ==========================================

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size must not exceed 10 MB."
        )


    # ==========================================
    # BƯỚC 4: NẾU LÀ ẢNH
    # ==========================================

    if content_type in ALLOWED_IMAGE_TYPES:

        # --------------------------------------
        # KIỂM TRA FILE CÓ PHẢI ẢNH THẬT KHÔNG
        # --------------------------------------

        try:
            image = Image.open(io.BytesIO(file_bytes))

            # Kiểm tra file ảnh có hợp lệ không
            image.verify()

            # verify() consume file nên phải mở lại
            image = Image.open(io.BytesIO(file_bytes))

        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid image."
            )


        # --------------------------------------
        # YOLO NHẬN DIỆN ẢNH
        # --------------------------------------

        results = model(image)


        # --------------------------------------
        # VẼ BOUNDING BOX
        # --------------------------------------

        annotated_array = results[0].plot()


        # --------------------------------------
        # CHUYỂN BGR → RGB
        # --------------------------------------

        annotated_image = Image.fromarray(
            annotated_array[..., ::-1]
        )


        # --------------------------------------
        # LƯU ẢNH VÀO RAM
        # --------------------------------------

        img_buffer = io.BytesIO()

        annotated_image.save(
            img_buffer,
            format="JPEG"
        )


        # --------------------------------------
        # TRẢ ẢNH VỀ FRONTEND
        # --------------------------------------

        return Response(
            content=img_buffer.getvalue(),
            media_type="image/jpeg"
        )


    # ==========================================
    # BƯỚC 5: NẾU LÀ VIDEO MP4
    # ==========================================

    elif content_type in ALLOWED_VIDEO_TYPES:

        input_path = None
        raw_output_path = None
        final_output_path = None

        cap = None
        out = None

        try:

            # --------------------------------------
            # 1. LƯU VIDEO UPLOAD VÀO FILE TẠM
            # --------------------------------------

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            input_file.write(file_bytes)
            input_file.close()

            input_path = input_file.name


            # --------------------------------------
            # 2. TẠO FILE TẠM CHO VIDEO RAW
            # --------------------------------------

            raw_fd, raw_output_path = tempfile.mkstemp(
                suffix="_raw.mp4"
            )

            # Đóng file descriptor.
            # OpenCV sẽ tự mở file bằng đường dẫn.
            os.close(raw_fd)


            # --------------------------------------
            # 3. TẠO FILE TẠM CHO VIDEO H.264
            # --------------------------------------

            final_fd, final_output_path = tempfile.mkstemp(
                suffix="_h264.mp4"
            )

            os.close(final_fd)


            # --------------------------------------
            # 4. MỞ VIDEO BẰNG OPENCV
            # --------------------------------------

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                raise HTTPException(
                    status_code=400,
                    detail="Cannot open the uploaded video."
                )


            # --------------------------------------
            # 5. LẤY THÔNG TIN VIDEO
            # --------------------------------------

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 30


            # Kiểm tra kích thước video
            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid video dimensions."
                )


            # --------------------------------------
            # 6. TẠO VIDEO WRITER
            # --------------------------------------

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                raw_output_path,
                fourcc,
                fps,
                (width, height)
            )


            # Kiểm tra VideoWriter
            if not out.isOpened():
                raise HTTPException(
                    status_code=500,
                    detail="Cannot create output video."
                )


            # --------------------------------------
            # 7. ĐỌC TỪNG FRAME + YOLO
            # --------------------------------------

            while True:

                ret, frame = cap.read()

                if not ret:
                    break


                # YOLO nhận diện frame
                results = model(frame)


                # Vẽ bounding box
                annotated_frame = results[0].plot()


                # Ghi frame vào video output
                out.write(annotated_frame)


            # --------------------------------------
            # 8. ĐÓNG VIDEO
            # --------------------------------------

            cap.release()
            cap = None

            out.release()
            out = None


            # --------------------------------------
            # 9. DÙNG FFMPEG CHUYỂN SANG H.264
            # --------------------------------------

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                raw_output_path,

                # Video codec H.264
                "-c:v",
                "libx264",

                # Pixel format tương thích trình duyệt
                "-pix_fmt",
                "yuv420p",

                # Copy audio nếu video có audio
                "-c:a",
                "aac",

                final_output_path
            ]


            subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )


            # --------------------------------------
            # 10. ĐỌC VIDEO KẾT QUẢ
            # --------------------------------------

            with open(
                final_output_path,
                "rb"
            ) as f:

                result_video = f.read()


            # --------------------------------------
            # 11. TRẢ VIDEO CHO FRONTEND
            # --------------------------------------

            return Response(
                content=result_video,
                media_type="video/mp4"
            )


        except subprocess.CalledProcessError:
            raise HTTPException(
                status_code=500,
                detail="FFmpeg failed to convert the processed video."
            )


        except HTTPException:
            raise


        except Exception as e:
            print("VIDEO PROCESSING ERROR:", e)

            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing the video."
            )


        finally:

            # --------------------------------------
            # ĐẢM BẢO VIDEO ĐƯỢC ĐÓNG
            # --------------------------------------

            if cap is not None:
                cap.release()

            if out is not None:
                out.release()


            # --------------------------------------
            # XÓA FILE TẠM
            # --------------------------------------

            for path in [
                input_path,
                raw_output_path,
                final_output_path
            ]:

                if path is not None and os.path.exists(path):

                    try:
                        os.remove(path)

                    except Exception:
                        pass