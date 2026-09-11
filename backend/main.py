import time
import io
import os
import subprocess
import tempfile

import cv2
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="YOLO Object Detection API",
    version="1.0.0",
)


# ============================================================
# 2. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Detection-Data"],
)


# ============================================================
# 3. YOLO MODEL
# ============================================================

model = YOLO("yolov8n.pt")


# ============================================================
# 4. CONFIGURATION
# ============================================================

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
}


# ============================================================
# 5. HEALTH CHECK
# ============================================================

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "YOLO Server đang hoạt động!",
    }


# ============================================================
# 6. PREDICTION ENDPOINT
# ============================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
):
    if confidence < 0 or confidence > 1:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1.",
        )
    # --------------------------------------------------------
    # 6.1. CHECK FILE TYPE
    # --------------------------------------------------------

    content_type = file.content_type

    if (
        content_type not in ALLOWED_IMAGE_TYPES
        and content_type not in ALLOWED_VIDEO_TYPES
    ):
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG and MP4 files are allowed.",
        )

    # --------------------------------------------------------
    # 6.2. READ UPLOADED FILE
    # --------------------------------------------------------

    file_bytes = await file.read()

    # --------------------------------------------------------
    # 6.3. CHECK FILE SIZE
    # --------------------------------------------------------

    start_time = time.perf_counter()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size must not exceed 10 MB.",
        )

    # ========================================================
    # 7. IMAGE PROCESSING
    # ========================================================

    if content_type in ALLOWED_IMAGE_TYPES:

        # ----------------------------------------------------
        # 7.1. VALIDATE IMAGE
        # ----------------------------------------------------

        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()

            # verify() consumes the image stream,
            # so the image must be opened again.
            image = Image.open(io.BytesIO(file_bytes))

        except (UnidentifiedImageError, OSError):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid image.",
            )

        # ----------------------------------------------------
        # 7.2. YOLO DETECTION
        # ----------------------------------------------------

        results = model(
            image,
            conf=confidence,
        )

        result = results[0]

        # ----------------------------------------------------
        # 7.3. GET DETECTION INFORMATION
        # ----------------------------------------------------

        detections = []

        boxes = result.boxes

        for box in boxes:

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            confidence_score = float(box.conf[0])

            coordinates = box.xyxy[0].tolist()

            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence_score,
                    "bbox": coordinates,
                }
            )

        # ----------------------------------------------------
        # 7.4. DRAW BOUNDING BOXES
        # ----------------------------------------------------

        annotated_array = result.plot()

        # OpenCV/YOLO returns BGR.
        # PIL expects RGB.
        annotated_image = Image.fromarray(
            annotated_array[..., ::-1]
        )

        # ----------------------------------------------------
        # 7.5. SAVE RESULT TO MEMORY
        # ----------------------------------------------------

        image_buffer = io.BytesIO()

        annotated_image.save(
            image_buffer,
            format="JPEG",
        )

        # ----------------------------------------------------
        # 7.6. RETURN IMAGE + DETECTION METADATA
        # ----------------------------------------------------

        # NOTE:
        # We cannot return the JPEG and JSON metadata
        # directly as a normal JSON response at the same time.
        #
        # Therefore, for this step we will temporarily
        # return the metadata through HTTP headers.

        import json

        detection_data = json.dumps(
            {
                "count": len(detections),
                "detections": detections,
                "processing_time": round(processing_time, 3),
            }
        )

        processing_time = time.perf_counter() - start_time

        return Response(
            content=image_buffer.getvalue(),
            media_type="image/jpeg",
            headers={
                "X-Detection-Data": detection_data,
            },
        )
    # ========================================================
    # 8. VIDEO PROCESSING
    # ========================================================

    if content_type in ALLOWED_VIDEO_TYPES:

        input_path = None
        raw_output_path = None
        final_output_path = None

        cap = None
        writer = None

        try:

            # ------------------------------------------------
            # 8.1. SAVE UPLOADED VIDEO TO TEMPORARY FILE
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            ) as input_file:

                input_file.write(file_bytes)
                input_path = input_file.name

            # ------------------------------------------------
            # 8.2. CREATE TEMPORARY RAW OUTPUT FILE
            # ------------------------------------------------

            raw_fd, raw_output_path = tempfile.mkstemp(
                suffix="_raw.mp4",
            )

            os.close(raw_fd)

            # ------------------------------------------------
            # 8.3. CREATE TEMPORARY H.264 OUTPUT FILE
            # ------------------------------------------------

            final_fd, final_output_path = tempfile.mkstemp(
                suffix="_h264.mp4",
            )

            os.close(final_fd)

            # ------------------------------------------------
            # 8.4. OPEN VIDEO
            # ------------------------------------------------

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                raise HTTPException(
                    status_code=400,
                    detail="Cannot open the uploaded video.",
                )

            # ------------------------------------------------
            # 8.5. GET VIDEO INFORMATION
            # ------------------------------------------------

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 30

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid video dimensions.",
                )

            # ------------------------------------------------
            # 8.6. CREATE VIDEO WRITER
            # ------------------------------------------------

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                raw_output_path,
                fourcc,
                fps,
                (width, height),
            )

            if not writer.isOpened():
                raise HTTPException(
                    status_code=500,
                    detail="Cannot create output video.",
                )

            # ------------------------------------------------
            # 8.7. PROCESS EACH FRAME WITH YOLO
            # ------------------------------------------------

                        # ------------------------------------------------
            # 8.7.1. DETECTION STATISTICS
            # ------------------------------------------------

            frame_count = 0
            total_objects = 0
            class_counts = {}

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                results = model(
                    frame,
                    conf=confidence,
                )

                result = results[0]

                # --------------------------------------------
                # COUNT DETECTED OBJECTS
                # --------------------------------------------

                boxes = result.boxes

                total_objects += len(boxes)

                for box in boxes:

                    class_id = int(box.cls[0])

                    class_name = model.names[class_id]

                    class_counts[class_name] = (
                        class_counts.get(class_name, 0) + 1
                    )

                # --------------------------------------------
                # DRAW BOUNDING BOXES
                # --------------------------------------------

                annotated_frame = result.plot()

                writer.write(annotated_frame)
            # ------------------------------------------------
            # 8.8. RELEASE OPENCV RESOURCES
            # ------------------------------------------------

            cap.release()
            cap = None

            writer.release()
            writer = None

            # ------------------------------------------------
            # 8.9. CONVERT VIDEO TO H.264 USING FFMPEG
            # ------------------------------------------------

            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-i",
                raw_output_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                final_output_path,
            ]

            subprocess.run(
                ffmpeg_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            # ------------------------------------------------
            # 8.10. READ PROCESSED VIDEO
            # ------------------------------------------------

            with open(
                final_output_path,
                "rb",
            ) as video_file:

                result_video = video_file.read()

            # ------------------------------------------------
            # 8.11. RETURN VIDEO + DETECTION METADATA
            # ------------------------------------------------

            import json

            processing_time = time.perf_counter() - start_time
            
            detection_data = json.dumps(
                {
                    "frames": frame_count,
                    "count": total_objects,
                    "counts": class_counts,
                    "processing_time": round(processing_time, 3),
                }
            )

            return Response(
                content=result_video,
                media_type="video/mp4",
                headers={
                    "X-Detection-Data": detection_data,
                },
            )

        except subprocess.CalledProcessError:

            raise HTTPException(
                status_code=500,
                detail="FFmpeg failed to convert the processed video.",
            )

        except HTTPException:
            raise

        except Exception as error:

            print(
                "VIDEO PROCESSING ERROR:",
                error,
            )

            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing the video.",
            )

        finally:

            # ------------------------------------------------
            # RELEASE OPENCV RESOURCES
            # ------------------------------------------------

            if cap is not None:
                cap.release()

            if writer is not None:
                writer.release()

            # ------------------------------------------------
            # DELETE TEMPORARY FILES
            # ------------------------------------------------

            temporary_files = [
                input_path,
                raw_output_path,
                final_output_path,
            ]

            for path in temporary_files:

                if path is not None and os.path.exists(path):

                    try:
                        os.remove(path)

                    except OSError:
                        pass

