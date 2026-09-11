import { useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const IMAGE_MAX_SIZE = 10 * 1024 * 1024; // 10 MB
const VIDEO_MAX_SIZE = 10 * 1024 * 1024; // 20 MB

const IMAGE_TYPES = [
  "image/jpeg",
  "image/png",
];

const VIDEO_TYPES = [
  "video/mp4",
];

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [confidence, setConfidence] = useState(0.5);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [result, setResult] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);

  const [mode, setMode] = useState(null);

  /*
   * Clean up object URLs when component is unmounted
   * or when a new file/result replaces the old one.
   */
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }

      if (resultUrl) {
        URL.revokeObjectURL(resultUrl);
      }
    };
  }, [previewUrl, resultUrl]);

  /*
   * Handle file selection
   */
  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) {
      return;
    }

    setError("");
    setResult(null);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (resultUrl) {
      URL.revokeObjectURL(resultUrl);
    }

    setResultUrl(null);

    /*
     * Validate file type
     */
    const isImage = IMAGE_TYPES.includes(selectedFile.type);
    const isVideo = VIDEO_TYPES.includes(selectedFile.type);

    if (!isImage && !isVideo) {
      setFile(null);
      setPreviewUrl(null);
      setMode(null);
      setError(
        "Unsupported file type. Please select JPG, JPEG, PNG, or MP4."
      );
      return;
    }

    /*
     * Validate file size
     */
    if (isImage && selectedFile.size > IMAGE_MAX_SIZE) {
      setFile(null);
      setPreviewUrl(null);
      setMode(null);
      setError("Image file is too large. Maximum size is 10 MB.");
      return;
    }

    if (isVideo && selectedFile.size > VIDEO_MAX_SIZE) {
      setFile(null);
      setPreviewUrl(null);
      setMode(null);
      setError("Video file is too large. Maximum size is 20 MB.");
      return;
    }

    /*
     * Save selected file
     */
    setFile(selectedFile);

    if (isImage) {
      setMode("image");
    } else {
      setMode("video");
    }

    /*
     * Create local preview
     */
    const localPreviewUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(localPreviewUrl);
  };

  /*
   * Detect image or video
   */
  const handleDetect = async () => {
    if (!file) {
      setError("Please select an image or video first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    if (resultUrl) {
      URL.revokeObjectURL(resultUrl);
      setResultUrl(null);
    }

    try {
      const formData = new FormData();

      formData.append("file", file);
      formData.append("confidence", confidence);

      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      /*
      * Handle HTTP errors
      */
      if (!response.ok) {
        let errorMessage = `Request failed with status ${response.status}.`;

        try {
          const errorData = await response.json();

          if (errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          // Response was not JSON.
        }

        throw new Error(errorMessage);
      }

      /*
      * Backend returns raw binary data:
      *
      * Image -> image/jpeg
      * Video -> video/mp4
      *
      * Therefore we MUST use response.blob().
      */
    const resultBlob = await response.blob();

    const resultObjectUrl = URL.createObjectURL(resultBlob);

    setResultUrl(resultObjectUrl);

    /*
    * Read detection metadata from HTTP header
    */
    const detectionHeader = response.headers.get(
      "X-Detection-Data"
    );

    let detectionData = null;

    if (detectionHeader) {
      try {
        detectionData = JSON.parse(detectionHeader);
      } catch {
        console.error(
          "Failed to parse detection metadata."
        );
      }
    }

    /*
    * Save result information
    */
    setResult({
      message:
        mode === "image"
          ? "Image processed successfully."
          : "Video processed successfully.",
      
      processing_time: detectionData?.processing_time,

      count: detectionData?.count ?? 0, // tong so detection qua tat ca cac frame, khong fai so object trong video

      detections:
        detectionData?.detections ?? [],

      frames:
        detectionData?.frames,

      counts:
        detectionData?.counts,
    });

    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred."
      );
    } finally {
      setLoading(false);
    }
  };

  /*
   * Clear everything
   */
  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setMode(null);
    setError("");
    setResult(null);

    if (resultUrl) {
      URL.revokeObjectURL(resultUrl);
    }

    setResultUrl(null);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>YOLO Object Detection</h1>

        <p>
          Upload an image or MP4 video and detect objects using YOLO.
        </p>
      </header>

      <main className="container">

        {/* =========================
            FILE UPLOAD
        ========================== */}
        <section className="card">
          <h2>1. Upload file</h2>

          <label className="file-label">
            Select image or video
            <input
              type="file"
              accept=".jpg,.jpeg,.png,.mp4,image/jpeg,image/png,video/mp4"
              onChange={handleFileChange}
            />
          </label>

          <p className="hint">
            Supported: JPG, JPEG, PNG, MP4
          </p>

          {file && (
            <div className="file-info">
              <strong>Selected file:</strong>

              <span>{file.name}</span>

              <span>
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </span>
            </div>
          )}
        </section>

        {/* =========================
            PREVIEW
        ========================== */}
        {previewUrl && (
          <section className="card">
            <h2>2. Preview</h2>

            {mode === "image" && (
              <img
                className="preview-image"
                src={previewUrl}
                alt="Selected preview"
              />
            )}

            {mode === "video" && (
              <video
                className="preview-video"
                src={previewUrl}
                controls
              />
            )}
          </section>
        )}

        {/* =========================
            CONFIDENCE
        ========================== */}
        <section className="card">
          <h2>3. Detection settings</h2>

          <div className="confidence-header">
            <label htmlFor="confidence">
              Confidence threshold
            </label>

            <strong>
              {confidence.toFixed(2)}
            </strong>
          </div>

          <input
            id="confidence"
            className="confidence-slider"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confidence}
            onChange={(event) =>
              setConfidence(Number(event.target.value))
            }
          />

          <div className="slider-labels">
            <span>0.00</span>
            <span>1.00</span>
          </div>
        </section>

        {/* =========================
            ACTION BUTTONS
        ========================== */}
        <section className="actions">
          <button
            className="detect-button"
            onClick={handleDetect}
            disabled={!file || loading}
          >
            {loading ? "Detecting..." : "Detect"}
          </button>

          <button
            className="reset-button"
            onClick={handleReset}
            disabled={loading}
          >
            Reset
          </button>
        </section>

        {/* =========================
            LOADING
        ========================== */}
        {loading && (
          <section className="status loading">
            <div className="spinner"></div>

            <p>
              Processing your file. Please wait...
            </p>
          </section>
        )}

        {/* =========================
            ERROR
        ========================== */}
        {error && (
          <section className="status error">
            <h2>Error</h2>

            <p>{error}</p>
          </section>
        )}

        {/* =========================
            RESULT
        ========================== */}
        {result && !loading && (
          <section className="card result-card">
            <h2>4. Detection result</h2>

            {/* Annotated image */}
            {mode === "image" && resultUrl && (
              <div className="result-media">
                <img
                  src={resultUrl}
                  alt="Detection result"
                  className="result-image"
                />
              </div>
            )}

            {/* Annotated video */}
            {mode === "video" && resultUrl && (
              <div className="result-media">
                <video
                  src={resultUrl}
                  controls
                  className="result-video"
                />
              </div>
            )}

            {/* Result information */}
            <div className="result-info">

              {result.message && (
                <p>
                  <strong>Message:</strong>{" "}
                  {result.message}
                </p>
              )}

              {result.processing_time !== undefined && (
                <p>
                  <strong>Processing time:</strong>{" "}
                  {result.processing_time} seconds
                </p>
              )}

              {result.frames !== undefined && (
                <p>
                  <strong>Frames processed:</strong>{" "}
                  {result.frames}
                </p>
              )}

              {result.count !== undefined && (
                <p>
                  <strong>Objects detected:</strong>{" "}
                  {result.count}
                </p>
              )}

              {/* Video summary counts */}
              {result.counts && (
                <div className="counts">
                  <h3>Detection summary</h3>

                  {Object.entries(result.counts).map(
                    ([className, count]) => (
                      <div
                        className="count-row"
                        key={className}
                      >
                        <span>{className}</span>

                        <strong>{count}</strong>
                      </div>
                    )
                  )}
                </div>
              )}

              {/* Image detection details */}
              {result.detections &&
                result.detections.length > 0 && (
                  <div className="detections">
                    <h3>Detected objects</h3>

                    <div className="table-wrapper">
                      <table>
                        <thead>
                          <tr>
                            <th>Class</th>
                            <th>Confidence</th>
                            <th>Bounding box</th>
                          </tr>
                        </thead>

                        <tbody>
                          {result.detections.map(
                            (detection, index) => (
                              <tr key={index}>
                                <td>
                                  {detection.class ||
                                    detection.class_name ||
                                    "Unknown"}
                                </td>

                                <td>
                                  {formatConfidence(
                                    detection.confidence
                                  )}
                                </td>

                                <td>
                                  {formatBoundingBox(
                                    detection.bbox
                                  )}
                                </td>
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

/*
 * Convert backend relative URL into absolute URL.
 *
 * Example:
 * /outputs/result.mp4
 *
 * becomes:
 * http://localhost:8000/outputs/result.mp4
 */
function makeAbsoluteUrl(url) {
  if (!url) {
    return null;
  }

  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }

  if (url.startsWith("/")) {
    return `${API_URL}${url}`;
  }

  return `${API_URL}/${url}`;
}

/*
 * Format confidence value
 */
function formatConfidence(value) {
  if (value === undefined || value === null) {
    return "-";
  }

  const number = Number(value);

  if (Number.isNaN(number)) {
    return value;
  }

  return `${(number * 100).toFixed(1)}%`;
}

/*
 * Format bounding box
 */
function formatBoundingBox(bbox) {
  if (!bbox || !Array.isArray(bbox)) {
    return "-";
  }

  return `[${bbox.join(", ")}]`;
}

export default App;