import os
import sys
import shutil
from pathlib import Path
from multiprocessing import freeze_support, Event

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

# ===================== MULTIPROCESS SAFETY (WINDOWS) =====================
if __name__ == "__main__":
    freeze_support()

try:
    from pipeline import run_pipeline
    PIPELINE_AVAILABLE = True
except Exception:
    run_pipeline = None
    PIPELINE_AVAILABLE = False

# ===================== PATH CONFIG =====================
BASE_DIR = Path.cwd()

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_FOLDER = STATIC_DIR / "uploads"
REPORTS_FOLDER = BASE_DIR / "reports"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
REPORTS_FOLDER.mkdir(exist_ok=True)

# ===================== FLASK APP =====================
app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR)
)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

# ===================== FILE VALIDATION =====================
ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".mp4", ".avi", ".mov", ".mkv"
}

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

# ===================== MAIN UI (MERGED ROUTE) =====================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # ----------- FILE PRESENCE CHECK -----------
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        # ----------- EXTENSION CHECK (Code-2 logic) -----------
        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported file type"}), 400

        save_path = UPLOAD_FOLDER / file.filename
        file.save(save_path)

        # PIPELINE MODE 
        # ======================================================
        if PIPELINE_AVAILABLE:

            cancel_event = Event()

            def progress_callback(*args, **kwargs):
                """
                Compatible with all existing/future pipeline signatures.
                Intentionally preserved even if unused.
                """
                pass

            try:
                result = run_pipeline(
                    input_file=str(save_path),
                    cancel_event=cancel_event,
                    progress_cb=progress_callback
                )
            except Exception as e:
                return jsonify({
                    "error": "Pipeline execution failed",
                    "details": str(e)
                }), 500

            # Preserve Code-2 report exposure
            result["report_pdf"] = "/reports/osint_report.pdf"
            return jsonify(result)


        report = {
            "file": file.filename,
            "status": "Processed successfully",
            "note": "Replace this with your OSINT engine output"
        }

        return jsonify({"report": report})

    return render_template("index.html")

# ===================== REPORT DOWNLOAD =====================
@app.route("/reports/<path:filename>")
def download_report(filename):
    return send_from_directory(
        directory=str(REPORTS_FOLDER),
        path=filename,
        as_attachment=True
    )

# ===================== RESET =====================
@app.route("/reset", methods=["POST"])
def reset_workspace():
    folders = [
        UPLOAD_FOLDER,
        Path("frames"),
        Path("faces"),
        Path("objects"),
        REPORTS_FOLDER
    ]

    for folder in folders:
        if folder.exists():
            for f in folder.iterdir():
                try:
                    if f.is_file():
                        f.unlink()
                except Exception:
                    pass

    return jsonify({"status": "Workspace cleared"})

# ===================== RUN SERVER =====================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=True,
        use_reloader=False
    )