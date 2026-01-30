# ===================== CORE =====================
import os, cv2, hashlib, datetime, json, time, random, html
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Any
from multiprocessing import Manager

# ===================== ML =====================
import torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from PIL import Image
import open_clip
import imagehash

# ===================== OSINT =====================
from duckduckgo_search import DDGS

# ===================== REPORT =====================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as PDFImage

# ===================== CPU SAFETY =====================
torch.set_num_threads(4)
device = torch.device("cpu")

# ===================== PATHS =====================
BASE = Path.cwd()
FRAMES, FACES, OBJECTS, REPORTS = [BASE / p for p in ("frames", "faces", "objects", "reports")]
for p in (FRAMES, FACES, OBJECTS, REPORTS):
    p.mkdir(exist_ok=True)

# ===================== MODELS =====================
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

yolo = YOLO("yolov8m.pt")

clip_model, _, clip_pre = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model = clip_model.to(device)
clip_model.eval()

# ===================== UTILS =====================
def sha256(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def phash(p):
    with Image.open(p) as im:
        return str(imagehash.phash(im))

def extract_frames(video):
    os.system(
        f'ffmpeg -y -i "{video}" '
        'ffmpeg -i input.mp4 -vf "scale=640:360" -pix_fmt yuvj420p output_%04d.jpg'
    )

# ===================== VIDEO META =====================
def video_info(path):
    cap = cv2.VideoCapture(path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    info["duration_sec"] = info["frames"] / info["fps"] if info["fps"] else 0
    cap.release()
    return info
  
# ===================== CONTROLLED KEYFRAME EXTRACTION =====================
def extract_video_keyframes(path, count=6, max_seconds=10):
    count = max(5, min(count, 10))
    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 1  # safety fallback

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 🔒 LIMIT TO FIRST N SECONDS ONLY
    max_frames_timebound = int(fps * max_seconds)
    usable_frames = min(total_frames, max_frames_timebound)

    idxs = np.linspace(
        0,
        max(usable_frames - 1, 0),
        count
    ).astype(int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            out = FRAMES / f"video_key_{i}.jpg"
            cv2.imwrite(str(out), frame)
            frames.append(str(out))

    cap.release()
    return frames


# ===================== CLIP KEYWORDS =====================
CLIP_VOCAB = [
    "person","man","woman","face","selfie","crowd","group photo",
    "protest","meeting","interview","street","indoor","outdoor",
    "car","bike","truck","weapon","gun","knife",
    "logo","brand","document","passport","identity card","license",
    "screen","laptop","mobile phone","chat screenshot",
    "news","social media","profile picture","youtube video",
    "cctv footage","surveillance camera","security footage",
    "digital illustration","sci-fi city","robot",
    "futuristic interface","surveillance drone","cyberpunk"
]

CLIP_MIN_SIM = 0.22

ILLUSTRATION_HINTS = {
    "illustration","digital","cgi","3d","render",
    "sci-fi","cyberpunk","robot","futuristic"
}

def clip_keywords(image_path, topk=7):
    try:
        img_pil = Image.open(image_path).convert("RGB")
        img = clip_pre(img_pil).unsqueeze(0).to(device)
    except Exception:
        return []

    with torch.no_grad():
        feat = clip_model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)

    tokens = open_clip.tokenize(CLIP_VOCAB).to(device)
    with torch.no_grad():
        txt = clip_model.encode_text(tokens)
        txt /= txt.norm(dim=-1, keepdim=True)

    sim = (feat @ txt.T)[0]
    values, idxs = sim.topk(min(topk * 2, len(CLIP_VOCAB)))

    results = []
    for v, i in zip(values.tolist(), idxs.tolist()):
        if v >= CLIP_MIN_SIM:
            results.append(CLIP_VOCAB[i])

    # Illustration / CGI suppression
    if any(h in " ".join(results) for h in ILLUSTRATION_HINTS):
        results = [k for k in results if k not in {"gun","knife","weapon","meeting"}]

    return list(dict.fromkeys(results))[:topk]

def build_search_query(keywords, limit=4):
    clean = [k for k in keywords if len(k.split()) <= 3]
    return " ".join(clean[:limit])

# ===================== DDG SAFE MODE =====================
DDG_MAX_RETRIES = 4
DDG_BASE_DELAY = 2.0

def safe_ddg_call(fn, *args, **kwargs):
    for attempt in range(DDG_MAX_RETRIES):
        try:
            time.sleep(DDG_BASE_DELAY + random.uniform(0.5, 1.5))
            return fn(*args, **kwargs)
        except Exception:
            if attempt == DDG_MAX_RETRIES - 1:
                return []
            time.sleep((2 ** attempt) + random.uniform(1, 2))

# ===================== DUCKDUCKGO OSINT =====================
def ddg_live_osint(query, images=8, videos=4):
    if not query.strip():
        return {"images": [], "videos": []}

    out = {"images": [], "videos": []}
    with DDGS() as d:
        for r in safe_ddg_call(d.images, query, max_results=images) or []:
            out["images"].append({
                "title": r.get("title"),
                "image_url": r.get("image"),
                "page_url": r.get("url"),
                "thumbnail": r.get("thumbnail")
            })
        for r in safe_ddg_call(d.videos, query, max_results=videos) or []:
            out["videos"].append({
                "title": r.get("title"),
                "url": r.get("content"),
                "publisher": r.get("publisher")
            })
    return out

# ===================== IMAGE PROCESS =====================
def process_image(path, corr):
    img = cv2.imread(path)
    if img is None:
        return

    h, w = img.shape[:2]
    faces = face_app.get(img)

    for i, f in enumerate(faces):
        x1, y1, x2, y2 = map(int, f.bbox)
        crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if crop.size:
            name = f"{Path(path).stem}_face_{i}.jpg"
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(FACES / name)
            corr.setdefault(name, [])

    face_keys = list(corr.keys())

    results = yolo(img, conf=0.45, max_det=6)[0]
    for i, b in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, b)
        crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if crop.size:
            oname = f"{Path(path).stem}_obj_{i}.jpg"
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(OBJECTS / oname)
            for fk in face_keys:
                corr[fk].append(oname)

# ===================== CLUSTER =====================
def cluster(embeds):
    if len(embeds) < 2:
        return {}
    X = np.array(list(embeds.values()))
    labels = DBSCAN(eps=0.4, metric="cosine", min_samples=2).fit_predict(X)
    out = {}
    for k, l in zip(embeds.keys(), labels):
        if l != -1:
            out.setdefault(str(l), []).append(k)
    return out

# ===================== PDF =====================
def build_pdf(report, out):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out, pagesize=A4)

    elems = [
        Paragraph("OSINT Visual Intelligence Report", styles["Title"]),
        Spacer(1, 20)
    ]

    def sec(title):
        elems.extend([
            Spacer(1, 14),
            Paragraph(title, styles["Heading2"]),
            Spacer(1, 8)
        ])

    # ===================== MEDIA =====================
    sec("Media Analysis")

    media = report["media"]

    # Metadata (clean, readable – no raw JSON removal)
    elems.append(Paragraph(f"<b>Type:</b> {media.get('type')}", styles["Normal"]))
    elems.append(Paragraph(f"<b>Path:</b> {html.escape(media.get('path',''))}", styles["Normal"]))
    if media.get("keywords"):
        elems.append(Paragraph(
            "<b>Keywords:</b> " + ", ".join(media["keywords"]),
            styles["Normal"]
        ))

    # Image preview
    if media.get("type") == "image":
        try:
            elems.append(Spacer(1, 10))
            elems.append(PDFImage(
                media["path"],
                width=220,
                height=140,
                kind="proportional"
            ))
        except Exception:
            pass

    # Video keyframes (max 2)
    for kf in media.get("keyframes", [])[:2]:
        try:
            elems.append(Spacer(1, 8))
            elems.append(PDFImage(
                kf,
                width=220,
                height=140,
                kind="proportional"
            ))
        except Exception:
            pass

    # ===================== OSINT =====================
    sec("OSINT Results")

    osint = media.get("osint", {})

    # ---------- IMAGE OSINT ----------
    if osint.get("images"):
        elems.append(Paragraph("<b>Image Sources:</b>", styles["Normal"]))

        for img in osint["images"][:2]:
            title = html.escape(img.get("title") or "Source page")
            page_url = img.get("page_url")

            # Thumbnail (visual reference)
            if img.get("thumbnail"):
                try:
                    elems.append(Spacer(1, 6))
                    elems.append(PDFImage(
                        img["thumbnail"],
                        width=120,
                        height=80,
                        kind="proportional"
                    ))
                except Exception:
                    pass

            # SAFE LINK → page_url ONLY
            if page_url:
                elems.append(Paragraph(
                    f'<a href="{page_url}">{title}</a>',
                    styles["Normal"]
                ))

    # ---------- VIDEO OSINT ----------
    if osint.get("videos"):
        elems.append(Spacer(1, 10))
        elems.append(Paragraph("<b>Video Sources:</b>", styles["Normal"]))

        for v in osint["videos"][:2]:
            title = html.escape(v.get("title") or "Video source")
            url = v.get("url")
            if url:
                elems.append(Paragraph(
                    f'<a href="{url}">{title}</a>',
                    styles["Normal"]
                ))

    # ===================== FACES =====================
    sec("Faces OSINT")

    if not report["faces"]:
        elems.append(Paragraph("No faces detected.", styles["Normal"]))
    else:
        for f, d in report["faces"].items():
            elems.append(Paragraph(f"<b>{f}</b>", styles["Normal"]))
            elems.append(Paragraph("SHA256: " + d["hash"], styles["Normal"]))
            elems.append(Paragraph(
                "Keywords: " + ", ".join(d["keywords"]),
                styles["Normal"]
            ))

            for img in d["duckduckgo"].get("images", [])[:2]:
                page_url = img.get("page_url")
                title = html.escape(img.get("title") or "Source page")
                if page_url:
                    elems.append(Paragraph(
                        f'<a href="{page_url}">{title}</a>',
                        styles["Normal"]
                    ))

    # ===================== CLUSTERS =====================
    sec("Face Clusters")
    for cid, members in report["clusters"].items():
        elems.append(Paragraph(
            f"Cluster {cid}: {', '.join(members)}",
            styles["Normal"]
        ))

    # ===================== META =====================
    sec("Metadata")
    elems.append(Paragraph(
        f"Generated (UTC): {report['generated']}",
        styles["Normal"]
    ))

    doc.build(elems)


# ===================== PIPELINE (UNCHANGED LOGIC) =====================
def run_pipeline(input_file: str,
                 progress_cb: Optional[Callable] = None,
                 cancel_event: Optional[Any] = None):

    corr = Manager().dict()
    media_osint = {}

    if input_file.lower().endswith((".mp4",".avi",".mov",".mkv")):
        extract_frames(input_file)
        keyframes = extract_video_keyframes(input_file, count=6)

        all_kw = []
        for k in keyframes:
            kw = clip_keywords(k)
            all_kw.extend(kw)
            media_osint[k] = ddg_live_osint(build_search_query(kw))

        video_kw = list(dict.fromkeys(all_kw))[:10]
        media_osint["video_text_search"] = ddg_live_osint(build_search_query(video_kw))

        media = {
            "type": "video",
            "path": input_file,
            "info": video_info(input_file),
            "keyframes": keyframes,
            "keywords": video_kw,
            "osint": media_osint
        }
        imgs = keyframes
    else:
        kw = clip_keywords(input_file)
        media = {
            "type": "image",
            "path": input_file,
            "sha256": sha256(input_file),
            "phash": phash(input_file),
            "keywords": kw,
            "osint": ddg_live_osint(build_search_query(kw))
        }
        imgs = [input_file]

    for img in imgs:
        process_image(str(img), corr)

    embeddings, osint_faces = {}, {}
    for f in FACES.iterdir():
        img = cv2.imread(str(f))
        if img is None:
            continue
        faces = face_app.get(img)
        if faces:
            embeddings[f.name] = faces[0].embedding
            kw = clip_keywords(str(f))
            osint_faces[f.name] = {
                "hash": sha256(str(f)),
                "keywords": kw,
                "duckduckgo": ddg_live_osint(build_search_query(kw))
            }

    report = {
        "media": media,
        "faces": osint_faces,
        "clusters": cluster(embeddings),
        "generated": datetime.datetime.utcnow().isoformat()
    }

    pdf = REPORTS / "osint_report.pdf"
    build_pdf(report, str(pdf))
    return {"pdf": str(pdf), "report": report}
