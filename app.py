from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Dict, Any, Optional

app = FastAPI(title="MergeSport Posture Analysis API", version="1.0.0")

# WordPress domainini buraya ekle. Geliştirme için "*" açık bırakıldı.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose


def point(lm, idx: int) -> Dict[str, float]:
    p = lm[idx]
    return {
        "x": float(p.x),
        "y": float(p.y),
        "z": float(p.z),
        "visibility": float(p.visibility),
    }


def safe_point(lm, idx: int, min_visibility: float = 0.25) -> Optional[Dict[str, float]]:
    p = point(lm, idx)
    if p["visibility"] < min_visibility:
        return None
    return p


def dist(a, b) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def line_tilt_deg(a, b) -> Optional[float]:
    if not a or not b:
        return None
    return abs(math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"])))


def vertical_deviation_deg(a, b) -> Optional[float]:
    if not a or not b:
        return None
    angle = abs(math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"])))
    return abs(90 - angle)


def mid(a, b):
    if not a or not b:
        return None
    return {
        "x": (a["x"] + b["x"]) / 2,
        "y": (a["y"] + b["y"]) / 2,
        "z": (a["z"] + b["z"]) / 2,
        "visibility": min(a["visibility"], b["visibility"]),
    }


def joint_angle(a, b, c) -> Optional[float]:
    if not a or not b or not c:
        return None
    ab = (a["x"] - b["x"], a["y"] - b["y"])
    cb = (c["x"] - b["x"], c["y"] - b["y"])
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)
    if mag_ab == 0 or mag_cb == 0:
        return None
    cos_v = max(-1, min(1, dot / (mag_ab * mag_cb)))
    return round(math.degrees(math.acos(cos_v)), 1)


def severity(value: float, mild=0.06, medium=0.12, high=0.18) -> str:
    if value >= high:
        return "belirgin"
    if value >= medium:
        return "orta"
    if value >= mild:
        return "hafif"
    return "normal"


def read_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Görsel okunamadı.")
    return img


def resize_for_pose(img, max_side=1280):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def analyze_image(img_bgr, view: str) -> Dict[str, Any]:
    img_bgr = resize_for_pose(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.35,
    ) as pose:
        result = pose.process(img_rgb)

    if not result.pose_landmarks:
        return {
            "view": view,
            "ok": False,
            "landmark_quality": 0,
            "reason": "Landmark bulunamadı. Fotoğraf tam kadraj, net ışık ve baştan ayağa görünür olmalı.",
            "landmarks": [],
            "angles": {},
            "metrics": {},
        }

    lm = result.pose_landmarks.landmark
    landmarks = [point(lm, i) for i in range(len(lm))]
    visible = sum(1 for p in landmarks if p["visibility"] >= 0.35)
    quality = round((visible / len(landmarks)) * 100)

    nose = safe_point(lm, 0)
    ls = safe_point(lm, 11)
    rs = safe_point(lm, 12)
    le = safe_point(lm, 13)
    re = safe_point(lm, 14)
    lw = safe_point(lm, 15)
    rw = safe_point(lm, 16)
    lh = safe_point(lm, 23)
    rh = safe_point(lm, 24)
    lk = safe_point(lm, 25)
    rk = safe_point(lm, 26)
    la = safe_point(lm, 27)
    ra = safe_point(lm, 28)

    shoulder_mid = mid(ls, rs)
    hip_mid = mid(lh, rh)

    angles = {
        "shoulder_tilt": line_tilt_deg(ls, rs) if ls and rs else None,
        "pelvis_tilt": line_tilt_deg(lh, rh) if lh and rh else None,
        "head_forward": vertical_deviation_deg(nose, shoulder_mid) if nose and shoulder_mid else None,
        "trunk_lean": vertical_deviation_deg(shoulder_mid, hip_mid) if shoulder_mid and hip_mid else None,
        "left_knee": joint_angle(lh, lk, la) if lh and lk and la else None,
        "right_knee": joint_angle(rh, rk, ra) if rh and rk and ra else None,
        "left_elbow": joint_angle(ls, le, lw) if ls and le and lw else None,
        "right_elbow": joint_angle(rs, re, rw) if rs and re and rw else None,
    }

    metrics = {
        "shoulder_asymmetry": 0,
        "pelvis_asymmetry": 0,
        "head_forward_ratio": 0,
        "rounded_shoulder_ratio": 0,
        "knee_valgus_ratio": 0,
    }

    if ls and rs:
        ref = max(0.0001, dist(ls, rs))
        metrics["shoulder_asymmetry"] = abs(ls["y"] - rs["y"]) / ref

    if lh and rh:
        ref = max(0.0001, dist(lh, rh))
        metrics["pelvis_asymmetry"] = abs(lh["y"] - rh["y"]) / ref

    if nose and shoulder_mid and ls and rs:
        ref = max(0.0001, dist(ls, rs))
        metrics["head_forward_ratio"] = abs(nose["x"] - shoulder_mid["x"]) / ref

    if shoulder_mid and hip_mid and ls and rs:
        ref = max(0.0001, dist(ls, rs))
        metrics["rounded_shoulder_ratio"] = abs(shoulder_mid["x"] - hip_mid["x"]) / ref

    if lk and rk and la and ra:
        knee_gap = abs(lk["x"] - rk["x"])
        ankle_gap = abs(la["x"] - ra["x"])
        if ankle_gap > 0:
            metrics["knee_valgus_ratio"] = max(0, (ankle_gap - knee_gap) / ankle_gap)

    return {
        "view": view,
        "ok": True,
        "landmark_quality": quality,
        "landmarks": landmarks,
        "angles": angles,
        "metrics": metrics,
    }


def build_report(front=None, side=None, back=None) -> Dict[str, Any]:
    analyses = [x for x in [front, side, back] if x]
    ok_analyses = [x for x in analyses if x.get("ok")]

    if not ok_analyses:
        return {
            "score": 70,
            "risk": "Sınırlı",
            "analysis_source": "Backend MediaPipe - Landmark alınamadı",
            "landmark_quality": 0,
            "angles": {},
            "levels": {
                "head": "limited",
                "shoulder": "limited",
                "pelvis": "limited",
                "knee": "limited",
            },
            "findings": ["Fotoğraflardan landmark alınamadı."],
            "fixes": ["Wall Posture Hold 2x30 sn", "Dead Bug 2x10", "Face Pull 2x12"],
            "notes": ["Tam analiz için baştan ayağa tam kadraj, iyi ışık ve sade arka plan gerekir."],
            "raw": {"front": front, "side": side, "back": back},
        }

    quality = round(sum(x["landmark_quality"] for x in ok_analyses) / len(ok_analyses))

    angles = {}
    metrics = {
        "shoulder_asymmetry": 0,
        "pelvis_asymmetry": 0,
        "head_forward_ratio": 0,
        "rounded_shoulder_ratio": 0,
        "knee_valgus_ratio": 0,
    }

    for item in ok_analyses:
        for k, v in item.get("angles", {}).items():
            if v is not None and k not in angles:
                angles[k] = v
        for k, v in item.get("metrics", {}).items():
            metrics[k] = max(metrics.get(k, 0), v or 0)

    levels = {
        "head": severity(metrics["head_forward_ratio"]),
        "shoulder": severity(max(metrics["shoulder_asymmetry"], metrics["rounded_shoulder_ratio"])),
        "pelvis": severity(metrics["pelvis_asymmetry"]),
        "knee": severity(metrics["knee_valgus_ratio"]),
    }

    score = 100
    penalty = {"normal": 0, "hafif": 6, "orta": 12, "belirgin": 18, "limited": 5}
    score -= penalty.get(levels["head"], 0)
    score -= penalty.get(levels["shoulder"], 0)
    score -= penalty.get(levels["pelvis"], 0)
    score -= penalty.get(levels["knee"], 0)
    score = max(40, score)

    risk = "Düşük" if score >= 88 else "Hafif" if score >= 75 else "Orta" if score >= 60 else "Belirgin"

    findings = []
    fixes = []
    stretches = []
    strengthen = []
    avoid = []
    chain = []

    if levels["head"] != "normal":
        findings.append(f"Baş-boyun hizası: {levels['head']}")
        fixes += ["Chin Tuck 2x10", "Wall Angel 2x8"]
        chain.append("Başın öne taşınması boyun ekstansör yükünü artırabilir ve omuz protraksyonunu tetikleyebilir.")

    if levels["shoulder"] != "normal":
        findings.append(f"Omuz kuşağı / protraksyon: {levels['shoulder']}")
        fixes += ["Face Pull 3x12", "Band Pull Apart 2x15", "Thoracic Extension 2x8"]
        stretches += ["Pectoralis minor", "Anterior deltoid"]
        strengthen += ["Lower trapezius", "Rhomboid", "External rotator"]

    if levels["pelvis"] != "normal":
        findings.append(f"Pelvis hizası / lordoz eğilimi: {levels['pelvis']}")
        fixes += ["Dead Bug 2x10", "Glute Bridge 2x12", "Hip Flexor Stretch 2x30 sn"]
        stretches += ["Iliopsoas", "Rectus femoris"]
        strengthen += ["Gluteus maximus", "Core brace"]

    if levels["knee"] != "normal":
        findings.append(f"Diz/ayak dizilimi: {levels['knee']}")
        fixes += ["Mini Band Lateral Walk 2x12", "Short Foot Drill 2x10", "Single Leg Balance 2x20 sn"]
        strengthen += ["Gluteus medius", "Ayak intrensek kasları"]

    if not findings:
        findings.append("Okunabilen landmarklara göre belirgin postür problemi saptanmadı.")
        fixes += ["Face Pull 2x12", "Dead Bug 2x10", "Hip Hinge Drill 2x8"]

    def unique(arr):
        out = []
        for x in arr:
            if x and x not in out:
                out.append(x)
        return out

    return {
        "score": score,
        "risk": risk,
        "analysis_source": "Backend MediaPipe + Landmark + Vektör",
        "landmark_quality": quality,
        "angles": angles,
        "metrics": metrics,
        "levels": levels,
        "findings": unique(findings),
        "fixes": unique(fixes),
        "stretches": unique(stretches),
        "strengthen": unique(strengthen),
        "avoid": unique(avoid),
        "chain": unique(chain),
        "raw": {"front": front, "side": side, "back": back},
    }


@app.get("/")
def health():
    return {"ok": True, "service": "MergeSport Posture Analysis API"}


@app.post("/analyze")
async def analyze(
    front: UploadFile = File(None),
    side: UploadFile = File(None),
    back: UploadFile = File(None),
):
    if not front and not side and not back:
        raise HTTPException(status_code=400, detail="En az bir fotoğraf gönderilmelidir.")

    front_result = None
    side_result = None
    back_result = None

    if front:
        front_result = analyze_image(read_image(await front.read()), "front")
    if side:
        side_result = analyze_image(read_image(await side.read()), "side")
    if back:
        back_result = analyze_image(read_image(await back.read()), "back")

    return build_report(front_result, side_result, back_result)
