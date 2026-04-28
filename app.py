from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import math
import base64
from typing import Dict, Any, Optional

app = FastAPI(title="MergeSport Posture Analysis API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose

COLORS = {
    "blue": (255, 120, 40),
    "green": (80, 220, 80),
    "yellow": (0, 200, 255),
    "orange": (0, 140, 255),
    "red": (60, 60, 255),
    "white": (255, 255, 255),
    "dark": (20, 25, 35),
    "cyan": (255, 220, 60),
    "purple": (220, 80, 220),
}

def pxy(p, w, h):
    return int(p["x"] * w), int(p["y"] * h)

def point(lm, idx: int) -> Dict[str, float]:
    p = lm[idx]
    return {"x": float(p.x), "y": float(p.y), "z": float(p.z), "visibility": float(p.visibility)}

def safe_point(lm, idx: int, min_visibility: float = 0.20) -> Optional[Dict[str, float]]:
    p = point(lm, idx)
    if p["visibility"] < min_visibility:
        return None
    return p

def dist(a, b) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

def line_tilt_deg(a, b) -> Optional[float]:
    if not a or not b:
        return None
    return round(abs(math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))), 1)

def vertical_deviation_deg(a, b) -> Optional[float]:
    if not a or not b:
        return None
    angle = abs(math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"])))
    return round(abs(90 - angle), 1)

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

def level_label(level: str) -> str:
    return {"normal": "Normal", "hafif": "Hafif", "orta": "Orta", "belirgin": "Belirgin", "limited": "Sınırlı"}.get(level, level)

def angle_color(value, mode="tilt"):
    if value is None:
        return COLORS["yellow"]
    v = abs(float(value))
    if mode == "knee":
        d = abs(v - 180)
        if d <= 8: return COLORS["green"]
        if d <= 15: return COLORS["yellow"]
        return COLORS["red"]
    if mode == "head":
        if v <= 8: return COLORS["green"]
        if v <= 15: return COLORS["yellow"]
        return COLORS["red"]
    if v <= 3: return COLORS["green"]
    if v <= 7: return COLORS["yellow"]
    return COLORS["red"]

def fmt_deg(v):
    if v is None:
        return "Olcum yok"
    return f"{float(v):.1f} deg"

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

def draw_point(img, p, color=COLORS["blue"]):
    if not p: return
    h, w = img.shape[:2]
    x, y = pxy(p, w, h)
    cv2.circle(img, (x, y), 10, COLORS["white"], -1)
    cv2.circle(img, (x, y), 6, color, -1)
    cv2.circle(img, (x, y), 10, COLORS["dark"], 2)

def draw_line(img, a, b, color=COLORS["green"], thickness=4):
    if not a or not b: return
    h, w = img.shape[:2]
    cv2.line(img, pxy(a, w, h), pxy(b, w, h), color, thickness, cv2.LINE_AA)

def draw_label(img, text, p, color=COLORS["green"], dx=10, dy=-12):
    if not p: return
    h, w = img.shape[:2]
    x, y = pxy(p, w, h)
    x = max(8, min(x + dx, w - 310))
    y = max(34, min(y + dy, h - 12))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x - 8, y - th - 12), (x + tw + 12, y + 10), COLORS["dark"], -1)
    cv2.rectangle(img, (x - 8, y - th - 12), (x + tw + 12, y + 10), color, 2)
    cv2.putText(img, text, (x, y), font, scale, COLORS["white"], thickness, cv2.LINE_AA)

def b64_image(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok: return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

def overlay_image(img_bgr, lm, view: str, angles: Dict[str, Any], ok: bool):
    img = resize_for_pose(img_bgr.copy(), max_side=1100)
    h, w = img.shape[:2]
    cv2.rectangle(img, (12, 12), (min(570, w - 12), 92), COLORS["dark"], -1)
    title = {"front": "On Vektorel Analiz", "side": "Yan Vektorel Analiz", "back": "Arka Vektorel Analiz"}.get(view, "Vektorel Analiz")
    cv2.putText(img, title, (26, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLORS["white"], 2, cv2.LINE_AA)

    if not ok or not lm:
        cv2.putText(img, "Landmark alinamadi: tam kadraj/net isik gerekli.", (26, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["yellow"], 2, cv2.LINE_AA)
        cv2.line(img, (w // 2, 105), (w // 2, h - 24), COLORS["cyan"], 3, cv2.LINE_AA)
        return b64_image(img)

    cv2.putText(img, "Gercek landmark + aci olcumu", (26, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 2, cv2.LINE_AA)

    nose=safe_point(lm,0); ls=safe_point(lm,11); rs=safe_point(lm,12); le=safe_point(lm,13); re=safe_point(lm,14)
    lw=safe_point(lm,15); rw=safe_point(lm,16); lh=safe_point(lm,23); rh=safe_point(lm,24); lk=safe_point(lm,25); rk=safe_point(lm,26); la=safe_point(lm,27); ra=safe_point(lm,28)

    sm = mid(ls, rs)
    if sm:
        x, _ = pxy(sm, w, h)
        cv2.line(img, (x, 105), (x, h - 24), COLORS["cyan"], 2, cv2.LINE_AA)

    for a,b,c,t in [
        (ls,rs,COLORS["red"],5),(lh,rh,COLORS["orange"],5),(ls,lh,COLORS["cyan"],3),(rs,rh,COLORS["cyan"],3),
        (lh,lk,COLORS["purple"],4),(lk,la,COLORS["green"],4),(rh,rk,COLORS["purple"],4),(rk,ra,COLORS["green"],4),
        (ls,le,COLORS["white"],2),(le,lw,COLORS["white"],2),(rs,re,COLORS["white"],2),(re,rw,COLORS["white"],2)
    ]:
        draw_line(img,a,b,c,t)

    for p in [nose,ls,rs,le,re,lw,rw,lh,rh,lk,rk,la,ra]:
        draw_point(img,p,COLORS["blue"])

    if view in ["front","back"]:
        draw_label(img, "Omuz: " + fmt_deg(angles.get("shoulder_tilt")), ls, angle_color(angles.get("shoulder_tilt"), "tilt"))
        draw_label(img, "Pelvis: " + fmt_deg(angles.get("pelvis_tilt")), lh, angle_color(angles.get("pelvis_tilt"), "tilt"))
        draw_label(img, "Sol diz: " + fmt_deg(angles.get("left_knee")), lk, angle_color(angles.get("left_knee"), "knee"))
        draw_label(img, "Sag diz: " + fmt_deg(angles.get("right_knee")), rk, angle_color(angles.get("right_knee"), "knee"))

    if view == "side":
        shoulder_mid = mid(ls, rs); hip_mid = mid(lh, rh)
        draw_line(img, nose, shoulder_mid, COLORS["cyan"], 5)
        draw_line(img, shoulder_mid, hip_mid, COLORS["green"], 5)
        draw_label(img, "Bas: " + fmt_deg(angles.get("head_forward")), shoulder_mid, angle_color(angles.get("head_forward"), "head"))
        draw_label(img, "Gövde: " + fmt_deg(angles.get("trunk_lean")), hip_mid, angle_color(angles.get("trunk_lean"), "tilt"))
        draw_label(img, "Diz: " + fmt_deg(angles.get("left_knee") or angles.get("right_knee")), lk or rk, angle_color(angles.get("left_knee") or angles.get("right_knee"), "knee"))
    return b64_image(img)

def analyze_image(img_bgr, view: str) -> Dict[str, Any]:
    img_pose = resize_for_pose(img_bgr)
    img_rgb = cv2.cvtColor(img_pose, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.30) as pose:
        result = pose.process(img_rgb)
    if not result.pose_landmarks:
        return {"view": view, "ok": False, "landmark_quality": 0, "reason": "Landmark bulunamadı.", "landmarks": [], "angles": {}, "metrics": {}, "overlay": overlay_image(img_bgr, None, view, {}, False)}
    lm = result.pose_landmarks.landmark
    landmarks = [point(lm, i) for i in range(len(lm))]
    quality = round((sum(1 for p in landmarks if p["visibility"] >= 0.30) / len(landmarks)) * 100)

    nose=safe_point(lm,0); ls=safe_point(lm,11); rs=safe_point(lm,12); le=safe_point(lm,13); re=safe_point(lm,14)
    lw=safe_point(lm,15); rw=safe_point(lm,16); lh=safe_point(lm,23); rh=safe_point(lm,24); lk=safe_point(lm,25); rk=safe_point(lm,26); la=safe_point(lm,27); ra=safe_point(lm,28)
    shoulder_mid = mid(ls, rs); hip_mid = mid(lh, rh)
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
    metrics = {"shoulder_asymmetry":0,"pelvis_asymmetry":0,"head_forward_ratio":0,"rounded_shoulder_ratio":0,"knee_valgus_ratio":0}
    if ls and rs:
        ref=max(0.0001,dist(ls,rs)); metrics["shoulder_asymmetry"]=abs(ls["y"]-rs["y"])/ref
    if lh and rh:
        ref=max(0.0001,dist(lh,rh)); metrics["pelvis_asymmetry"]=abs(lh["y"]-rh["y"])/ref
    if nose and shoulder_mid and ls and rs:
        ref=max(0.0001,dist(ls,rs)); metrics["head_forward_ratio"]=abs(nose["x"]-shoulder_mid["x"])/ref
    if shoulder_mid and hip_mid and ls and rs:
        ref=max(0.0001,dist(ls,rs)); metrics["rounded_shoulder_ratio"]=abs(shoulder_mid["x"]-hip_mid["x"])/ref
    if lk and rk and la and ra:
        kg=abs(lk["x"]-rk["x"]); ag=abs(la["x"]-ra["x"])
        if ag>0: metrics["knee_valgus_ratio"]=max(0,(ag-kg)/ag)
    return {"view": view, "ok": True, "landmark_quality": quality, "landmarks": landmarks, "angles": angles, "metrics": metrics, "overlay": overlay_image(img_bgr, lm, view, angles, True)}

def build_report(front=None, side=None, back=None) -> Dict[str, Any]:
    analyses = [x for x in [front, side, back] if x]
    ok_analyses = [x for x in analyses if x.get("ok")]
    overlays = {"front": front.get("overlay") if front else "", "side": side.get("overlay") if side else "", "back": back.get("overlay") if back else ""}
    if not ok_analyses:
        return {"score":70,"risk":"Sınırlı","analysis_source":"Backend MediaPipe - Landmark alınamadı","landmark_quality":0,"angles":{},"levels":{"head":"limited","shoulder":"limited","pelvis":"limited","knee":"limited"},"findings":["Fotoğraflardan landmark alınamadı."],"fixes":["Wall Posture Hold 2x30 sn","Dead Bug 2x10","Face Pull 2x12"],"notes":["Tam analiz için baştan ayağa tam kadraj, iyi ışık ve sade arka plan gerekir."],"overlays":overlays,"program_bridge":{},"raw":{"front":front,"side":side,"back":back}}
    quality = round(sum(x["landmark_quality"] for x in ok_analyses) / len(ok_analyses))
    angles = {}; metrics = {"shoulder_asymmetry":0,"pelvis_asymmetry":0,"head_forward_ratio":0,"rounded_shoulder_ratio":0,"knee_valgus_ratio":0}
    for item in ok_analyses:
        for k,v in item.get("angles",{}).items():
            if v is not None and k not in angles: angles[k]=v
        for k,v in item.get("metrics",{}).items():
            metrics[k]=max(metrics.get(k,0),v or 0)
    levels = {"head":severity(metrics["head_forward_ratio"]),"shoulder":severity(max(metrics["shoulder_asymmetry"],metrics["rounded_shoulder_ratio"])),"pelvis":severity(metrics["pelvis_asymmetry"]),"knee":severity(metrics["knee_valgus_ratio"])}
    score=100; penalty={"normal":0,"hafif":6,"orta":12,"belirgin":18,"limited":5}
    for v in levels.values(): score -= penalty.get(v,0)
    score=max(40,score); risk="Düşük" if score>=88 else "Hafif" if score>=75 else "Orta" if score>=60 else "Belirgin"
    findings=[]; fixes=[]; stretches=[]; strengthen=[]; avoid=[]; chain=[]
    if levels["head"]!="normal":
        findings.append(f"Baş-boyun hizası: {level_label(levels['head'])}"); fixes += ["Chin Tuck 2x10","Wall Angel 2x8"]; chain.append("Başın öne taşınması boyun ekstansör yükünü artırabilir.")
    if levels["shoulder"]!="normal":
        findings.append(f"Omuz kuşağı / protraksyon: {level_label(levels['shoulder'])}"); fixes += ["Face Pull 3x12","Band Pull Apart 2x15","Thoracic Extension 2x8"]; stretches += ["Pectoralis minor","Anterior deltoid"]; strengthen += ["Lower trapezius","Rhomboid","External rotator"]; avoid += ["Kontrolsüz dips","Aşırı öne kapanarak yapılan press varyasyonları"]
    if levels["pelvis"]!="normal":
        findings.append(f"Pelvis hizası / lordoz eğilimi: {level_label(levels['pelvis'])}"); fixes += ["Dead Bug 2x10","Glute Bridge 2x12","Hip Flexor Stretch 2x30 sn"]; stretches += ["Iliopsoas","Rectus femoris"]; strengthen += ["Gluteus maximus","Core brace"]; avoid += ["Kontrolsüz lumbar extension","Aşırı bel boşluğu ile overhead press"]
    if levels["knee"]!="normal":
        findings.append(f"Diz/ayak dizilimi: {level_label(levels['knee'])}"); fixes += ["Mini Band Lateral Walk 2x12","Short Foot Drill 2x10","Single Leg Balance 2x20 sn"]; strengthen += ["Gluteus medius","Ayak intrensek kasları"]; avoid += ["Diz içe kaçarak yapılan squat/lunge"]
    if not findings:
        findings.append("Okunabilen landmarklara göre belirgin postür problemi saptanmadı."); fixes += ["Face Pull 2x12","Dead Bug 2x10","Hip Hinge Drill 2x8"]
    def unique(arr):
        out=[]
        for x in arr:
            if x and x not in out: out.append(x)
        return out
    program_bridge={"posture_score":score,"posture_risk":risk,"corrective_exercises":unique(fixes),"stretching_focus":unique(stretches),"strengthening_focus":unique(strengthen),"avoid_exercises":unique(avoid),"program_notes":unique(chain),"priority":"corrective_block_first" if score<88 else "maintenance_block"}
    return {"score":score,"risk":risk,"analysis_source":"Backend MediaPipe + Landmark + Vektör","landmark_quality":quality,"angles":angles,"metrics":metrics,"levels":levels,"findings":unique(findings),"fixes":unique(fixes),"stretches":unique(stretches),"strengthen":unique(strengthen),"avoid":unique(avoid),"chain":unique(chain),"overlays":overlays,"program_bridge":program_bridge,"raw":{"front":front,"side":side,"back":back}}

@app.get("/")
def health():
    return {"ok": True, "service": "MergeSport Posture Analysis API", "version": "2.0.0"}

@app.post("/analyze")
async def analyze(front: UploadFile = File(None), side: UploadFile = File(None), back: UploadFile = File(None)):
    if not front and not side and not back:
        raise HTTPException(status_code=400, detail="En az bir fotoğraf gönderilmelidir.")
    front_result = analyze_image(read_image(await front.read()), "front") if front else None
    side_result = analyze_image(read_image(await side.read()), "side") if side else None
    back_result = analyze_image(read_image(await back.read()), "back") if back else None
    return build_report(front_result, side_result, back_result)
