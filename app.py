from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import math
import base64
import os
from PIL import Image, ImageDraw, ImageFont

app = FastAPI(title="MergeSport Clinical Posture API", version="5.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose


# ======================================================
# FONT / TÜRKÇE KARAKTER DESTEĞİ
# ======================================================

def get_font(size=24, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_text_box_cv(img, text, xy, border=(34, 197, 94), fill=(15, 23, 42), size=22):
    pil = cv_to_pil(img)
    draw = ImageDraw.Draw(pil)
    font = get_font(size, True)

    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 8
    rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)

    draw.rounded_rectangle(rect, radius=8, fill=fill, outline=border, width=3)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return pil_to_cv(pil)


def draw_title_cv(img, title, subtitle):
    pil = cv_to_pil(img)
    draw = ImageDraw.Draw(pil)

    font_b = get_font(25, True)
    font = get_font(15, False)

    draw.rounded_rectangle((14, 14, 650, 96), radius=12, fill=(15, 23, 42))
    draw.text((28, 28), title, fill=(255, 255, 255), font=font_b)
    draw.text((28, 64), subtitle, fill=(203, 213, 225), font=font)

    return pil_to_cv(pil)


# ======================================================
# TEMEL YARDIMCI FONKSİYONLAR
# ======================================================

def read_image(b):
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Görsel okunamadı.")
    return img


def resize(img, max_side=1280):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def pt(lm, i):
    p = lm[i]
    return {
        "x": float(p.x),
        "y": float(p.y),
        "z": float(p.z),
        "visibility": float(p.visibility),
    }


def safe(lm, i, v=0.20):
    p = pt(lm, i)
    return p if p["visibility"] >= v else None


def pxy(p, w, h):
    return int(p["x"] * w), int(p["y"] * h)


def dist(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def mid(a, b):
    if not a or not b:
        return None
    return {
        "x": (a["x"] + b["x"]) / 2,
        "y": (a["y"] + b["y"]) / 2,
        "z": 0,
        "visibility": min(a["visibility"], b["visibility"]),
    }


def tilt(a, b):
    """
    Yatay hatta göre eğim.
    Omuz / pelvis asimetrisinde 0° normal kabul edilir.
    """
    if not a or not b:
        return None

    raw = abs(math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"])))
    return round(min(raw, abs(180 - raw)), 1)


def vdev(a, b):
    """
    Dikey hatta göre sapma.
    Baş-boyun hizasında artık BURUN DEĞİL KULAK referansı kullanılır.
    """
    if not a or not b:
        return None

    angle = abs(math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"])))
    return round(abs(90 - angle), 1)


def joint_angle(a, b, c):
    if not a or not b or not c:
        return None

    ab = (a["x"] - b["x"], a["y"] - b["y"])
    cb = (c["x"] - b["x"], c["y"] - b["y"])

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    m1 = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    m2 = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if not m1 or not m2:
        return None

    return round(math.degrees(math.acos(max(-1, min(1, dot / (m1 * m2))))), 1)


def level_ratio(x):
    if x >= 0.18:
        return "belirgin"
    if x >= 0.12:
        return "orta"
    if x >= 0.06:
        return "hafif"
    return "normal"


def label_tr(x):
    return {
        "normal": "Normal",
        "hafif": "Hafif",
        "orta": "Orta",
        "belirgin": "Belirgin",
        "limited": "Sınırlı",
    }.get(x, x)


def unique(items):
    out = []
    for x in items:
        if x and x not in out:
            out.append(x)
    return out


def color(v, kind="tilt"):
    if v is None:
        return (245, 158, 11)

    v = abs(float(v))

    if kind == "knee":
        d = abs(v - 180)
        if d <= 8:
            return (34, 197, 94)
        if d <= 15:
            return (245, 158, 11)
        return (239, 68, 68)

    if kind == "head":
        if v <= 8:
            return (34, 197, 94)
        if v <= 15:
            return (245, 158, 11)
        return (239, 68, 68)

    if v <= 3:
        return (34, 197, 94)
    if v <= 7:
        return (245, 158, 11)
    return (239, 68, 68)


def deg(v):
    if v is None:
        return "Analiz edilemedi"
    return f"{float(v):.1f}°"


def line(img, a, b, col, t=4):
    if a and b:
        h, w = img.shape[:2]
        cv2.line(img, pxy(a, w, h), pxy(b, w, h), col, t, cv2.LINE_AA)


def pointdraw(img, p, col=(37, 99, 235)):
    if p:
        h, w = img.shape[:2]
        x, y = pxy(p, w, h)
        cv2.circle(img, (x, y), 10, (255, 255, 255), -1)
        cv2.circle(img, (x, y), 6, col, -1)
        cv2.circle(img, (x, y), 10, (15, 23, 42), 2)


def img64(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


# ======================================================
# OVERLAY ÇİZİMİ
# ======================================================

def make_overlay(img_bgr, lm, view, angles, ok):
    img = resize(img_bgr.copy(), 1200)
    h, w = img.shape[:2]

    titles = {
        "front": "Ön Vektörel Analiz",
        "side": "Yan Vektörel Analiz",
        "back": "Arka Vektörel Analiz",
    }

    img = draw_title_cv(
        img,
        titles.get(view, "Vektörel Analiz"),
        "Landmark noktaları, aks çizgileri ve açı ölçümleri",
    )

    if not ok or not lm:
        img = draw_text_box_cv(
            img,
            "Landmark alınamadı: tam kadraj / net ışık gerekli",
            (28, 108),
            (245, 158, 11),
            (15, 23, 42),
            20,
        )
        cv2.line(img, (w // 2, 135), (w // 2, h - 24), (56, 189, 248), 3, cv2.LINE_AA)
        return img64(img)

    P = {
        "nose": safe(lm, 0),
        "leye": safe(lm, 2),
        "reye": safe(lm, 5),
        "lear": safe(lm, 7),
        "rear": safe(lm, 8),
        "ls": safe(lm, 11),
        "rs": safe(lm, 12),
        "le": safe(lm, 13),
        "re": safe(lm, 14),
        "lw": safe(lm, 15),
        "rw": safe(lm, 16),
        "lh": safe(lm, 23),
        "rh": safe(lm, 24),
        "lk": safe(lm, 25),
        "rk": safe(lm, 26),
        "la": safe(lm, 27),
        "ra": safe(lm, 28),
    }

    shoulder_mid = mid(P["ls"], P["rs"])
    hip_mid = mid(P["lh"], P["rh"])

    if shoulder_mid:
        x, _ = pxy(shoulder_mid, w, h)
        cv2.line(img, (x, 110), (x, h - 24), (56, 189, 248), 2, cv2.LINE_AA)

    pairs = [
        ("ls", "rs", (239, 68, 68), 5),
        ("lh", "rh", (245, 158, 11), 5),
        ("ls", "lh", (6, 182, 212), 3),
        ("rs", "rh", (6, 182, 212), 3),
        ("lh", "lk", (168, 85, 247), 4),
        ("lk", "la", (34, 197, 94), 4),
        ("rh", "rk", (168, 85, 247), 4),
        ("rk", "ra", (34, 197, 94), 4),
        ("ls", "le", (148, 163, 184), 2),
        ("le", "lw", (148, 163, 184), 2),
        ("rs", "re", (148, 163, 184), 2),
        ("re", "rw", (148, 163, 184), 2),
    ]

    for a, b, col, t in pairs:
        line(img, P[a], P[b], col, t)

    for p in P.values():
        pointdraw(img, p)

    if view in ["front", "back"]:
        img = draw_text_box_cv(
            img,
            "Omuz: " + deg(angles.get("shoulder_tilt")),
            pxy(P["ls"], w, h) if P["ls"] else (30, 120),
            color(angles.get("shoulder_tilt")),
            (15, 23, 42),
            21,
        )
        img = draw_text_box_cv(
            img,
            "Pelvis: " + deg(angles.get("pelvis_tilt")),
            pxy(P["lh"], w, h) if P["lh"] else (30, 165),
            color(angles.get("pelvis_tilt")),
            (15, 23, 42),
            21,
        )
        img = draw_text_box_cv(
            img,
            "Sol diz: " + deg(angles.get("left_knee")),
            pxy(P["lk"], w, h) if P["lk"] else (30, 210),
            color(angles.get("left_knee"), "knee"),
            (15, 23, 42),
            20,
        )
        img = draw_text_box_cv(
            img,
            "Sağ diz: " + deg(angles.get("right_knee")),
            pxy(P["rk"], w, h) if P["rk"] else (30, 255),
            color(angles.get("right_knee"), "knee"),
            (15, 23, 42),
            20,
        )

    if view == "side":
        # Baş-boyun için KULAK referansı kullanılır.
        ear_ref = P["lear"] or P["rear"]
        sh = shoulder_mid
        hip = hip_mid

        line(img, ear_ref, sh, (6, 182, 212), 5)
        line(img, sh, hip, (34, 197, 94), 5)

        img = draw_text_box_cv(
            img,
            "Baş-boyun: " + deg(angles.get("head_forward")),
            pxy(sh, w, h) if sh else (30, 120),
            color(angles.get("head_forward"), "head"),
            (15, 23, 42),
            21,
        )
        img = draw_text_box_cv(
            img,
            "Gövde: " + deg(angles.get("trunk_lean")),
            pxy(hip, w, h) if hip else (30, 165),
            color(angles.get("trunk_lean")),
            (15, 23, 42),
            21,
        )
        knee = P["lk"] or P["rk"]
        img = draw_text_box_cv(
            img,
            "Diz: " + deg(angles.get("left_knee") or angles.get("right_knee")),
            pxy(knee, w, h) if knee else (30, 210),
            color(angles.get("left_knee") or angles.get("right_knee"), "knee"),
            (15, 23, 42),
            20,
        )

    return img64(img)


# ======================================================
# TEK GÖRSEL ANALİZ
# ======================================================

def analyze_img(img_bgr, view):
    rgb = cv2.cvtColor(resize(img_bgr), cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.30,
    ) as pose:
        res = pose.process(rgb)

    if not res.pose_landmarks:
        return {
            "view": view,
            "ok": False,
            "landmark_quality": 0,
            "angles": {},
            "metrics": {},
            "overlay": make_overlay(img_bgr, None, view, {}, False),
            "landmarks": [],
        }

    lm = res.pose_landmarks.landmark
    landmarks = [pt(lm, i) for i in range(len(lm))]
    q = round(sum(1 for p in landmarks if p["visibility"] >= 0.30) / len(landmarks) * 100)

    nose = safe(lm, 0)
    lear = safe(lm, 7)
    rear = safe(lm, 8)
    ear_ref = lear or rear

    ls = safe(lm, 11)
    rs = safe(lm, 12)
    le = safe(lm, 13)
    re = safe(lm, 14)
    lw = safe(lm, 15)
    rw = safe(lm, 16)

    lh = safe(lm, 23)
    rh = safe(lm, 24)
    lk = safe(lm, 25)
    rk = safe(lm, 26)
    la = safe(lm, 27)
    ra = safe(lm, 28)

    shoulder_mid = mid(ls, rs)
    hip_mid = mid(lh, rh)

    # BURADA DÜZELTME YAPILDI:
    # Baş-boyun hizası artık burun değil kulak-omuz referansıyla ölçülür.
    A = {
        "shoulder_tilt": tilt(ls, rs) if ls and rs else None,
        "pelvis_tilt": tilt(lh, rh) if lh and rh else None,
        "head_forward": vdev(ear_ref, shoulder_mid) if ear_ref and shoulder_mid else None,
        "trunk_lean": vdev(shoulder_mid, hip_mid) if shoulder_mid and hip_mid else None,
        "left_knee": joint_angle(lh, lk, la) if lh and lk and la else None,
        "right_knee": joint_angle(rh, rk, ra) if rh and rk and ra else None,
        "left_elbow": joint_angle(ls, le, lw) if ls and le and lw else None,
        "right_elbow": joint_angle(rs, re, rw) if rs and re and rw else None,
    }

    M = {
        "shoulder_asymmetry": 0,
        "pelvis_asymmetry": 0,
        "head_forward_ratio": 0,
        "rounded_shoulder_ratio": 0,
        "knee_valgus_ratio": 0,
    }

    if ls and rs:
        M["shoulder_asymmetry"] = abs(ls["y"] - rs["y"]) / max(0.0001, dist(ls, rs))

    if lh and rh:
        M["pelvis_asymmetry"] = abs(lh["y"] - rh["y"]) / max(0.0001, dist(lh, rh))

    # BURADA DA KULAK REFERANSI KULLANILIR.
    if ear_ref and shoulder_mid and ls and rs:
        M["head_forward_ratio"] = abs(ear_ref["x"] - shoulder_mid["x"]) / max(0.0001, dist(ls, rs))

    if shoulder_mid and hip_mid and ls and rs:
        M["rounded_shoulder_ratio"] = abs(shoulder_mid["x"] - hip_mid["x"]) / max(0.0001, dist(ls, rs))

    if lk and rk and la and ra:
        knee_gap = abs(lk["x"] - rk["x"])
        ankle_gap = abs(la["x"] - ra["x"])
        if ankle_gap > 0:
            M["knee_valgus_ratio"] = max(0, (ankle_gap - knee_gap) / ankle_gap)

    return {
        "view": view,
        "ok": True,
        "landmark_quality": q,
        "angles": A,
        "metrics": M,
        "overlay": make_overlay(img_bgr, lm, view, A, True),
        "landmarks": landmarks,
    }


# ======================================================
# RAPOR OLUŞTURMA
# ======================================================

def build_bridge(score, risk, levels, fixes, stretches, strengthen, avoid, chain):
    return {
        "priority": "corrective_block_first" if score < 88 else "maintenance_block",
        "posture_score": score,
        "posture_risk": risk,
        "levels": levels,
        "corrective_block": unique(
            ["90/90 Breathing 2x5 nefes", "Cat-Camel 1x8", "Wall Slide 2x10"] + fixes[:5]
        ),
        "corrective_exercises": unique(fixes),
        "stretching_focus": unique(stretches),
        "strengthening_focus": unique(strengthen),
        "avoid_exercises": unique(avoid),
        "program_notes": unique(chain),
        "program_rule": "Ana antrenmandan önce 8-12 dakikalık düzeltici blok ekle; kaçınılacak hareketleri güvenli varyasyonla değiştir.",
    }


def report_html(data):
    a = data.get("angles", {})

    def li(arr):
        return "".join(f"<li>{x}</li>" for x in (arr or []))

    return f"""
    <div>
        <h1>MergeSport Klinik Postür Raporu</h1>
        <h2>Skor: {data.get('score')}/100 | Risk: {data.get('risk')} | Landmark: {data.get('landmark_quality')}%</h2>
        <h3>Açılar</h3>
        <table>
            <tr><td>Omuz</td><td>{deg(a.get('shoulder_tilt'))}</td></tr>
            <tr><td>Pelvis</td><td>{deg(a.get('pelvis_tilt'))}</td></tr>
            <tr><td>Baş-boyun</td><td>{deg(a.get('head_forward'))}</td></tr>
            <tr><td>Gövde</td><td>{deg(a.get('trunk_lean'))}</td></tr>
            <tr><td>Sol Diz</td><td>{deg(a.get('left_knee'))}</td></tr>
            <tr><td>Sağ Diz</td><td>{deg(a.get('right_knee'))}</td></tr>
        </table>
        <h3>Problemler</h3>
        <ul>{li(data.get('findings'))}</ul>
        <h3>Düzeltici Egzersizler</h3>
        <ul>{li(data.get('fixes'))}</ul>
        <h3>Chain Analysis</h3>
        <ul>{li(data.get('chain'))}</ul>
    </div>
    """


def build_report(front=None, side=None, back=None):
    analyses = [x for x in [front, side, back] if x]
    ok = [x for x in analyses if x.get("ok")]

    overlays = {
        "front": front.get("overlay") if front else "",
        "side": side.get("overlay") if side else "",
        "back": back.get("overlay") if back else "",
    }

    if not ok:
        data = {
            "score": 70,
            "score_label": "70/100",
            "risk": "Sınırlı",
            "analysis_source": "Backend MediaPipe - Landmark alınamadı",
            "landmark_quality": 0,
            "angles": {},
            "metrics": {},
            "levels": {
                "head": "limited",
                "shoulder": "limited",
                "pelvis": "limited",
                "knee": "limited",
            },
            "findings": ["Fotoğraflardan landmark alınamadı."],
            "fixes": ["Wall Posture Hold 2x30 sn", "Dead Bug 2x10", "Face Pull 2x12"],
            "stretches": [],
            "strengthen": [],
            "avoid": [],
            "chain": ["Tam analiz için baştan ayağa tam kadraj, iyi ışık ve sade arka plan gerekir."],
            "overlays": overlays,
        }
    else:
        q = round(sum(x["landmark_quality"] for x in ok) / len(ok))

        A = {}
        M = {
            "shoulder_asymmetry": 0,
            "pelvis_asymmetry": 0,
            "head_forward_ratio": 0,
            "rounded_shoulder_ratio": 0,
            "knee_valgus_ratio": 0,
        }

        for item in ok:
            for k, v in item.get("angles", {}).items():
                if v is not None and k not in A:
                    A[k] = v

            for k, v in item.get("metrics", {}).items():
                M[k] = max(M.get(k, 0), v or 0)

        L = {
            "head": level_ratio(M["head_forward_ratio"]),
            "shoulder": level_ratio(max(M["shoulder_asymmetry"], M["rounded_shoulder_ratio"])),
            "pelvis": level_ratio(M["pelvis_asymmetry"]),
            "knee": level_ratio(M["knee_valgus_ratio"]),
        }

        score = 100
        penalty = {
            "normal": 0,
            "hafif": 6,
            "orta": 12,
            "belirgin": 18,
            "limited": 5,
        }

        for v in L.values():
            score -= penalty.get(v, 0)

        score = max(40, score)

        if score >= 88:
            risk = "Düşük"
        elif score >= 75:
            risk = "Hafif"
        elif score >= 60:
            risk = "Orta"
        else:
            risk = "Belirgin"

        findings = []
        fixes = []
        stretches = []
        strengthen = []
        avoid = []
        chain = []

        if L["head"] != "normal":
            findings.append(f"Baş-boyun hizasında {label_tr(L['head']).lower()} öne taşınma eğilimi.")
            fixes += ["Chin Tuck 2x10", "Wall Angel 2x8", "Thoracic Extension 2x8"]
            chain.append("Kulak-omuz hattı öne kaydığında servikal yük artabilir; torakal mobilite ve boyun hizalama egzersizleri önceliklidir.")

        if L["shoulder"] != "normal":
            findings.append(f"Omuz kuşağında {label_tr(L['shoulder']).lower()} protraksyon/asimetri eğilimi.")
            fixes += ["Face Pull 3x12", "Band Pull Apart 2x15", "Thoracic Extension 2x8"]
            stretches += ["Pectoralis minor", "Anterior deltoid"]
            strengthen += ["Lower trapezius", "Rhomboid", "External rotator"]
            avoid += ["Kontrolsüz dips", "Aşırı öne kapanarak yapılan press varyasyonları"]
            chain.append("Yuvarlak omuz eğilimi press hareketlerinde kompansasyon ve skapular kontrol kaybı oluşturabilir.")

        if L["pelvis"] != "normal":
            findings.append(f"Pelvis hizasında {label_tr(L['pelvis']).lower()} sapma/lordoz eğilimi.")
            fixes += ["Dead Bug 2x10", "Glute Bridge 2x12", "Hip Flexor Stretch 2x30 sn"]
            stretches += ["Iliopsoas", "Rectus femoris"]
            strengthen += ["Gluteus maximus", "Core brace"]
            avoid += ["Kontrolsüz lumbar extension", "Aşırı bel boşluğu ile overhead press"]
            chain.append("Pelvis kontrol kaybı bel boşluğunu artırabilir; core anti-extension ve glute aktivasyon önceliklidir.")

        if L["knee"] != "normal":
            findings.append(f"Diz/ayak diziliminde {label_tr(L['knee']).lower()} kontrol kaybı eğilimi.")
            fixes += ["Mini Band Lateral Walk 2x12", "Short Foot Drill 2x10", "Single Leg Balance 2x20 sn"]
            strengthen += ["Gluteus medius", "Ayak intrensek kasları"]
            avoid += ["Diz içe kaçarak yapılan squat/lunge"]
            chain.append("Diz valgus eğilimi alt ekstremite kuvvet aktarımını bozabilir; gluteus medius ve ayak arkı kontrolü desteklenmelidir.")

        if not findings:
            findings.append("Okunabilen landmarklara göre belirgin postür problemi saptanmadı.")
            fixes += ["Face Pull 2x12", "Dead Bug 2x10", "Hip Hinge Drill 2x8"]
            chain.append("Koruyucu amaçlı skapular stabilizasyon, core kontrol ve kalça menteşe paterni önerilir.")

        data = {
            "score": score,
            "score_label": f"{score}/100",
            "risk": risk,
            "analysis_source": "Backend MediaPipe + Landmark + Vektör",
            "landmark_quality": q,
            "angles": A,
            "metrics": M,
            "levels": L,
            "findings": unique(findings),
            "fixes": unique(fixes),
            "stretches": unique(stretches),
            "strengthen": unique(strengthen),
            "avoid": unique(avoid),
            "chain": unique(chain),
            "overlays": overlays,
        }

    data["program_bridge"] = build_bridge(
        data["score"],
        data["risk"],
        data["levels"],
        data["fixes"],
        data["stretches"],
        data["strengthen"],
        data["avoid"],
        data["chain"],
    )

    data["report_html"] = report_html(data)

    return data


# ======================================================
# ENDPOINTS
# ======================================================

@app.get("/")
def health():
    return JSONResponse(
        content={
            "ok": True,
            "service": "MergeSport Clinical Posture API",
            "version": "5.1.0",
            "head_reference": "ear_to_shoulder",
            "charset": "utf-8",
        },
        media_type="application/json; charset=utf-8",
    )


@app.post("/analyze")
async def analyze(
    front: UploadFile = File(None),
    side: UploadFile = File(None),
    back: UploadFile = File(None),
):
    if not front and not side and not back:
        raise HTTPException(status_code=400, detail="En az bir fotoğraf gönderilmelidir.")

    f = analyze_img(read_image(await front.read()), "front") if front else None
    s = analyze_img(read_image(await side.read()), "side") if side else None
    b = analyze_img(read_image(await back.read()), "back") if back else None

    data = build_report(f, s, b)

    return JSONResponse(
        content=data,
        media_type="application/json; charset=utf-8",
    )
