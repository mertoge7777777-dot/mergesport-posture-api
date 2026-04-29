from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2, mediapipe as mp, numpy as np, math, base64

app = FastAPI(title="MergeSport Posture + Program API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
mp_pose = mp.solutions.pose

def read_image(b):
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Görsel okunamadı.")
    return img

def resize(img, max_side=1280):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1 else img

def pt(lm, i):
    p = lm[i]
    return {"x": float(p.x), "y": float(p.y), "z": float(p.z), "visibility": float(p.visibility)}

def safe(lm, i, v=.20):
    p = pt(lm, i)
    return p if p["visibility"] >= v else None

def pxy(p, w, h):
    return int(p["x"] * w), int(p["y"] * h)

def dist(a,b):
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2)

def mid(a,b):
    if not a or not b: return None
    return {"x":(a["x"]+b["x"])/2, "y":(a["y"]+b["y"])/2, "z":0, "visibility":min(a["visibility"], b["visibility"])}

def tilt(a,b):
    if not a or not b: return None
    return round(abs(math.degrees(math.atan2(b["y"]-a["y"], b["x"]-a["x"]))), 1)

def vdev(a,b):
    if not a or not b: return None
    angle = abs(math.degrees(math.atan2(b["y"]-a["y"], b["x"]-a["x"])))
    return round(abs(90-angle), 1)

def angle(a,b,c):
    if not a or not b or not c: return None
    ab=(a["x"]-b["x"], a["y"]-b["y"])
    cb=(c["x"]-b["x"], c["y"]-b["y"])
    dot=ab[0]*cb[0]+ab[1]*cb[1]
    m1=math.sqrt(ab[0]**2+ab[1]**2)
    m2=math.sqrt(cb[0]**2+cb[1]**2)
    if not m1 or not m2: return None
    return round(math.degrees(math.acos(max(-1,min(1,dot/(m1*m2))))), 1)

def sev(x):
    return "belirgin" if x>=.18 else "orta" if x>=.12 else "hafif" if x>=.06 else "normal"

def lab(x):
    return {"normal":"Normal","hafif":"Hafif","orta":"Orta","belirgin":"Belirgin","limited":"Sınırlı"}.get(x,x)

def unique(a):
    o=[]
    for x in a:
        if x and x not in o: o.append(x)
    return o

def color(v, kind="tilt"):
    if v is None: return (0,200,255)
    v=abs(float(v))
    if kind=="knee":
        d=abs(v-180)
        return (80,220,80) if d<=8 else (0,200,255) if d<=15 else (60,60,255)
    if kind=="head":
        return (80,220,80) if v<=8 else (0,200,255) if v<=15 else (60,60,255)
    return (80,220,80) if v<=3 else (0,200,255) if v<=7 else (60,60,255)

def deg(v):
    return "Ölçüm yok" if v is None else f"{float(v):.1f}°"

def line(img,a,b,col,t=4):
    if a and b:
        h,w=img.shape[:2]
        cv2.line(img, pxy(a,w,h), pxy(b,w,h), col, t, cv2.LINE_AA)

def pointdraw(img,p,col=(255,120,40)):
    if p:
        h,w=img.shape[:2]
        x,y=pxy(p,w,h)
        cv2.circle(img,(x,y),10,(255,255,255),-1)
        cv2.circle(img,(x,y),6,col,-1)
        cv2.circle(img,(x,y),10,(20,25,35),2)

def tag(img,text,p,col):
    if not p: return
    h,w=img.shape[:2]
    x,y=pxy(p,w,h)
    x=max(8,min(x+10,w-310))
    y=max(35,min(y-10,h-12))
    font=cv2.FONT_HERSHEY_SIMPLEX
    scale=.62
    th=2
    (tw,hh),_=cv2.getTextSize(text,font,scale,th)
    cv2.rectangle(img,(x-8,y-hh-12),(x+tw+12,y+10),(20,25,35),-1)
    cv2.rectangle(img,(x-8,y-hh-12),(x+tw+12,y+10),col,2)
    cv2.putText(img,text,(x,y),font,scale,(255,255,255),th,cv2.LINE_AA)

def img64(img):
    ok,buf=cv2.imencode(".jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY),88])
    return "" if not ok else "data:image/jpeg;base64,"+base64.b64encode(buf.tobytes()).decode()

def make_overlay(img_bgr,lm,view,angles,ok):
    img=resize(img_bgr.copy(),1100)
    h,w=img.shape[:2]
    dark=(20,25,35)
    cyan=(255,220,60)
    title={"front":"Ön Vektörel Analiz","side":"Yan Vektörel Analiz","back":"Arka Vektörel Analiz"}.get(view,"Vektörel Analiz")
    cv2.rectangle(img,(12,12),(min(600,w-12),92),dark,-1)
    cv2.putText(img,title,(26,43),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2,cv2.LINE_AA)
    if not ok or not lm:
        cv2.putText(img,"Landmark alinamadi: tam kadraj/net isik gerekli.",(26,72),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,200,255),2,cv2.LINE_AA)
        cv2.line(img,(w//2,105),(w//2,h-24),cyan,3,cv2.LINE_AA)
        return img64(img)

    P={n:safe(lm,i) for n,i in {"nose":0,"ls":11,"rs":12,"le":13,"re":14,"lw":15,"rw":16,"lh":23,"rh":24,"lk":25,"rk":26,"la":27,"ra":28}.items()}
    sm=mid(P["ls"],P["rs"])
    if sm:
        x,_=pxy(sm,w,h)
        cv2.line(img,(x,105),(x,h-24),cyan,2,cv2.LINE_AA)

    pairs=[("ls","rs",(60,60,255),5),("lh","rh",(0,140,255),5),("ls","lh",cyan,3),("rs","rh",cyan,3),
           ("lh","lk",(220,80,220),4),("lk","la",(80,220,80),4),("rh","rk",(220,80,220),4),("rk","ra",(80,220,80),4),
           ("ls","le",(255,255,255),2),("le","lw",(255,255,255),2),("rs","re",(255,255,255),2),("re","rw",(255,255,255),2)]
    for a,b,col,t in pairs:
        line(img,P[a],P[b],col,t)
    for p in P.values():
        pointdraw(img,p)

    if view in ["front","back"]:
        tag(img,"Omuz: "+deg(angles.get("shoulder_tilt")),P["ls"],color(angles.get("shoulder_tilt")))
        tag(img,"Pelvis: "+deg(angles.get("pelvis_tilt")),P["lh"],color(angles.get("pelvis_tilt")))
        tag(img,"Sol diz: "+deg(angles.get("left_knee")),P["lk"],color(angles.get("left_knee"),"knee"))
        tag(img,"Sağ diz: "+deg(angles.get("right_knee")),P["rk"],color(angles.get("right_knee"),"knee"))
    if view=="side":
        sh=mid(P["ls"],P["rs"]); hip=mid(P["lh"],P["rh"])
        line(img,P["nose"],sh,cyan,5)
        line(img,sh,hip,(80,220,80),5)
        tag(img,"Baş: "+deg(angles.get("head_forward")),sh,color(angles.get("head_forward"),"head"))
        tag(img,"Gövde: "+deg(angles.get("trunk_lean")),hip,color(angles.get("trunk_lean")))
        tag(img,"Diz: "+deg(angles.get("left_knee") or angles.get("right_knee")),P["lk"] or P["rk"],color(angles.get("left_knee") or angles.get("right_knee"),"knee"))
    return img64(img)

def analyze_img(img_bgr,view):
    rgb=cv2.cvtColor(resize(img_bgr),cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True,model_complexity=2,enable_segmentation=False,min_detection_confidence=.30) as pose:
        res=pose.process(rgb)
    if not res.pose_landmarks:
        return {"view":view,"ok":False,"landmark_quality":0,"angles":{},"metrics":{},"overlay":make_overlay(img_bgr,None,view,{},False),"landmarks":[]}

    lm=res.pose_landmarks.landmark
    landmarks=[pt(lm,i) for i in range(len(lm))]
    q=round(sum(1 for p in landmarks if p["visibility"]>=.30)/len(landmarks)*100)
    nose=safe(lm,0); ls=safe(lm,11); rs=safe(lm,12); le=safe(lm,13); re=safe(lm,14); lw=safe(lm,15); rw=safe(lm,16)
    lh=safe(lm,23); rh=safe(lm,24); lk=safe(lm,25); rk=safe(lm,26); la=safe(lm,27); ra=safe(lm,28)
    sh=mid(ls,rs); hip=mid(lh,rh)
    A={"shoulder_tilt":tilt(ls,rs) if ls and rs else None,"pelvis_tilt":tilt(lh,rh) if lh and rh else None,"head_forward":vdev(nose,sh) if nose and sh else None,"trunk_lean":vdev(sh,hip) if sh and hip else None,"left_knee":angle(lh,lk,la) if lh and lk and la else None,"right_knee":angle(rh,rk,ra) if rh and rk and ra else None,"left_elbow":angle(ls,le,lw) if ls and le and lw else None,"right_elbow":angle(rs,re,rw) if rs and re and rw else None}
    M={"shoulder_asymmetry":0,"pelvis_asymmetry":0,"head_forward_ratio":0,"rounded_shoulder_ratio":0,"knee_valgus_ratio":0}
    if ls and rs: M["shoulder_asymmetry"]=abs(ls["y"]-rs["y"])/max(.0001,dist(ls,rs))
    if lh and rh: M["pelvis_asymmetry"]=abs(lh["y"]-rh["y"])/max(.0001,dist(lh,rh))
    if nose and sh and ls and rs: M["head_forward_ratio"]=abs(nose["x"]-sh["x"])/max(.0001,dist(ls,rs))
    if sh and hip and ls and rs: M["rounded_shoulder_ratio"]=abs(sh["x"]-hip["x"])/max(.0001,dist(ls,rs))
    if lk and rk and la and ra:
        kg=abs(lk["x"]-rk["x"]); ag=abs(la["x"]-ra["x"])
        if ag>0: M["knee_valgus_ratio"]=max(0,(ag-kg)/ag)
    return {"view":view,"ok":True,"landmark_quality":q,"angles":A,"metrics":M,"overlay":make_overlay(img_bgr,lm,view,A,True),"landmarks":landmarks}

def build_bridge(score,risk,levels,fixes,stretches,strengthen,avoid,chain):
    return {"priority":"corrective_block_first" if score<88 else "maintenance_block","posture_score":score,"posture_risk":risk,"levels":levels,"corrective_block":unique(["90/90 Breathing 2x5 nefes","Cat-Camel 1x8","Wall Slide 2x10"]+fixes[:5]),"corrective_exercises":unique(fixes),"stretching_focus":unique(stretches),"strengthening_focus":unique(strengthen),"avoid_exercises":unique(avoid),"program_notes":unique(chain),"program_rule":"Ana antrenmandan önce 8-12 dakikalık düzeltici blok ekle; kaçınılacak hareketleri güvenli varyasyonla değiştir."}

def report_html(data):
    a=data.get("angles",{})
    li=lambda arr:"".join(f"<li>{x}</li>" for x in (arr or []))
    return f"<div style='font-family:Arial;max-width:900px;margin:auto'><h1>MergeSport Postür Analiz Raporu</h1><h2>Skor: {data.get('score')}/100 | Risk: {data.get('risk')} | Landmark: {data.get('landmark_quality')}%</h2><h3>Açılar</h3><table border='1' cellpadding='8'><tr><td>Omuz</td><td>{deg(a.get('shoulder_tilt'))}</td></tr><tr><td>Pelvis</td><td>{deg(a.get('pelvis_tilt'))}</td></tr><tr><td>Baş</td><td>{deg(a.get('head_forward'))}</td></tr><tr><td>Gövde</td><td>{deg(a.get('trunk_lean'))}</td></tr><tr><td>Sol Diz</td><td>{deg(a.get('left_knee'))}</td></tr><tr><td>Sağ Diz</td><td>{deg(a.get('right_knee'))}</td></tr></table><h3>Bulgular</h3><ul>{li(data.get('findings'))}</ul><h3>Düzeltici Egzersizler</h3><ul>{li(data.get('fixes'))}</ul><h3>Chain Analysis</h3><ul>{li(data.get('chain'))}</ul></div>"

def build_report(front=None,side=None,back=None):
    analyses=[x for x in [front,side,back] if x]
    ok=[x for x in analyses if x.get("ok")]
    overlays={"front":front.get("overlay") if front else "","side":side.get("overlay") if side else "","back":back.get("overlay") if back else ""}
    if not ok:
        data={"score":70,"risk":"Sınırlı","analysis_source":"Backend MediaPipe - Landmark alınamadı","landmark_quality":0,"angles":{},"metrics":{},"levels":{"head":"limited","shoulder":"limited","pelvis":"limited","knee":"limited"},"findings":["Fotoğraflardan landmark alınamadı."],"fixes":["Wall Posture Hold 2x30 sn","Dead Bug 2x10","Face Pull 2x12"],"stretches":[],"strengthen":[],"avoid":[],"chain":["Tam analiz için baştan ayağa tam kadraj, iyi ışık ve sade arka plan gerekir."],"overlays":overlays}
    else:
        q=round(sum(x["landmark_quality"] for x in ok)/len(ok))
        A={}
        M={"shoulder_asymmetry":0,"pelvis_asymmetry":0,"head_forward_ratio":0,"rounded_shoulder_ratio":0,"knee_valgus_ratio":0}
        for it in ok:
            for k,v in it.get("angles",{}).items():
                if v is not None and k not in A: A[k]=v
            for k,v in it.get("metrics",{}).items():
                M[k]=max(M.get(k,0),v or 0)
        L={"head":sev(M["head_forward_ratio"]),"shoulder":sev(max(M["shoulder_asymmetry"],M["rounded_shoulder_ratio"])),"pelvis":sev(M["pelvis_asymmetry"]),"knee":sev(M["knee_valgus_ratio"])}
        score=100; pen={"normal":0,"hafif":6,"orta":12,"belirgin":18,"limited":5}
        for v in L.values(): score-=pen.get(v,0)
        score=max(40,score)
        risk="Düşük" if score>=88 else "Hafif" if score>=75 else "Orta" if score>=60 else "Belirgin"
        findings=[]; fixes=[]; stretches=[]; strengthen=[]; avoid=[]; chain=[]
        if L["head"]!="normal":
            findings.append(f"Baş-boyun hizası: {lab(L['head'])}"); fixes+=["Chin Tuck 2x10","Wall Angel 2x8"]; chain.append("Başın öne taşınması boyun ekstansör yükünü artırabilir.")
        if L["shoulder"]!="normal":
            findings.append(f"Omuz kuşağı / protraksyon: {lab(L['shoulder'])}"); fixes+=["Face Pull 3x12","Band Pull Apart 2x15","Thoracic Extension 2x8"]; stretches+=["Pectoralis minor","Anterior deltoid"]; strengthen+=["Lower trapezius","Rhomboid","External rotator"]; avoid+=["Kontrolsüz dips","Aşırı öne kapanarak yapılan press varyasyonları"]
        if L["pelvis"]!="normal":
            findings.append(f"Pelvis hizası / lordoz eğilimi: {lab(L['pelvis'])}"); fixes+=["Dead Bug 2x10","Glute Bridge 2x12","Hip Flexor Stretch 2x30 sn"]; stretches+=["Iliopsoas","Rectus femoris"]; strengthen+=["Gluteus maximus","Core brace"]; avoid+=["Kontrolsüz lumbar extension","Aşırı bel boşluğu ile overhead press"]
        if L["knee"]!="normal":
            findings.append(f"Diz/ayak dizilimi: {lab(L['knee'])}"); fixes+=["Mini Band Lateral Walk 2x12","Short Foot Drill 2x10","Single Leg Balance 2x20 sn"]; strengthen+=["Gluteus medius","Ayak intrensek kasları"]; avoid+=["Diz içe kaçarak yapılan squat/lunge"]
        if not findings:
            findings.append("Okunabilen landmarklara göre belirgin postür problemi saptanmadı."); fixes+=["Face Pull 2x12","Dead Bug 2x10","Hip Hinge Drill 2x8"]
        data={"score":score,"risk":risk,"analysis_source":"Backend MediaPipe + Landmark + Vektör","landmark_quality":q,"angles":A,"metrics":M,"levels":L,"findings":unique(findings),"fixes":unique(fixes),"stretches":unique(stretches),"strengthen":unique(strengthen),"avoid":unique(avoid),"chain":unique(chain),"overlays":overlays}
    data["program_bridge"]=build_bridge(data["score"],data["risk"],data["levels"],data["fixes"],data["stretches"],data["strengthen"],data["avoid"],data["chain"])
    data["report_html"]=report_html(data)
    return data

@app.get("/")
def health():
    return {"ok":True,"service":"MergeSport Posture + Program API","version":"3.0.0"}

@app.post("/analyze")
async def analyze(front: UploadFile = File(None), side: UploadFile = File(None), back: UploadFile = File(None)):
    if not front and not side and not back:
        raise HTTPException(status_code=400, detail="En az bir fotoğraf gönderilmelidir.")
    f=analyze_img(read_image(await front.read()),"front") if front else None
    s=analyze_img(read_image(await side.read()),"side") if side else None
    b=analyze_img(read_image(await back.read()),"back") if back else None
    return build_report(f,s,b)
