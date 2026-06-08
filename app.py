"""
app.py — What this file does:
Flask web server. User uploads a leaf photo → model predicts disease → shows result.

Run with:  python app.py
Then open: http://localhost:5000
"""

import io
import os
import tempfile
from datetime import datetime

import torch
import torchvision.transforms.functional as TF
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from torchvision import transforms
from fpdf import FPDF

import config
from model import build_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ── Load model once at startup (not on every request) ────────────
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    config.MODEL_SAVE_PATH / "best_model.pth",
    map_location=device, weights_only=False
)
model = build_model(freeze_backbone=False).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()
CLASS_NAMES = checkpoint['class_names']

transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Detailed disease info: cause, severity, symptoms, treatment steps, prevention
DISEASE_INFO = {
    "Cashew anthracnose": {
        "cause": "Fungal — Colletotrichum gloeosporioides",
        "severity": "high",
        "symptoms": "Dark brown to black lesions on leaves, flowers, and fruits. Infected fruits shrivel and drop early.",
        "treatment": [
            "Remove and destroy all infected plant parts immediately",
            "Apply copper-based fungicide (Copper Oxychloride 50% WP) every 10–14 days",
            "Use Mancozeb 75% WP spray during flowering and fruiting stages",
            "Avoid overhead irrigation — wet leaves worsen spread",
        ],
        "prevention": "Use disease-free planting material. Ensure good air circulation by pruning crowded branches. Apply fungicide preventively at the start of the rainy season.",
    },
    "Cashew gumosis": {
        "cause": "Fungal — Lasiodiplodia theobromae",
        "severity": "medium",
        "symptoms": "Gum oozing from the stem or branches, bark cracking, branch dieback, and yellowing of leaves.",
        "treatment": [
            "Prune and remove all infected branches 15cm below visible infection",
            "Apply Bordeaux paste or copper fungicide to cut surfaces",
            "Drench soil around the base with Carbendazim 50% WP",
            "Avoid mechanical damage to the stem during farming",
        ],
        "prevention": "Do not wound stems during harvesting. Maintain good drainage around roots. Avoid waterlogged conditions.",
    },
    "Cashew healthy": {
        "cause": "No disease detected",
        "severity": "none",
        "symptoms": "Leaves are green, firm, and show no spots, lesions, or discolouration.",
        "treatment": ["No treatment needed — your plant is healthy!"],
        "prevention": "Continue regular monitoring. Water consistently, avoid over-fertilising with nitrogen, and inspect weekly for early signs of pests or disease.",
    },
    "Cashew leaf miner": {
        "cause": "Insect pest — Acrocercops syngramma (moth larvae)",
        "severity": "medium",
        "symptoms": "Silvery winding trails (mines) on leaves, blister-like patches, leaves curling and drying up.",
        "treatment": [
            "Spray Dimethoate 30% EC or Imidacloprid 17.8% SL at first sign",
            "Remove and burn heavily infested leaves",
            "Apply neem-based insecticide (Azadirachtin) as an eco-friendly option",
            "Repeat spray every 14 days until infestation clears",
        ],
        "prevention": "Monitor new flushes of growth closely — larvae attack young leaves. Use yellow sticky traps to detect adult moths early.",
    },
    "Cashew red rust": {
        "cause": "Algal — Cephaleuros virescens",
        "severity": "low",
        "symptoms": "Rusty orange to brick-red velvety patches on upper leaf surface, slight raised texture.",
        "treatment": [
            "Spray Copper Oxychloride 50% WP (3g per litre of water)",
            "Apply 2–3 sprays at 14-day intervals",
            "Remove heavily infected leaves and dispose of them away from the farm",
        ],
        "prevention": "Improve sunlight penetration by thinning dense canopies. Avoid excessive shade and maintain good drainage.",
    },
    "Cassava bacterial blight": {
        "cause": "Bacterial — Xanthomonas axonopodis pv. manihotis",
        "severity": "high",
        "symptoms": "Angular water-soaked spots on leaves, wilting of shoot tips, stem cankers, sticky exudate on stems.",
        "treatment": [
            "Remove and destroy infected plants — there is no chemical cure",
            "Disinfect farming tools with 10% bleach solution between uses",
            "Plant only disease-free certified cuttings in next season",
            "Apply copper-based bactericide as a protective spray on remaining plants",
        ],
        "prevention": "Use resistant cassava varieties (e.g., TME 419, IITA varieties). Rotate crops. Never replant infected cuttings.",
    },
    "Cassava brown spot": {
        "cause": "Fungal — Cercosporidium henningsii",
        "severity": "medium",
        "symptoms": "Circular brown spots with yellow halos on leaves, premature leaf drop, reduced photosynthesis.",
        "treatment": [
            "Spray Mancozeb 75% WP (2.5g per litre) every 2 weeks",
            "Remove and bury infected leaves to reduce fungal spore load",
            "Apply Carbendazim 50% WP for severe infections",
        ],
        "prevention": "Plant resistant varieties. Avoid dense planting that reduces airflow. Remove crop debris after harvest.",
    },
    "Cassava green mite": {
        "cause": "Pest — Mononychellus tanajoa (mite)",
        "severity": "high",
        "symptoms": "Pale yellow speckling on young leaves, leaf distortion and stunting, severe leaf drop in dry conditions.",
        "treatment": [
            "Spray Abamectin 1.8% EC — highly effective against mites",
            "Apply neem oil solution (5ml per litre) as organic alternative",
            "Introduce predatory mites (Typhlodromalus aripo) for biological control",
            "Avoid water stress — mites thrive on stressed plants",
        ],
        "prevention": "Intercrop with plants that harbour natural predators. Do not use broad-spectrum insecticides that kill mite predators.",
    },
    "Cassava healthy": {
        "cause": "No disease detected",
        "severity": "none",
        "symptoms": "Leaves are dark green, no spots or distortion visible. Plant growing normally.",
        "treatment": ["No treatment needed — your plant is healthy!"],
        "prevention": "Top-dress with NPK fertiliser at 8 weeks after planting. Keep field weed-free in first 3 months. Monitor for whiteflies that spread mosaic virus.",
    },
    "Cassava mosaic": {
        "cause": "Viral — Cassava Mosaic Virus (CMV), spread by whiteflies",
        "severity": "high",
        "symptoms": "Mosaic yellowing and green patches on leaves, leaf distortion, stunted growth, twisted leaflets.",
        "treatment": [
            "There is NO chemical cure for viral diseases",
            "Remove and destroy all infected plants immediately to stop spread",
            "Control whitefly vectors using Imidacloprid or yellow sticky traps",
            "Replant with virus-free certified cuttings from healthy plants",
        ],
        "prevention": "Use CMV-resistant varieties (TME 419, NASE 14). Never take cuttings from infected plants. Rogue out infected plants early.",
    },
    "Maize fall armyworm": {
        "cause": "Pest — Spodoptera frugiperda (caterpillar)",
        "severity": "high",
        "symptoms": "Ragged holes in leaves, frass (sawdust-like droppings) in the whorl, window-pane feeding on young plants.",
        "treatment": [
            "Apply insecticide directly into the whorl (centre funnel) of the plant",
            "Use Chlorpyrifos 20% EC or Emamectin Benzoate 5% SG",
            "Apply in the early morning or evening when larvae are active",
            "Spray neem seed kernel extract (50g/litre) as an organic option",
        ],
        "prevention": "Plant early to avoid peak moth seasons. Monitor fields twice a week. Use pheromone traps to detect adult moths before egg-laying.",
    },
    "Maize grasshoper": {
        "cause": "Pest — Various grasshopper species",
        "severity": "medium",
        "symptoms": "Ragged leaf edges, large chewed holes in leaves, defoliation in severe cases.",
        "treatment": [
            "Spray Malathion 50% EC or Lambda-cyhalothrin 5% EC",
            "Apply Metarhizium anisopliae (biological fungal pesticide) for organic farms",
            "Treat field borders first where grasshoppers enter from",
        ],
        "prevention": "Clear weeds and crop residues around field borders. Avoid late planting. Encourage birds by maintaining some trees nearby.",
    },
    "Maize healthy": {
        "cause": "No disease detected",
        "severity": "none",
        "symptoms": "Leaves are green with no spots, lesions, or pest damage visible.",
        "treatment": ["No treatment needed — your plant is healthy!"],
        "prevention": "Top-dress with urea at the knee-high stage. Ensure adequate soil moisture. Scout fields weekly for fall armyworm in the whorl.",
    },
    "Maize leaf beetle": {
        "cause": "Pest — Chaetocnema species (flea beetle)",
        "severity": "medium",
        "symptoms": "Tiny round holes scattered across leaves, silvery streaks, stunted seedlings in heavy infestations.",
        "treatment": [
            "Spray Carbofuran 3G granules or Chlorpyrifos 20% EC",
            "Apply Kaolin clay spray as a physical barrier on young seedlings",
            "For severe infestation, use Lambda-cyhalothrin 5% EC",
        ],
        "prevention": "Treat seeds with insecticide before planting. Early planting reduces exposure during beetle peak. Remove crop residues after harvest.",
    },
    "Maize leaf blight": {
        "cause": "Fungal — Exserohilum turcicum (Northern Leaf Blight)",
        "severity": "high",
        "symptoms": "Long cigar-shaped grey-green to tan lesions on leaves, lesions may have dark borders, premature leaf death.",
        "treatment": [
            "Spray Propiconazole 25% EC or Tebuconazole 25% WG",
            "Apply at first appearance of lesions and repeat every 14 days",
            "Remove severely infected leaves to reduce spore spread",
        ],
        "prevention": "Plant resistant hybrids. Practice crop rotation with non-grass crops. Bury or compost infected crop debris after harvest.",
    },
    "Maize leaf spot": {
        "cause": "Fungal — Kabatiella zeae or Cercospora zeae-maydis",
        "severity": "medium",
        "symptoms": "Small round to oval tan or grey spots with dark brown borders on leaves, spots may merge in severe cases.",
        "treatment": [
            "Apply Mancozeb 75% WP or Chlorothalonil 75% WP spray",
            "Spray every 10–14 days during humid weather",
            "Improve air circulation by maintaining proper plant spacing",
        ],
        "prevention": "Use resistant varieties. Rotate with legumes or other non-maize crops. Avoid overhead irrigation in the evening.",
    },
    "Maize streak virus": {
        "cause": "Viral — Maize Streak Virus (MSV), spread by leafhoppers",
        "severity": "high",
        "symptoms": "Narrow yellow streaks running along leaf veins, stunted growth, pale green to yellow overall colour.",
        "treatment": [
            "No chemical cure — remove and destroy infected plants",
            "Control leafhopper vectors with Imidacloprid 70% WS seed treatment",
            "Spray Thiamethoxam 25% WG on remaining plants to prevent leafhopper spread",
        ],
        "prevention": "Plant MSV-resistant varieties (KSTP 94, H614D). Plant during cooler months when leafhoppers are less active. Use reflective mulches to repel leafhoppers.",
    },
    "Tomato healthy": {
        "cause": "No disease detected",
        "severity": "none",
        "symptoms": "Leaves are dark green, firm, and free of spots, yellowing, or lesions.",
        "treatment": ["No treatment needed — your plant is healthy!"],
        "prevention": "Water at the base, never on foliage. Stake plants for good air circulation. Apply balanced NPK fertiliser. Scout weekly for early blight or pest signs.",
    },
    "Tomato leaf blight": {
        "cause": "Fungal — Alternaria solani (Early Blight)",
        "severity": "high",
        "symptoms": "Dark brown spots with concentric rings (target-board pattern) on older leaves, yellow halo around spots, defoliation.",
        "treatment": [
            "Remove and destroy all infected lower leaves immediately",
            "Spray Chlorothalonil 75% WP or Mancozeb 75% WP every 7–10 days",
            "Apply Azoxystrobin + Difenoconazole for systemic protection",
            "Avoid wetting foliage during irrigation",
        ],
        "prevention": "Mulch soil to prevent spore splash from soil. Rotate tomatoes with non-solanaceous crops for 2–3 years. Remove crop debris after harvest.",
    },
    "Tomato leaf curl": {
        "cause": "Viral — Tomato Leaf Curl Virus (ToLCV), spread by whiteflies",
        "severity": "high",
        "symptoms": "Upward curling and cupping of leaves, leaf thickening, stunted plant growth, reduced fruit set.",
        "treatment": [
            "No chemical cure for viral infection",
            "Remove and destroy infected plants to prevent whitefly spread",
            "Apply Imidacloprid 17.8% SL or Thiamethoxam 25% WG to control whiteflies",
            "Use yellow sticky traps to monitor and reduce whitefly population",
        ],
        "prevention": "Use virus-resistant tomato varieties (e.g., Arka Rakshak, Sahel). Install 40-mesh insect-proof nets in nursery. Avoid planting near infected fields.",
    },
    "Tomato septoria leaf spot": {
        "cause": "Fungal — Septoria lycopersici",
        "severity": "medium",
        "symptoms": "Numerous small circular spots (2–4mm) with dark borders and light grey centres on lower leaves first, then spreading upward.",
        "treatment": [
            "Remove all infected lower leaves and dispose away from field",
            "Spray Chlorothalonil 75% WP or Copper Oxychloride 50% WP",
            "Apply every 7–10 days during wet weather",
            "Ensure plants are well-spaced for air circulation",
        ],
        "prevention": "Stake plants off the ground. Mulch to prevent soil splash. Rotate crops for at least 2 years. Water at the base only.",
    },
    "Tomato verticulium wilt": {
        "cause": "Fungal — Verticillium dahliae (soil-borne)",
        "severity": "high",
        "symptoms": "V-shaped yellow lesions on leaf margins, one-sided wilting, brown discolouration inside stems when cut, lower leaves wilting first.",
        "treatment": [
            "There is NO effective cure once the plant is infected",
            "Remove and destroy infected plants — do NOT compost them",
            "Solarise the soil (cover with clear plastic for 4–6 weeks in hot sun) to kill fungus",
            "Apply Trichoderma viride biological agent to reduce soil fungal load",
        ],
        "prevention": "Plant resistant varieties (VF hybrids). Rotate with non-solanaceous crops for 3–4 years. Improve soil drainage. Never plant tomatoes in previously infected soil.",
    },
}


def predict_image(img: Image.Image):
    """Run TTA (5 augmented views) and return top 3 predictions."""
    views = [
        img,
        TF.hflip(img),
        TF.vflip(img),
        TF.rotate(img, 90),
        TF.rotate(img, 270),
    ]
    probs_list = []
    with torch.no_grad():
        for view in views:
            t = transform(view).unsqueeze(0).to(device)
            probs_list.append(torch.softmax(model(t), dim=1))

    avg_probs           = torch.stack(probs_list).mean(dim=0)
    top3_probs, top3_idx = avg_probs.topk(3, dim=1)

    results = []
    for i in range(3):
        name  = CLASS_NAMES[top3_idx[0][i].item()]
        prob  = round(top3_probs[0][i].item() * 100, 1)
        results.append({"disease": name, "confidence": prob})

    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        results   = predict_image(img)

        top      = results[0]
        low_conf = top["confidence"] < 60
        info     = DISEASE_INFO.get(top["disease"], {
            "cause": "Unknown", "severity": "unknown",
            "symptoms": "No information available.",
            "treatment": ["Consult an agronomist."],
            "prevention": "No information available.",
        })

        return jsonify({
            "top":      top,
            "top3":     results,
            "info":     info,
            "low_conf": low_conf,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def sanitize(text: str) -> str:
    """Replace Unicode characters that Helvetica can't encode."""
    return (str(text)
        .replace("—", " - ")   # em dash
        .replace("–", " - ")   # en dash
        .replace("‘", "'").replace("’", "'")   # curly quotes
        .replace("“", '"').replace("”", '"')
        .replace("é", "e").replace("è", "e")   # accented chars
        .replace("°", " deg")
        .replace("±", "+/-")
        .replace("•", "-")     # bullet
        .replace(" ", " ")     # non-breaking space
    )


def generate_pdf(img: Image.Image, results: list, info: dict) -> bytes:
    """Build a styled PDF report and return it as bytes."""

    top      = results[0]
    disease  = top["disease"]
    conf     = top["confidence"]
    severity = info.get("severity", "unknown")
    now      = datetime.now().strftime("%d %B %Y  %H:%M")

    # Severity colours (R, G, B)
    sev_color = {
        "high":   (220, 80,  80),
        "medium": (230, 160, 40),
        "low":    (80,  160, 80),
        "none":   (60,  160, 100),
    }.get(severity, (120, 120, 120))

    pdf = FPDF()
    pdf.set_margins(16, 16, 16)
    pdf.add_page()
    W = pdf.w - 32   # usable width

    # ── Header bar ───────────────────────────────────────────────
    pdf.set_fill_color(20, 80, 40)
    pdf.rect(0, 0, pdf.w, 28, 'F')
    pdf.set_xy(16, 7)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(W * 0.6, 8, "PlantDoc AI", ln=0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 230, 180)
    pdf.set_xy(16, 16)
    pdf.cell(W * 0.6, 6, "Plant Disease Diagnosis Report", ln=0)
    pdf.set_xy(pdf.w - 70, 10)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(200, 240, 200)
    pdf.cell(60, 5, now, align="R")

    pdf.set_y(35)

    # ── Save leaf image to temp file and embed ────────────────────
    tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        thumb = img.copy()
        thumb.thumbnail((200, 200))
        thumb.save(tmp_img.name, "JPEG")
        tmp_img.close()

        # Image on the left
        pdf.image(tmp_img.name, x=16, y=35, w=52)
    except Exception:
        pass
    finally:
        try: os.unlink(tmp_img.name)
        except: pass

    # ── Diagnosis box on the right ────────────────────────────────
    pdf.set_xy(74, 35)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 130, 100)
    pdf.cell(W - 58, 5, "DIAGNOSIS RESULT", ln=1)

    pdf.set_xy(74, 41)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(15, 60, 25)
    pdf.multi_cell(W - 58, 7, sanitize(disease))

    # Confidence pill
    cx = 74
    cy = pdf.get_y() + 2
    pill_w = 60
    pdf.set_fill_color(230, 245, 230)
    pdf.set_draw_color(120, 190, 120)
    pdf.set_xy(cx, cy)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 100, 50)
    pdf.cell(pill_w, 7, f"Confidence: {conf}%", border=1, fill=True, ln=0)

    # Severity badge
    pdf.set_xy(cx + pill_w + 4, cy)
    pdf.set_fill_color(*sev_color)
    pdf.set_draw_color(*sev_color)
    pdf.set_text_color(255, 255, 255)
    sev_label = {"high": "HIGH SEVERITY", "medium": "MEDIUM", "low": "LOW", "none": "HEALTHY"}.get(severity, severity.upper())
    pdf.cell(40, 7, sev_label, border=1, fill=True, ln=1)

    pdf.set_y(max(pdf.get_y(), 93))  # push below the image

    def section(title, color_rgb, content_fn):
        """Draw a coloured section header then call content_fn for the body."""
        pdf.set_y(pdf.get_y() + 5)
        # Coloured left bar
        pdf.set_fill_color(*color_rgb)
        pdf.rect(16, pdf.get_y(), 4, 6, 'F')
        pdf.set_xy(22, pdf.get_y())
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*color_rgb)
        pdf.cell(W - 6, 6, title, ln=1)
        pdf.set_xy(16, pdf.get_y() + 1)
        pdf.set_text_color(50, 50, 50)
        pdf.set_font("Helvetica", "", 9)
        content_fn()
        pdf.set_y(pdf.get_y() + 2)
        # Divider line
        pdf.set_draw_color(220, 220, 220)
        pdf.line(16, pdf.get_y(), pdf.w - 16, pdf.get_y())

    def body_text(text):
        pdf.multi_cell(W, 5, sanitize(text))

    def treatment_steps(steps):
        for i, step in enumerate(steps, 1):
            pdf.set_xy(16, pdf.get_y())
            pdf.set_fill_color(40, 120, 60)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 7)
            pdf.cell(6, 5, str(i), fill=True, align="C")
            pdf.set_x(24)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(W - 8, 5, sanitize(step))
            pdf.set_y(pdf.get_y() + 1)

    section("CAUSE",           (30, 100, 200), lambda: body_text(info.get("cause", "")))
    section("SYMPTOMS",        (200, 130, 30), lambda: body_text(info.get("symptoms", "")))
    section("TREATMENT STEPS", (40, 140, 70),  lambda: treatment_steps(info.get("treatment", [])))
    section("PREVENTION",      (130, 60, 180), lambda: body_text(info.get("prevention", "")))

    # ── Top 3 table ───────────────────────────────────────────────
    pdf.set_y(pdf.get_y() + 6)
    pdf.set_fill_color(20, 80, 40)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(16)
    pdf.cell(W * 0.65, 7, "Disease", fill=True, border=0)
    pdf.cell(W * 0.35, 7, "Confidence", fill=True, border=0, align="R", ln=1)

    bar_colors = [(40, 140, 70), (30, 110, 55), (20, 80, 40)]
    for i, r in enumerate(results[:3]):
        pdf.set_x(16)
        pdf.set_fill_color(245, 250, 245) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "B" if i == 0 else "", 9)
        pdf.cell(W * 0.65, 6.5, sanitize(r["disease"]), fill=True)
        # Mini bar
        bar_max = W * 0.28
        bar_w   = bar_max * r["confidence"] / 100
        bx      = pdf.get_x()
        by      = pdf.get_y()
        pdf.set_fill_color(230, 240, 230)
        pdf.rect(bx, by + 1.5, bar_max, 3.5, 'F')
        pdf.set_fill_color(*bar_colors[i])
        pdf.rect(bx, by + 1.5, bar_w, 3.5, 'F')
        pdf.set_xy(bx + bar_max + 2, by)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*bar_colors[i])
        pdf.cell(14, 6.5, f"{r['confidence']}%", align="R", ln=1)

    # ── Footer ────────────────────────────────────────────────────
    pdf.set_y(pdf.h - 18)
    pdf.set_fill_color(20, 80, 40)
    pdf.rect(0, pdf.h - 14, pdf.w, 14, 'F')
    pdf.set_xy(16, pdf.h - 11)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(180, 230, 180)
    pdf.cell(W / 2, 5, "Powered by MobileNetV2 · PyTorch · PlantDoc AI", ln=0)
    pdf.set_xy(pdf.w / 2, pdf.h - 11)
    pdf.cell(W / 2, 5, "This report is AI-generated. Consult an agronomist for final decisions.", align="R")

    return pdf.output()


@app.route("/report", methods=["POST"])
def download_report():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    try:
        img_bytes = request.files["file"].read()
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        results   = predict_image(img)
        top       = results[0]
        info      = DISEASE_INFO.get(top["disease"], {
            "cause": "Unknown", "severity": "unknown",
            "symptoms": "No information available.",
            "treatment": ["Consult an agronomist."],
            "prevention": "No information available.",
        })
        pdf_bytes = generate_pdf(img, results, info)
        filename  = f"PlantDoc_Report_{top['disease'].replace(' ', '_')}.pdf"
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))   # HF Spaces sets PORT=7860
    print(f"Model loaded on device: {device}")
    print(f"Open http://localhost:{port} in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=port)
