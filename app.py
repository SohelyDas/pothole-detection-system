import streamlit as st
import numpy as np
import cv2
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from datetime import datetime

from detect import load_model, detect_potholes, annotate_image

st.set_page_config(page_title="Pothole Detection System", page_icon="🕳️", layout="wide")

# ============================================================
# CONFIG — replace these with your own Google Form details
# (see setup instructions in the comment block at the bottom)
# ============================================================
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/YOUR_FORM_ID/formResponse"
FORM_FIELDS = {
    "name": "entry.111111111",
    "issue_type": "entry.222222222",
    "message": "entry.333333333",
}

SEVERITY_COLORS = {"Minor": "#27AE60", "Moderate": "#F2994A", "Severe": "#EB5757"}
SEVERITY_ICONS = {"Minor": "🟢", "Moderate": "⚠️", "Severe": "🚨"}


# ============================================================
# STYLE
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

    :root {
        --bg: #14161A;
        --surface: #1E2126;
        --surface-2: #262A30;
        --accent: #FFC72C;
        --accent-dim: rgba(255,199,44,0.14);
        --blue: #2D9CDB;
        --orange: #F2994A;
        --green: #27AE60;
        --red: #EB5757;
        --lane: #ECE7DD;
        --text: #F5F5F3;
        --text-muted: #8B9096;
        --border: rgba(255,255,255,0.08);
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: var(--bg); color: var(--text); }
    .block-container { padding-top: 2rem; max-width: 1100px; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        background:
            radial-gradient(circle at 15% 20%, rgba(255,199,44,0.16), transparent 40%),
            radial-gradient(circle at 85% 15%, rgba(45,156,219,0.14), transparent 45%),
            radial-gradient(circle at 60% 90%, rgba(235,87,87,0.10), transparent 50%),
            linear-gradient(180deg, #191C21 0%, #14161A 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 2.4rem 2.2rem 1.6rem 2.2rem;
        margin-bottom: 1.8rem;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute; top: 0; left: -20%;
        width: 45%; height: 100%;
        background: linear-gradient(100deg, transparent, rgba(255,199,44,0.14), transparent);
        animation: sweep 5s ease-in-out infinite;
    }
    @keyframes sweep {
        0%   { left: -30%; }
        50%  { left: 100%; }
        100% { left: 100%; }
    }
    .hero .eyebrow {
        font-family:'IBM Plex Mono', monospace; font-size:0.72rem; letter-spacing:0.14em;
        text-transform:uppercase; color: var(--eyebrow-color, var(--accent));
        background: var(--eyebrow-bg, var(--accent-dim));
        padding: 4px 10px; border-radius: 5px; border: 1px solid var(--eyebrow-border, rgba(255,199,44,0.25));
        position: relative; z-index: 1;
    }
    .hero h1 { margin: 0.6rem 0 0.3rem 0; font-size: 2.1rem; position: relative; z-index: 1; }
    .hero p { color: var(--text-muted); margin: 0; position: relative; z-index: 1; max-width: 60ch; }
    .zebra-crossing {
        margin-top: 1.6rem;
        height: 16px; width: 100%;
        border-radius: 3px;
        background-image: repeating-linear-gradient(90deg, var(--lane) 0 16px, transparent 16px 30px);
        opacity: 0.85;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] { background-color: var(--surface); border-right: 1px solid var(--border); }
    .avatar {
        width: 38px; height: 38px; border-radius: 50%;
        background: linear-gradient(135deg, var(--accent), var(--orange));
        color: #14161A; font-family:'Space Grotesk', sans-serif;
        font-weight: 700; display:flex; align-items:center; justify-content:center;
        font-size: 1rem;
    }
    .side-user { display:flex; align-items:center; gap:0.6rem; margin-bottom: 0.4rem; }
    .side-user .name { font-weight: 600; font-size: 0.95rem; }
    .side-user .tag { font-size: 0.72rem; color: var(--text-muted); }

    /* ---------- Cards ---------- */
    .card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 10px; padding: 1.1rem 1.3rem; margin-bottom: 0.9rem;
    }
    .detection-card {
        background: var(--surface); border: 1px solid var(--border);
        border-left: 3px solid var(--sev-color, var(--accent));
        border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 0.55rem;
        transition: border-color 0.15s ease, transform 0.15s ease;
    }
    .detection-card:hover { transform: translateX(2px); border-color: var(--sev-color, var(--accent)); }
    .detection-card .row { display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem; }
    .detection-card .id { font-family:'IBM Plex Mono', monospace; color: var(--text-muted); font-size:0.82rem; letter-spacing: 0.04em; }
    .detection-card .conf { font-family:'IBM Plex Mono', monospace; font-size:1.05rem; font-weight:600; }
    .sev-pill {
        font-family:'IBM Plex Mono', monospace; font-size:0.74rem; font-weight:600;
        padding: 3px 10px; border-radius: 999px; color:#14161A;
    }
    .conf-track { height: 5px; border-radius: 3px; background: var(--surface-2); overflow: hidden; }
    .conf-fill { height: 100%; border-radius: 3px; background: var(--sev-color, var(--accent)); }

    /* ---------- Stats ---------- */
    .stat { text-align:center; padding: 0.4rem 0; }
    .stat .num { font-family:'Space Grotesk', sans-serif; font-size:2.1rem; font-weight:700; color: var(--stat-color, var(--accent)); line-height:1; }
    .stat .label { font-family:'IBM Plex Mono', monospace; font-size:0.7rem; letter-spacing:0.09em;
        text-transform:uppercase; color: var(--text-muted); margin-top: 0.35rem; }

    /* ---------- Empty state ---------- */
    .empty-state {
        text-align:center; padding: 3rem 1rem; border: 1.5px dashed var(--border);
        border-radius: 12px; background: var(--surface);
    }
    .empty-state .icon { font-size: 2.4rem; margin-bottom: 0.6rem; opacity: 0.95; filter: saturate(1.3); }
    .empty-state .title { font-family:'Space Grotesk', sans-serif; font-weight:600; font-size:1.05rem; margin-bottom:0.3rem; }
    .empty-state .sub { color: var(--text-muted); font-size:0.9rem; }

    /* ---------- Buttons & inputs ---------- */
    .stButton > button, .stDownloadButton > button {
        background: var(--accent); color: #14161A; border: none;
        font-weight: 600; border-radius: 8px; padding: 0.55rem 1.1rem;
        transition: filter 0.15s ease, transform 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { filter: brightness(1.08); transform: translateY(-1px); }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
        outline: 2px solid var(--accent); outline-offset: 2px;
    }
    div[data-testid="column"]:nth-of-type(2) .stDownloadButton > button {
        background: var(--blue); color: #14161A;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface); border: 1.5px dashed var(--border); border-radius: 10px;
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background: var(--surface-2) !important; color: var(--text) !important;
        border-radius: 8px !important; border: 1px solid var(--border) !important;
    }
    .stRadio > div { gap: 0.3rem; }
    hr { border-color: var(--border) !important; }

    #MainMenu, footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; height: 2.5rem; }
    [data-testid="collapsedControl"] { visibility: visible !important; }
    </style>
    """, unsafe_allow_html=True)


def hero(eyebrow, title, subtitle, color="accent"):
    palette = {
        "accent": ("var(--accent)", "var(--accent-dim)", "rgba(255,199,44,0.25)"),
        "blue": ("var(--blue)", "rgba(45,156,219,0.14)", "rgba(45,156,219,0.3)"),
        "red": ("var(--red)", "rgba(235,87,87,0.14)", "rgba(235,87,87,0.3)"),
        "green": ("var(--green)", "rgba(39,174,96,0.14)", "rgba(39,174,96,0.3)"),
    }
    fg, bg, bd = palette.get(color, palette["accent"])
    st.markdown(f"""
        <div class="hero" style="--eyebrow-color:{fg}; --eyebrow-bg:{bg}; --eyebrow-border:{bd};">
            <span class="eyebrow">{eyebrow}</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="zebra-crossing"></div>
        </div>
    """, unsafe_allow_html=True)


def empty_state(icon, title, sub):
    st.markdown(f"""
        <div class="empty-state">
            <div class="icon">{icon}</div>
            <div class="title">{title}</div>
            <div class="sub">{sub}</div>
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# Session state
# ============================================================
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "note" not in st.session_state:
    st.session_state.note = ""


@st.cache_resource
def get_model():
    return load_model()


def submit_feedback(name, issue_type, message):
    payload = {
        FORM_FIELDS["name"]: name,
        FORM_FIELDS["issue_type"]: issue_type,
        FORM_FIELDS["message"]: message,
    }
    try:
        response = requests.post(GOOGLE_FORM_URL, data=payload, timeout=10)
        return response.status_code in (200, 302)
    except requests.RequestException:
        return False


def _find_unicode_font():
    """Try a few common system font paths that render non-ASCII text reliably.
    Falls back to PIL's built-in font if none are found (ASCII only in that case)."""
    candidates = [
        "C:/Windows/Fonts/Nirmala.ttf",   # Windows — supports Bengali, Devanagari, etc.
        "C:/Windows/Fonts/arial.ttf",     # Windows — Latin
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux/Mac fallback
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _wrap_text(text, font, max_width, measure_draw):
    """Word-wrap text so it fits within max_width pixels for the given font."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = (current + " " + word).strip()
        if measure_draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def compose_shareable_image(annotated_rgb, detections, note, author):
    """Bake a caption strip (summary + note, word-wrapped) onto the bottom of the
    annotated image using PIL, so the image and message travel together as one file."""
    base_img = Image.fromarray(annotated_rgb)
    w, h = base_img.size
    note = note.strip()

    font_path = _find_unicode_font()
    font_body = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    font_small = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()

    measure_img = Image.new("RGB", (1, 1))
    measure_draw = ImageDraw.Draw(measure_img)
    note_lines = _wrap_text(note, font_body, w - 32, measure_draw) if note else []

    severe = sum(1 for d in detections if d["severity"] == "Severe")
    moderate = sum(1 for d in detections if d["severity"] == "Moderate")
    minor = sum(1 for d in detections if d["severity"] == "Minor")
    summary = f"{len(detections)} pothole(s) detected  |  Severe: {severe}  Moderate: {moderate}  Minor: {minor}"
    footer = f"Reported by {author}  |  {datetime.now().strftime('%d %b %Y, %H:%M')}"

    line_h = 26
    strip_h = 16 + line_h + (line_h * len(note_lines)) + 30
    strip = Image.new("RGB", (w, strip_h), (30, 32, 36))
    draw = ImageDraw.Draw(strip)

    y = 12
    draw.text((16, y), summary, font=font_body, fill=(255, 199, 44))
    y += line_h
    for line in note_lines:
        draw.text((16, y), f'"{line}"', font=font_body, fill=(245, 245, 243))
        y += line_h
    draw.text((16, strip_h - 24), footer, font=font_small, fill=(154, 160, 166))

    combined = Image.new("RGB", (w, h + strip_h))
    combined.paste(base_img, (0, 0))
    combined.paste(strip, (0, h))

    buf = BytesIO()
    combined.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ============================================================
# SIDEBAR
# ============================================================
inject_css()

with st.sidebar:
    st.markdown("### 🕳️ Pothole Detector")
    st.caption("⚠️ Road condition analysis")
    st.write("")

    if not st.session_state.user_name:
        name_input = st.text_input("Your name", label_visibility="collapsed", placeholder="Enter your name")
        if st.button("Continue →", use_container_width=True) and name_input.strip():
            st.session_state.user_name = name_input.strip()
            st.rerun()
    else:
        initials = "".join([w[0].upper() for w in st.session_state.user_name.split()[:2]])
        st.markdown(f"""
            <div class="side-user">
                <div class="avatar">{initials}</div>
                <div>
                    <div class="name">{st.session_state.user_name}</div>
                    <div class="tag">Signed in</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Switch user", use_container_width=True):
            st.session_state.user_name = ""
            st.rerun()

    st.divider()
    page = st.radio("Navigate", ["Detect Potholes", "Report an Issue", "About"], label_visibility="collapsed")


if not st.session_state.user_name:
    hero("GET STARTED", "Pothole Detection System", "Enter your name in the sidebar to get started.", color="blue")
    st.stop()


# ============================================================
# PAGE: Detect Potholes
# ============================================================
if page == "Detect Potholes":
    hero(
        "🚧 ROAD CONDITION SCAN",
        "Detect Potholes",
        "Upload a road image to detect potholes and estimate severity using a YOLOv8-seg model.",
        color="accent",
    )

    uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
    conf_threshold = st.slider("Detection confidence threshold", 0.05, 0.9, 0.25, 0.05)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner("Scanning image..."):
            model = get_model()
            detections, result = detect_potholes(model, img_bgr, conf=conf_threshold)
            annotated = annotate_image(result)

        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("**Detected**")
            st.image(annotated, use_container_width=True)

        st.write("")

        if not detections:
            empty_state("✅", "No potholes detected", "Try lowering the confidence threshold above.")
        else:
            severe = sum(1 for d in detections if d["severity"] == "Severe")
            moderate = sum(1 for d in detections if d["severity"] == "Moderate")
            minor = sum(1 for d in detections if d["severity"] == "Minor")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            stat_items = [
                (s1, len(detections), "Total", "var(--accent)"),
                (s2, severe, "Severe", "var(--red)"),
                (s3, moderate, "Moderate", "var(--orange)"),
                (s4, minor, "Minor", "var(--green)"),
            ]
            for col, num, label, stat_color in stat_items:
                col.markdown(
                    f'<div class="stat" style="--stat-color:{stat_color};">'
                    f'<div class="num">{num}</div><div class="label">{label}</div></div>',
                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            for i, d in enumerate(detections, start=1):
                color = SEVERITY_COLORS.get(d["severity"], "#FFC72C")
                icon = SEVERITY_ICONS.get(d["severity"], "⚠️")
                pct = d['confidence'] * 100
                st.markdown(f"""
                    <div class="detection-card" style="--sev-color:{color};">
                        <div class="row">
                            <span class="id">POTHOLE #{i}</span>
                            <span class="sev-pill" style="background:{color};">{icon} {d['severity']}</span>
                            <span class="conf">{pct:.0f}%</span>
                        </div>
                        <div class="conf-track"><div class="conf-fill" style="width:{pct}%; --sev-color:{color};"></div></div>
                    </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown("**Add a note & share this result**")
            note = st.text_area(
                "Optional note", key="note", height=80, label_visibility="collapsed",
                placeholder="e.g. Near Park Street junction, needs urgent repair",
            )

            share_col1, share_col2 = st.columns(2)
            with share_col1:
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                success, buf = cv2.imencode(".jpg", annotated_bgr)
                if success:
                    st.download_button("⬇️ Image only (no note)", data=BytesIO(buf.tobytes()),
                                        file_name="pothole_result.jpg", mime="image/jpeg", use_container_width=True)
            with share_col2:
                combined_bytes = compose_shareable_image(annotated, detections, note, st.session_state.user_name)
                if combined_bytes:
                    st.download_button("📤 Image WITH your note", data=BytesIO(combined_bytes),
                                        file_name="pothole_report.jpg", mime="image/jpeg", use_container_width=True)

            if note.strip():
                with st.expander("Preview: how your note will look on the image", expanded=True):
                    preview_arr = np.frombuffer(combined_bytes, dtype=np.uint8)
                    preview_img = cv2.imdecode(preview_arr, cv2.IMREAD_COLOR)
                    st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            with st.expander("Copy as text (for WhatsApp / email)"):
                summary_text = (
                    f"Pothole Report — {datetime.now().strftime('%d %b %Y')}\n"
                    f"Reported by: {st.session_state.user_name}\n"
                    f"Detected: {len(detections)} pothole(s) "
                    f"(Severe: {severe}, Moderate: {moderate}, Minor: {minor})\n"
                    f"Note: {note.strip() if note.strip() else '—'}"
                )
                st.code(summary_text, language=None)
    else:
        empty_state("🚧", "No image uploaded yet", "Upload a road photo above to run detection.")


# ============================================================
# PAGE: Report an Issue
# ============================================================
elif page == "Report an Issue":
    hero("⚠️ FEEDBACK", "Report an Issue", "Faced a bug, wrong detection, or have a feature request? Let us know.", color="red")

    with st.form("feedback_form"):
        issue_type = st.selectbox("Type", ["Wrong detection", "App bug", "Feature request", "Other"])
        message = st.text_area("Describe the issue or request", height=150)
        submitted = st.form_submit_button("Submit")

        if submitted:
            if not message.strip():
                st.warning("Please describe the issue before submitting.")
            else:
                ok = submit_feedback(st.session_state.user_name, issue_type, message.strip())
                if ok:
                    st.success("Thanks! Your feedback has been submitted.")
                else:
                    st.error("Couldn't submit right now — please try again later.")


# ============================================================
# PAGE: About
# ============================================================
elif page == "About":
    hero("🚦 PROJECT INFO", "About This System", "How the detection pipeline works, end to end.", color="blue")
    st.markdown("""
    **Pothole Detection System** uses a YOLOv8 segmentation model to detect potholes
    in road images and estimate their severity based on how much of the frame they cover.

    - **Model:** YOLOv8n-seg, fine-tuned on a public pothole segmentation dataset
    - **Severity scoring:** Minor / Moderate / Severe, based on relative pothole size
    - **Built with:** Ultralytics YOLOv8, OpenCV, Streamlit

    This is a baseline system — accuracy improves with more training data and epochs.
    """)


# ============================================================
# SETUP INSTRUCTIONS — Google Form for the "Report an Issue" page
# ============================================================
# 1. Go to forms.google.com and create a new form with:
#      - Short answer field: "Name"
#      - Short answer field: "Issue Type"
#      - Paragraph field: "Message"
# 2. Click "Responses" tab → click the green Sheets icon to link a Google Sheet.
# 3. Click "Send" → click the link icon → copy the URL, change "viewform" to
#    "formResponse", paste into GOOGLE_FORM_URL above.
# 4. Get entry IDs: open the live form → right-click → "View Page Source" →
#    search "entry." → match each ID to Name/Issue Type/Message in FORM_FIELDS.