"""Campus Cafe Drinks & Snack Recommender Engine — Streamlit app.

Academic Linear Algebra (PCA) mini-project.
All PCA math is performed from scratch with NumPy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data_processing import (
    FEATURES,
    DataError,
    clean_data,
    create_matrix,
    feature_means,
    load_data,
)
from src.pca_analysis import PCAResult, perform_pca, project_new_user
from src.recommender import (
    DRINK_FEATURES,
    FOOD_FEATURES,
    calculate_distances,
    generate_all_combos,
    generate_combo_recommendation,
)
from src.visualization import plot_explained_variance, plot_taste_map

RATING_LABELS = {
    1: "Never Order",
    2: "Rarely Order",
    3: "Sometimes Order",
    4: "Usually Order",
    5: "Must Have",
}

# Navigation groups.
NAV_DISCOVER = ["Home", "Rate Your Preferences", "Your Taste Profile", "Taste Map"]
NAV_RESULTS = ["Similar Students", "Your Combo"]
NAV_HOOD = ["Data & PCA", "How the Mathematics Works"]
ALL_PAGES = NAV_DISCOVER + NAV_RESULTS + NAV_HOOD

# Step mapping for the progress indicator (1-based).
PAGE_STEP = {
    "Home": 1,
    "Rate Your Preferences": 2,
    "Your Taste Profile": 3,
    "Taste Map": 4,
    "Similar Students": 5,
    "Your Combo": 5,
    "Data & PCA": 0,
    "How the Mathematics Works": 0,
}


# ---------- cached data + PCA ----------
@st.cache_data
def load_and_run(path: str):
    """Load data, clean, build matrix, and run PCA. Cached for the session."""
    df = load_data(path)
    observed, imputed = clean_data(df)
    matrix, respondent_names = create_matrix(imputed)
    pca = perform_pca(matrix, FEATURES)
    return df, observed, imputed, matrix, respondent_names, pca


# ---------- page config ----------
st.set_page_config(
    page_title="Campus Cafe Recommender",
    page_icon="☕",
    layout="wide",
)

# ---------- session state init ----------
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "started" not in st.session_state:
    st.session_state.started = False
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {f: 3 for f in FEATURES}
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "user_coords" not in st.session_state:
    st.session_state.user_coords = None
if "neighbors" not in st.session_state:
    st.session_state.neighbors = None
if "combo_list" not in st.session_state:
    st.session_state.combo_list = []
if "combo_cache_key" not in st.session_state:
    st.session_state.combo_cache_key = None
if "combo_index" not in st.session_state:
    st.session_state.combo_index = 0
if "shown_combo_keys" not in st.session_state:
    st.session_state.shown_combo_keys = set()


def _combo_item_key(combo) -> tuple[str | None, str | None]:
    """Identity of a combo, used to track which ones have been shown."""
    return (
        combo.snack.item if combo.snack else None,
        combo.drink.item if combo.drink else None,
    )


def _advance_to_next_combo() -> None:
    """Move to the next distinct combo in the ranked sequence.

    Walks forward through the pre-ranked combo list (best -> next best ->
    ...), skipping any combo already shown this session. Only once every
    distinct combo has been shown does it allow a repeat, by moving to the
    next one in the ranked order.
    """
    combo_list = st.session_state.combo_list
    n = len(combo_list)
    if n <= 1:
        return
    shown = st.session_state.shown_combo_keys
    start = st.session_state.combo_index
    for offset in range(1, n + 1):
        idx = (start + offset) % n
        if _combo_item_key(combo_list[idx]) not in shown:
            st.session_state.combo_index = idx
            return
    # Every distinct combo has already been shown this session -> cycle on.
    st.session_state.combo_index = (start + 1) % n


# Inject custom CSS for a warm, modern campus-cafe look.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .main .block-container {
        padding-top: 1.5rem; max-width: 1100px;
    }

    /* ---- Text color reset for cream cards (root cause fix) ---- */
    /* Streamlit's dark theme injects white text; these cards are cream,
       so we force dark text on the card and every descendant. */
    .cafe-card, .combo-card, .match-card {
        color: #2d2418 !important;
    }
    .cafe-card *, .combo-card *, .match-card * {
        color: inherit;
    }
    .cafe-card p, .cafe-card li, .cafe-card span, .cafe-card div,
    .cafe-card td, .cafe-card th, .cafe-card b, .cafe-card strong {
        color: #2d2418;
    }
    .combo-card p, .combo-card li, .combo-card span, .combo-card div,
    .combo-card b, .combo-card strong {
        color: #2d2418;
    }
    .match-card p, .match-card li, .match-card span, .match-card div,
    .match-card b, .match-card strong {
        color: #2d2418;
    }

    .cafe-card {
        background: #fffaf3; border: 1px solid #f0e6d6; border-radius: 16px;
        padding: 1.2rem 1.4rem; box-shadow: 0 2px 10px rgba(120,80,30,0.07);
        transition: transform .15s ease, box-shadow .15s ease;
        color: #2d2418;
    }
    .cafe-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(120,80,30,0.14);
    }
    .combo-card {
        background: linear-gradient(135deg,#fff7ec,#fdf0e0);
        border: 1px solid #ecd9bc; border-radius: 20px;
        padding: 2rem 1.5rem; text-align:center;
        box-shadow: 0 4px 16px rgba(150,100,40,0.12);
        transition: transform .15s ease;
        color: #2d2418;
    }
    .combo-card:hover { transform: translateY(-4px); }
    .section-pill {
        display:inline-block; background:#f4e9d8; color:#7a4a1b;
        font-weight:600; font-size:.8rem; padding:.3rem .8rem;
        border-radius:999px; margin-bottom:.6rem;
    }
    .step-circle {
        display:inline-flex; align-items:center; justify-content:center;
        width:34px; height:34px; border-radius:50%;
        background:#d97706; color:#ffffff; font-weight:700; font-size:1rem;
    }
    .step-circle.done { background:#16a34a; }
    .step-circle.current { background:#d97706; box-shadow:0 0 0 4px #fde68a; }
    .step-line {
        flex:1; height:3px; background:#e5d6c0; margin:0 .3rem; border-radius:2px;
    }
    .step-line.done { background:#16a34a; }
    .match-card {
        background:#fffaf3; border:1px solid #f0e6d6; border-radius:14px;
        padding:1rem 1.2rem; margin-bottom:.7rem;
        box-shadow:0 2px 8px rgba(120,80,30,0.06);
        color: #2d2418;
    }

    /* ---- Ensure Streamlit native markdown headings/paragraphs are readable
           on the dark app background (not inside cream cards). ---- */
    .main h1, .main h2, .main h3, .main h4, .main h5 {
        color: #f5efe6;
    }
    .main p, .main li {
        color: #e8e0d4;
    }
    /* Captions and helper text — keep them visible on dark bg */
    .main [data-testid="stCaptionContainer"] {
        color: #c9bfae !important;
    }
    /* Sidebar text */
    .sidebar .sidebar-content, [data-testid="stSidebar"] {
        color: #e8e0d4;
    }
    [data-testid="stSidebar"] * {
        color: inherit;
    }
    /* Dataframe / tables inside expanders — Streamlit renders these on a
       light surface already; ensure cell text is dark. */
    .stDataFrame, .stTable {
        color: #1a1a1a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_progress(current_page: str) -> None:
    """Render a 5-step progress indicator for the main recommendation flow."""
    if current_page not in PAGE_STEP or PAGE_STEP[current_page] == 0:
        return
    current_step = PAGE_STEP[current_page]
    labels = ["Welcome", "Rate", "Profile", "Map", "Combo"]
    parts = []
    parts.append("<div style='display:flex; align-items:center; margin:1rem 0 1.5rem;'>")
    for i, label in enumerate(labels, 1):
        cls = "step-circle current" if i == current_step else (
            "step-circle done" if i < current_step else "step-circle"
        )
        parts.append(
            f"<div style='text-align:center; min-width:60px;'>"
            f"<div class='{cls}'>{i}</div>"
            f"<div style='font-size:.7rem; color:#d4b896; margin-top:.25rem;'>{label}</div>"
            f"</div>"
        )
        if i < len(labels):
            line_cls = "step-line done" if i < current_step else "step-line"
            parts.append(f"<div class='{line_cls}'></div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def cafe_card(body_md: str) -> None:
    st.markdown(f"<div class='cafe-card'>{body_md}</div>", unsafe_allow_html=True)


# ============================================================
# START / NAME SCREEN
# ============================================================
if not st.session_state.started:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center; font-size:2.6rem;'>"
        "☕ Campus Cafe Recommender</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align:center; color:#d4b896; font-weight:500;'">
        "Discover your campus taste profile.</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#c9bfae; max-width:620px; margin:0 auto 1.5rem;'">
        "Rate your favourite campus cafe items, discover your taste profile using PCA, "
        "and find a snack + drink combination based on students with similar preferences."
        "</p>",
        unsafe_allow_html=True,
    )

    # Visual process flow.
    flow = "① Rate Your Preferences &nbsp;→&nbsp; ② Discover Your Taste Profile &nbsp;→&nbsp; ③ Find Your Taste Matches &nbsp;→&nbsp; ④ Get Your Campus Combo"
    st.markdown(
        f"<p style='text-align:center; color:#d4b896; font-weight:500;'>{flow}</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        name = st.text_input("What should we call you?", value="",
                             placeholder="Type your name here...",
                             label_visibility="visible")
        if st.button("Start My Recommendation →", type="primary", use_container_width=True):
            if name.strip():
                st.session_state.user_name = name.strip()
                st.session_state.started = True
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.stop()


# ---------- data loading (only after name entered) ----------
uploaded = st.sidebar.file_uploader("Upload survey CSV (optional)", type=["csv"])
data_path = "data/survey_data.csv"
if uploaded is not None:
    data_path = "data/_uploaded.csv"
    with open(data_path, "wb") as f:
        f.write(uploaded.getvalue())

try:
    df, observed_df, imputed_df, matrix, respondent_names, pca = load_and_run(data_path)
except DataError as e:
    st.error(str(e))
    st.stop()


# ---------- sidebar navigation ----------
st.sidebar.markdown("### ☕ CAMPUS CAFE")
st.sidebar.markdown("**Taste Lab**")
st.sidebar.caption("Powered by PCA")
if st.session_state.user_name:
    st.sidebar.markdown(f"<small>Welcome, **{st.session_state.user_name}**!</small>",
                        unsafe_allow_html=True)
st.sidebar.markdown("---")


def _nav_radio(label: str, options: list[str], key: str) -> str | None:
    """Render a grouped radio and return the selected page, or None."""
    choice = st.sidebar.radio(label, options, key=key, label_visibility="collapsed")
    return choice if choice else None


selected = st.session_state.get("_selected_page", "Home")

st.sidebar.markdown("**DISCOVER**")
discover_choice = st.sidebar.radio(
    "DISCOVER", ["🏠 Home", "⭐ Rate Preferences", "🧬 Your Taste Profile", "🗺️ Taste Map"],
    key="nav_discover", label_visibility="collapsed"
)
st.sidebar.markdown("**YOUR RESULTS**")
results_choice = st.sidebar.radio(
    "YOUR RESULTS", ["👥 Similar Students", "🍴 Your Combo"],
    key="nav_results", label_visibility="collapsed"
)
st.sidebar.markdown("**UNDER THE HOOD**")
hood_choice = st.sidebar.radio(
    "UNDER THE HOOD", ["📊 Data & PCA", "📐 How the Mathematics Works"],
    key="nav_hood", label_visibility="collapsed"
)

# Map sidebar labels back to canonical page names.
_LABEL_TO_PAGE = {
    "🏠 Home": "Home",
    "⭐ Rate Preferences": "Rate Your Preferences",
    "🧬 Your Taste Profile": "Your Taste Profile",
    "🗺️ Taste Map": "Taste Map",
    "👥 Similar Students": "Similar Students",
    "🍴 Your Combo": "Your Combo",
    "📊 Data & PCA": "Data & PCA",
    "📐 How the Mathematics Works": "How the Mathematics Works",
}

# Determine which group was most recently interacted with via session state.
last_group = st.session_state.get("_last_nav_group", "discover")
d_val = st.session_state.get("nav_discover")
r_val = st.session_state.get("nav_results")
h_val = st.session_state.get("nav_hood")

# Heuristic: the group whose value differs from the previous render wins.
if d_val != st.session_state.get("_prev_d"):
    last_group = "discover"
elif r_val != st.session_state.get("_prev_r"):
    last_group = "results"
elif h_val != st.session_state.get("_prev_h"):
    last_group = "hood"
st.session_state["_last_nav_group"] = last_group
st.session_state["_prev_d"] = d_val
st.session_state["_prev_r"] = r_val
st.session_state["_prev_h"] = h_val

if last_group == "discover":
    nav = _LABEL_TO_PAGE[d_val]
elif last_group == "results":
    nav = _LABEL_TO_PAGE[r_val]
else:
    nav = _LABEL_TO_PAGE[h_val]

# Progress indicator on flow pages.
render_progress(nav)


# ============================================================
# HOME
# ============================================================
if nav == "Home":
    st.markdown("<h1 style='font-size:2.2rem;'>☕ Campus Cafe Recommender</h1>",
                unsafe_allow_html=True)
    st.markdown(f"### Welcome, {st.session_state.user_name}! 👋")

    st.markdown(
        "<div class='cafe-card'>"
        "<p>This project learns <b>patterns in campus cafe preferences</b> using student "
        "ratings and Linear Algebra. It then recommends <b>one snack + one drink</b> "
        "tailored to your taste.</p>"
        "<p><b>Principal Component Analysis (PCA)</b> takes your 10 item ratings and "
        "reduces them into a simple <b>2D taste profile</b>. We then find the existing "
        "students whose taste profiles are closest to yours and see what they love.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Your journey")
    steps = [
        ("①", "Rate Your Preferences", "Tell us how much you'd order each cafe item."),
        ("②", "Discover Your Taste Profile", "PCA maps your preferences into 2D."),
        ("③", "Find Your Taste Matches", "See the students most similar to you."),
        ("④", "Get Your Campus Combo", "Receive one snack + one drink recommendation."),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(
            f"<div class='cafe-card' style='text-align:center;'>"
            f"<div style='font-size:1.6rem;'>{num}</div>"
            f"<div style='font-weight:600; color:#8a5a2b; margin:.4rem 0;'>{title}</div>"
            f"<div style='font-size:.82rem; color:#5a4a35;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Students surveyed", len(respondent_names))
    c2.metric("Cafe items", len(FEATURES))
    c3.metric("Taste map captures", f"{(pca.pc1_variance + pca.pc2_variance)*100:.1f}% of variation")

    st.markdown("#### The cafe items")
    sc, dc = st.columns(2)
    with sc:
        st.markdown(
            "<div class='cafe-card'><b>🍴 Snacks / Food</b><br>"
            + "".join(f"<br>• {f}" for f in FOOD_FEATURES).lstrip("<br>")
            + "</div>",
            unsafe_allow_html=True,
        )
    with dc:
        st.markdown(
            "<div class='cafe-card'><b>🥤 Drinks</b><br>"
            + "".join(f"<br>• {f}" for f in DRINK_FEATURES).lstrip("<br>")
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👉 Head to **⭐ Rate Preferences** in the sidebar to begin.")


# ============================================================
# RATE YOUR PREFERENCES
# ============================================================
elif nav == "Rate Your Preferences":
    st.markdown("<h1>⭐ Rate Your Preferences</h1>", unsafe_allow_html=True)
    st.markdown(f"Tell us about your taste, {st.session_state.user_name}.")

    # --- Rating guide ---
    st.markdown(
        "<div class='cafe-card'>"
        "<span class='section-pill'>Rating Guide</span>"
        "<p style='margin:0 0 .6rem;'>How much would you order this?</p>"
        "<table style='width:100%; border-collapse:collapse;'>"
        "<tr><th style='text-align:left; padding:.3rem;'>Rating</th>"
        "<th style='text-align:left; padding:.3rem;'>Meaning</th></tr>"
        + "".join(
            f"<tr><td style='padding:.25rem .3rem;'><b>{k}</b></td>"
            f"<td style='padding:.25rem .3rem;'>{v}</td></tr>"
            for k, v in RATING_LABELS.items()
        )
        + "</table></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Snacks ---
    st.markdown("<span class='section-pill'>🍴 Snacks / Food</span>", unsafe_allow_html=True)
    snack_cols = st.columns(len(FOOD_FEATURES))
    for i, item in enumerate(FOOD_FEATURES):
        with snack_cols[i]:
            st.markdown(
                f"<div class='cafe-card' style='text-align:center;'>"
                f"<div style='font-weight:600; color:#8a5a2b; margin-bottom:.5rem;'>{item}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            current = st.session_state.user_ratings[item]
            st.session_state.user_ratings[item] = st.slider(
                item, 1, 5, current, key=f"slider_{item}", label_visibility="collapsed"
            )
            label = RATING_LABELS[st.session_state.user_ratings[item]]
            st.markdown(
                f"<p style='text-align:center; font-size:.78rem; color:#16a34a; "
                f"font-weight:600;'>{label}</p>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    # --- Drinks ---
    st.markdown("<span class='section-pill'>🥤 Drinks</span>", unsafe_allow_html=True)
    drink_cols = st.columns(len(DRINK_FEATURES))
    for i, item in enumerate(DRINK_FEATURES):
        with drink_cols[i]:
            st.markdown(
                f"<div class='cafe-card' style='text-align:center;'>"
                f"<div style='font-weight:600; color:#8a5a2b; margin-bottom:.5rem;'>{item}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            current = st.session_state.user_ratings[item]
            st.session_state.user_ratings[item] = st.slider(
                item, 1, 5, current, key=f"slider_{item}", label_visibility="collapsed"
            )
            label = RATING_LABELS[st.session_state.user_ratings[item]]
            st.markdown(
                f"<p style='text-align:center; font-size:.78rem; color:#16a34a; "
                f"font-weight:600;'>{label}</p>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✓ Submit My Ratings", type="primary", use_container_width=False):
        st.session_state.submitted = True
        ratings = np.array([st.session_state.user_ratings[f] for f in FEATURES], dtype=float)
        with st.spinner("Mapping your taste profile with PCA…"):
            coords = project_new_user(ratings, pca)
            st.session_state.user_coords = coords
            neighbors = calculate_distances(coords, pca.student_coords, respondent_names)
            st.session_state.neighbors = neighbors[:5]
        st.success(
            f"Thanks, {st.session_state.user_name}! Your taste profile is ready. "
            "Visit **🧬 Your Taste Profile**, **🗺️ Taste Map**, **👥 Similar Students**, "
            "and **🍴 Your Combo** to see your results."
        )


# ============================================================
# YOUR TASTE PROFILE
# ============================================================
elif nav == "Your Taste Profile":
    st.markdown("<h1>🧬 Your Taste Profile</h1>", unsafe_allow_html=True)
    if st.session_state.user_coords is None:
        st.info("Please submit your ratings on **⭐ Rate Preferences** first.")
        st.stop()

    coords = st.session_state.user_coords
    ratings = np.array([st.session_state.user_ratings[f] for f in FEATURES], dtype=float)

    st.markdown(f"Hey, {st.session_state.user_name}! Your cafe preferences have been mapped.")
    st.markdown(
        "<div class='cafe-card'>"
        "PCA reduces your 10-dimensional preference profile into two main taste "
        "dimensions so we can visualize how your preferences compare with other students."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    total_var = (pca.pc1_variance + pca.pc2_variance) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("PC1", f"{coords[0]:.3f}", f"{pca.pc1_variance*100:.1f}% variance")
    c2.metric("PC2", f"{coords[1]:.3f}", f"{pca.pc2_variance*100:.1f}% variance")
    c3.metric("Combined", f"{total_var:.1f}%", "PC1 + PC2")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### What you rated")
    display = pd.DataFrame(
        {"Item": FEATURES, "Your rating": ratings.astype(int),
         "Rating meaning": [RATING_LABELS[int(r)] for r in ratings]}
    )
    st.dataframe(display, hide_index=True, use_container_width=True)


# ============================================================
# 2D TASTE MAP
# ============================================================
elif nav == "Taste Map":
    st.markdown("<h1>🗺️ Taste Map</h1>", unsafe_allow_html=True)
    coords = st.session_state.user_coords
    neighbor_ids: list[str] | None = None

    if coords is not None and st.session_state.neighbors is not None:
        neighbor_ids = [n.name for n in st.session_state.neighbors]

    fig = plot_taste_map(pca, respondent_names, new_user_coords=coords, neighbor_ids=neighbor_ids)
    st.plotly_chart(fig, use_container_width=True)

    if coords is None:
        st.info(f"{st.session_state.user_name}, submit your ratings to see yourself plotted on the map.")
    else:
        st.success(
            f"{st.session_state.user_name}, you are the red star. "
            "Your 5 nearest students are highlighted in amber."
        )

    st.markdown(
        "<div class='cafe-card'>"
        "Each point represents a student's preference profile after projection onto "
        "PC1 and PC2. The axes divide the map into four quadrants."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(plot_explained_variance(pca), use_container_width=True)


# ============================================================
# SIMILAR STUDENTS
# ============================================================
elif nav == "Similar Students":
    st.markdown("<h1>👥 Your Taste Matches</h1>", unsafe_allow_html=True)
    coords = st.session_state.user_coords
    if coords is None:
        st.info(f"{st.session_state.user_name}, submit your ratings first.")
        st.stop()

    neighbors = calculate_distances(coords, pca.student_coords, respondent_names)
    st.markdown(
        f"{st.session_state.user_name}, here are the students whose PCA taste profiles "
        "are closest to yours, measured by Euclidean distance in (PC1, PC2) space."
    )

    k = st.slider("Number of taste matches (K)", 3, 7, 5)
    st.markdown("<br>", unsafe_allow_html=True)

    top = neighbors[:k]
    for n in top:
        st.markdown(
            f"<div class='match-card'>"
            f"<span style='font-weight:600; font-size:1.05rem; color:#8a5a2b;'>{n.name}</span>"
            f"<span style='float:right; color:#5a4a35; font-size:.9rem;'>"
            f"🧭 Taste distance: <b>{n.distance:.3f}</b></span><br>"
            f"<span style='font-size:.8rem; color:#5a4a35;'>PC1: {n.pc1:.2f} &nbsp; PC2: {n.pc2:.2f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.session_state.neighbors = top


# ============================================================
# YOUR COMBO  (exactly ONE snack + ONE drink)
# ============================================================
elif nav == "Your Combo":
    st.markdown("<h1>☕ Your Campus Combo</h1>", unsafe_allow_html=True)
    coords = st.session_state.user_coords
    neighbors = st.session_state.neighbors
    if coords is None or neighbors is None:
        st.info(f"{st.session_state.user_name}, submit your ratings and visit **👥 Similar Students** first.")
        st.stop()

    ratings = np.array([st.session_state.user_ratings[f] for f in FEATURES], dtype=float)

    # Rank every possible combo once per (ratings, neighbor set); this is
    # what "Next Combo" steps through, so it only needs recomputing when the
    # user's ratings or nearest neighbors actually change.
    cache_key = (
        tuple(ratings.tolist()),
        tuple(n.name for n in neighbors),
    )
    if st.session_state.combo_cache_key != cache_key:
        with st.spinner("Finding your perfect combo…"):
            st.session_state.combo_list = generate_all_combos(
                ratings, observed_df, neighbors, FEATURES, min_rating=4
            )
        st.session_state.combo_cache_key = cache_key
        st.session_state.combo_index = 0
        st.session_state.shown_combo_keys = set()

    combo_list = st.session_state.combo_list
    if not combo_list:
        st.info("No combo could be generated from the current ratings and neighbor data.")
        st.stop()
    combo = combo_list[st.session_state.combo_index]
    st.session_state.shown_combo_keys.add(_combo_item_key(combo))

    st.markdown(f"Here is your personalized recommendation, {st.session_state.user_name}.")

    # --- Combo cards ---
    left, right = st.columns(2)
    with left:
        if combo.snack:
            st.markdown(
                f"<div class='combo-card'>"
                f"<div style='font-size:.9rem; color:#7a4a1b; font-weight:600;'>🍴 YOUR SNACK</div>"
                f"<div style='font-size:1.8rem; font-weight:700; color:#8a5a2b; margin:.5rem 0;'>"
                f"{combo.snack.item}</div>"
                f"<div style='font-size:.85rem; color:#5a4a35;'>"
                f"Neighbor average: <b>{combo.snack.score:.2f} / 5</b></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='combo-card'>No snack candidate found.</div>",
                        unsafe_allow_html=True)
    with right:
        if combo.drink:
            st.markdown(
                f"<div class='combo-card'>"
                f"<div style='font-size:.9rem; color:#7a4a1b; font-weight:600;'>🥤 YOUR DRINK</div>"
                f"<div style='font-size:1.8rem; font-weight:700; color:#8a5a2b; margin:.5rem 0;'>"
                f"{combo.drink.item}</div>"
                f"<div style='font-size:.85rem; color:#5a4a35;'>"
                f"Neighbor average: <b>{combo.drink.score:.2f} / 5</b></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='combo-card'>No drink candidate found.</div>",
                        unsafe_allow_html=True)

    # Small celebratory note (only the first time a combo is shown).
    if len(st.session_state.shown_combo_keys) == 1:
        st.balloons()

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Next Combo control ---
    nav_left, nav_right = st.columns([3, 1])
    with nav_right:
        st.button(
            "Next Combo →",
            key="next_combo_btn",
            type="primary",
            use_container_width=True,
            on_click=_advance_to_next_combo,
            disabled=len(combo_list) <= 1,
        )
    with nav_left:
        total = len(combo_list)
        shown = min(len(st.session_state.shown_combo_keys), total)
        if total > 1:
            st.caption(f"Combo {shown} of {total} distinct pairings you can explore.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Why this combo?")
    st.markdown(
        f"<div class='cafe-card'>"
        f"{combo.neighbor_explanation} Their ratings were used to identify a matching "
        f"snack and drink."
        f"</div>",
        unsafe_allow_html=True,
    )

    snack_name = combo.snack.item if combo.snack else "—"
    drink_name = combo.drink.item if combo.drink else "—"
    st.markdown(
        f"These similar students rated **{snack_name}** highly among the available "
        f"snack options and **{drink_name}** highly among the available drink options."
    )

    if combo.snack and combo.snack.supporting_students:
        st.caption(f"🍴 Snack high-rated by: {', '.join(combo.snack.supporting_students)}")
    if combo.drink and combo.drink.supporting_students:
        st.caption(f"🥤 Drink high-rated by: {', '.join(combo.drink.supporting_students)}")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("See detailed recommendation reasoning"):
        if combo.snack:
            st.markdown(f"**🍴 {combo.snack.item}** — {combo.snack.reason}")
        if combo.drink:
            st.markdown(f"**🥤 {combo.drink.item}** — {combo.drink.reason}")


# ============================================================
# DATA & PCA
# ============================================================
elif nav == "Data & PCA":
    st.markdown("<h1>📊 Data & PCA Insights</h1>", unsafe_allow_html=True)
    st.markdown("The technical reference section — useful for your viva presentation.")

    st.markdown("#### Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Respondents", len(respondent_names))
    c2.metric("Features", len(FEATURES))
    c3.metric("PC1 + PC2 variance", f"{(pca.pc1_variance + pca.pc2_variance)*100:.1f}%")

    with st.expander("Raw survey data (observed ratings)", expanded=False):
        st.dataframe(observed_df, hide_index=True, use_container_width=True)

    with st.expander("Feature means (observed vs. imputation)", expanded=False):
        means_df = pd.DataFrame({
            "Feature": FEATURES,
            "Observed mean": [f"{v:.2f}" for v in feature_means(observed_df)],
            "Imputation mean used for PCA": [f"{v:.2f}" for v in pca.means],
        })
        st.dataframe(means_df, hide_index=True, use_container_width=True)

    with st.expander("Covariance matrix", expanded=False):
        cov_df = pd.DataFrame(pca.covariance, index=FEATURES, columns=FEATURES)
        st.dataframe(cov_df.style.format("{:.3f}"), use_container_width=True)

    with st.expander("Eigenvalues & explained variance (descending)", expanded=False):
        eig_df = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(pca.eigenvalues))],
            "Eigenvalue": [f"{v:.4f}" for v in pca.eigenvalues],
            "Explained variance %": [f"{v*100:.2f}%" for v in pca.explained_variance_ratio],
        })
        st.dataframe(eig_df, hide_index=True, use_container_width=True)

    with st.expander("Eigenvectors (full matrix)", expanded=False):
        vec_df = pd.DataFrame(
            pca.eigenvectors,
            index=FEATURES,
            columns=[f"PC{i+1}" for i in range(len(pca.eigenvalues))],
        )
        st.dataframe(vec_df.style.format("{:.4f}"), use_container_width=True)

    with st.expander("PC1 & PC2 loadings (feature contributions)", expanded=False):
        load_df = pd.DataFrame({
            "Feature": FEATURES,
            "PC1 loading": pca.pc1,
            "PC2 loading": pca.pc2,
        })
        st.dataframe(load_df.style.format({"PC1 loading": "{:.4f}", "PC2 loading": "{:.4f}"}),
                      hide_index=True, use_container_width=True)


# ============================================================
# HOW THE MATHEMATICS WORKS
# ============================================================
elif nav == "How the Mathematics Works":
    st.markdown("<h1>📐 How the Mathematics Works</h1>", unsafe_allow_html=True)
    st.markdown("This section explains every step so you can demonstrate it in a viva.")

    # Visual pipeline.
    pipeline = [
        "Survey Data", "Mean Centering", "Covariance Matrix", "Eigendecomposition",
        "PC1 + PC2", "2D Taste Map", "Nearest Neighbors", "Snack + Drink Recommendation",
    ]
    chain = "  →  ".join(pipeline)
    st.markdown(
        f"<div class='cafe-card' style='text-align:center; font-weight:600; "
        f"color:#8a5a2b;'>{chain}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    sections = [
        ("1. The rating matrix",
         "We collect ratings into an *n × 10* matrix **X** where each row is a student "
         "and each column is a cafe item. Entries are 1–5 (or imputed feature means for blanks)."),
        ("2. Mean",
         "For each item (column) we compute the average rating across all students who rated it. "
         "This tells us the campus-wide popularity of each item."),
        ("3. Mean-centering",
         "We subtract each column's mean from every entry: **X_centered = X − mean(X)**. "
         "This removes the 'overall popularity' effect so PCA finds *variation in taste*, "
         "not just 'everything is popular'."),
        ("4. Covariance matrix",
         "The covariance matrix **C** (10×10) measures how pairs of items co-vary. "
         "A positive covariance between *Vada Pav* and *Samosa Pav* means students who "
         "like one tend to like the other. Computed as (1/(n−1)) · X_centeredᵀ · X_centered."),
        ("5. Eigendecomposition",
         "We decompose **C = QΛQᵀ**. Each eigenvector (column of Q) is a *direction* in "
         "taste-space along which preferences vary together; each eigenvalue (diagonal of Λ) "
         "measures how much variance lies along that direction. We use np.linalg.eigh because "
         "C is real and symmetric."),
        ("6. Sorting",
         "Eigenvalues are sorted **descending** and eigenvectors reordered to match. "
         "The first eigenvector (largest eigenvalue) captures the most variation — this is PC1."),
        ("7. PC1 & PC2",
         "PC1 and PC2 are the two eigenvectors with the largest eigenvalues. "
         "They form a 2D plane that best preserves the taste differences between students."),
        ("8. Explained variance",
         "explained_variance_ratio_i = eigenvalue_i / Σ(eigenvalues). "
         "It tells us what fraction of total taste-variation is captured by each component."),
        ("9. Projection",
         "Each student's centered 10D vector is projected onto PC1 and PC2: "
         "**coords = X_centered · [PC1  PC2]**. The result is a 2D point (PC1 score, PC2 score) "
         "for each student — this is what we plot."),
        ("10. New-user projection",
         "A new user rates the same 10 items. We center their vector with the **original** "
         "training means and project with the **original** PC1/PC2. The model is never retrained — "
         "the new user simply drops into the existing taste space."),
        ("11. Nearest neighbors",
         "Euclidean distance in (PC1, PC2) space between the new user and every existing student. "
         "The K closest students are the user's taste neighbors."),
        ("12. Recommendations (one snack + one drink)",
         "We examine the K nearest neighbors' observed ratings **separately** for the five snack "
         "items and the five drink items. The best snack candidate and the best drink candidate "
         "(by neighbor average, preferring items you rated low or haven't tried) are combined "
         "into a single combo — never two snacks or two drinks."),
    ]
    for title, body in sections:
        st.markdown(f"##### {title}")
        st.markdown(body)
