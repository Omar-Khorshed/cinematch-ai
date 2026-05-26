import streamlit as st
import pandas as pd
from engine import load_and_prepare_data, bm25_search_func

st.set_page_config(
    page_title="Cinematch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* ── Background: dark + faint poster collage ── */
.stApp {
    background-color: #0d0d0d;
    background-image:
        linear-gradient(rgba(13,13,13,0.88), rgba(13,13,13,0.97)),
        url('https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM0Mw@@._V1_.jpg');
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15,15,15,0.7) !important;
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
    animation: fadeInDown 0.9s ease-out;
}
.hero h1 {
    font-size: 5rem;
    font-weight: 800;
    color: #E50914;
    letter-spacing: -3px;
    margin-bottom: 0.3rem;
    text-shadow: 0 0 50px rgba(229,9,20,0.3);
}
.hero p {
    font-size: 1.2rem;
    color: #999;
    font-weight: 300;
}
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Search Input ── */
div[data-testid="stTextInput"] input {
    background-color: rgba(22,22,22,0.9) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
    font-size: 1.15rem !important;
    padding: 0.7rem 1rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5) !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #E50914 !important;
    box-shadow: 0 0 22px rgba(229,9,20,0.35) !important;
}

/* ── Suggestion chips (below search box) ── */
.suggest-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin: 0.5rem 0 1rem;
    animation: fadeIn 0.35s ease;
}
.suggest-chip {
    background: rgba(229,9,20,0.1);
    border: 1px solid rgba(229,9,20,0.3);
    color: #ddd;
    padding: 0.28rem 0.9rem;
    border-radius: 999px;
    font-size: 0.88rem;
    cursor: pointer;
    transition: background 0.2s, transform 0.15s;
}
.suggest-chip:hover {
    background: rgba(229,9,20,0.3);
    transform: translateY(-2px);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Movie Cards ── */
.movie-card {
    background: rgba(22,22,22,0.75);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1.4rem;
    transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
    box-shadow: 0 6px 24px rgba(0,0,0,0.4);
}
.movie-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 14px 38px rgba(229,9,20,0.16);
    border-color: rgba(229,9,20,0.4);
}
.movie-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.4rem;
    line-height: 1.2;
}
.movie-meta {
    font-size: 0.87rem;
    color: #888;
    margin-bottom: 0.85rem;
}
.movie-badge {
    display: inline-block;
    background: rgba(229,9,20,0.15);
    color: #ff6060;
    padding: 0.13rem 0.5rem;
    border-radius: 5px;
    font-size: 0.76rem;
    font-weight: 600;
    margin-right: 0.35rem;
    border: 1px solid rgba(229,9,20,0.3);
}
.movie-overview {
    font-size: 0.95rem;
    color: #bbb;
    line-height: 1.55;
    margin-bottom: 0.85rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.movie-score {
    float: right;
    background: linear-gradient(135deg, #FFD700, #FFA500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 1.2rem;
}

/* ── Search Button ── */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #E50914, #9b0610);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.05rem;
    transition: all 0.22s ease;
    box-shadow: 0 4px 16px rgba(229,9,20,0.4);
}
div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #f40612, #b0060f);
    transform: scale(1.03);
    box-shadow: 0 6px 24px rgba(229,9,20,0.55);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
try:
    movies_df, models = load_and_prepare_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

ALL_TITLES = movies_df['title'].dropna().tolist()

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "trigger_search" not in st.session_state:
    st.session_state.trigger_search = False

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color:white;text-align:center;margin-bottom:1.5rem;'>🎛️ Settings</h2>", unsafe_allow_html=True)

st.sidebar.markdown("### ⚖️ BM25 Field Weights")
title_w    = st.sidebar.slider("Title Weight",    0.0, 5.0, 2.0, 0.5)
overview_w = st.sidebar.slider("Overview Weight", 0.0, 5.0, 1.0, 0.5)
cast_w     = st.sidebar.slider("Cast Weight",     0.0, 5.0, 1.5, 0.5)
director_w = st.sidebar.slider("Director Weight", 0.0, 5.0, 1.5, 0.5)

st.sidebar.markdown("### 🎭 Filters")
all_genres = sorted(set(
    g for genres in movies_df['genres_clean'].dropna()
    for g in genres.split()
    if g.strip() and g != 'Unknown'
))
all_genres.insert(0, "All")
selected_genre = st.sidebar.selectbox("Genre", all_genres)
min_rating  = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.5)
max_results = st.sidebar.slider("Max Results",    6,   30,   12,   2)

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>CINE<span style="color:white;">MATCH</span></h1>
    <p>Advanced BM25 Movie Search Engine &nbsp;·&nbsp; 5,000+ Films</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Search bar — Enter triggers search via st.form
# ─────────────────────────────────────────────
_, center, _ = st.columns([1, 6, 1])
with center:
    with st.form(key="search_form", clear_on_submit=False):
        query_input = st.text_input(
            "query",
            value=st.session_state.search_query,
            placeholder="e.g.  space survival  ·  christopher nolan  ·  brad pitt",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Search Movies 🔍", use_container_width=True)

    current_query = query_input.strip()

    # ── Live suggestions (outside form so they don't require submit) ──
    if current_query and len(current_query) >= 2:
        suggestions = [t for t in ALL_TITLES if current_query.lower() in t.lower()][:6]
        if suggestions:
            st.markdown(
                "<div class='suggest-wrap'>"
                + "".join(f"<span class='suggest-chip'>{t}</span>" for t in suggestions)
                + "</div>",
                unsafe_allow_html=True,
            )
            # Real clickable buttons (compact row)
            sug_cols = st.columns(len(suggestions))
            for col, title in zip(sug_cols, suggestions):
                with col:
                    if st.button(title, key=f"sug_{title}", use_container_width=True):
                        st.session_state.search_query = title
                        st.session_state.trigger_search = True
                        st.rerun()

# ─────────────────────────────────────────────
# Determine final query + whether to search
# ─────────────────────────────────────────────
if st.session_state.trigger_search:
    active_query = st.session_state.search_query
    st.session_state.trigger_search = False
    do_search = True
elif submitted:
    active_query = current_query
    st.session_state.search_query = current_query
    do_search = bool(current_query)
else:
    active_query = ""
    do_search = False

# ─────────────────────────────────────────────
# Search & display results
# ─────────────────────────────────────────────
if do_search and active_query:
    with st.spinner("🔍 Ranking with BM25…"):
        results = bm25_search_func(
            active_query, movies_df, models,
            title_w, overview_w, cast_w, director_w,
            top_k=100
        )
        if selected_genre != "All":
            results = results[
                results['genres_clean'].str.contains(selected_genre, case=False, na=False)
            ]
        results = results[results['vote_average'] >= min_rating]
        results = results.head(max_results)

    if len(results) == 0:
        st.markdown("<h3 style='text-align:center;color:white;margin-top:2rem;'>No matches found. Try:</h3>",
                    unsafe_allow_html=True)
        fb_cols = st.columns(4)
        for col, sug in zip(fb_cols, ["alien invasion", "brad pitt action", "romantic comedy", "sci-fi space"]):
            with col:
                if st.button(sug, key=f"fb_{sug}", use_container_width=True):
                    st.session_state.search_query = sug
                    st.session_state.trigger_search = True
                    st.rerun()
    else:
        st.markdown(
            f"<p style='color:#666;text-align:center;margin-bottom:1.5rem;'>"
            f"🎬 <strong style='color:#ddd;'>{len(results)}</strong> results for "
            f"\"<em style='color:#E50914;'>{active_query}</em>\"</p>",
            unsafe_allow_html=True,
        )

        for i in range(0, len(results), 2):
            col_a, col_b = st.columns(2)
            for col, idx in zip([col_a, col_b], [i, i + 1]):
                if idx < len(results):
                    row = results.iloc[idx]
                    genres_html = "".join(
                        f"<span class='movie-badge'>{g}</span>"
                        for g in str(row['genres_clean']).split()[:3]
                    )
                    with col:
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-score">{row['score']:.2f} ⭐</div>
                            <div class="movie-title">{row['title']}</div>
                            <div class="movie-meta">
                                {genres_html}
                                <span style="margin-left:8px;">
                                    📊 {row['vote_average']} &nbsp;|&nbsp; 🔥 {row['popularity']:.1f}
                                </span>
                            </div>
                            <div class="movie-overview">{row['overview']}</div>
                            <div class="movie-meta" style="margin-top:0.75rem;margin-bottom:0;">
                                <strong>🎬</strong> {row['director']}<br>
                                <strong>🎭</strong> {row['main_cast']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
