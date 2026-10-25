"""RedSpell: grammar and spelling correction. Run from the repo root:  streamlit run app/app.py"""
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.infer import Corrector, normalize, spell_flags          # noqa: E402
from src.utils.diff import apply_choices, diff_ops, render_html   # noqa: E402

st.set_page_config(page_title="RedSpell", page_icon="✍️", layout="wide")
st.markdown("""
<style>
.rs-err { text-decoration: underline wavy #e5484d; text-underline-offset: 3px; background: rgba(229,72,77,.12); border-radius: 3px; }
.rs-gap { display: inline-block; width: 8px; border-bottom: 2px solid #e5484d; }
.rs-fix { background: rgba(48,164,108,.22); border-radius: 3px; padding: 0 1px; }
.rs-box { border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: 12px 14px; line-height: 1.7; min-height: 120px; }
</style>""", unsafe_allow_html=True)

st.title("RedSpell")
st.caption("Grammar and spelling correction. Errors are underlined in red, suggestions highlighted in green.")

model_dirs = sorted(p.name for p in (ROOT / "models").glob("*") if (p / "config.json").exists())
if not model_dirs:
    st.error("No model found. Put a fine-tuned model folder (for example `models/t5-base`) in the `models/` directory.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", model_dirs, index=model_dirs.index("t5-base") if "t5-base" in model_dirs else 0)
    beams = st.slider("Beam size", 1, 4, 4, help="Experimental. On the test CPU it was slower and less accurate, so leave it off.")
    quant = st.checkbox("Faster on CPU (int8)", value=False, help="Quantized model: quicker, may reduce accuracy slightly.")


@st.cache_resource(show_spinner="Loading model...")
def get_corrector(name: str, quantize: bool) -> Corrector:
    return Corrector(ROOT / "models" / name, quantize=quantize)


EXAMPLE = "Yesterday I go to the shop and buy a apple. She don't like the weather, but we was happy to be their."
text = st.text_area("Paste a paragraph", height=170, placeholder=EXAMPLE)

if st.button("Correct", type="primary") and text.strip():
    src = normalize(text)
    corr = get_corrector(model_name, quant)
    t0 = time.time()
    hyp = corr.correct_text(src, num_beams=beams)
    st.session_state["res"] = {"src": src, "hyp": hyp, "secs": time.time() - t0, "flags": spell_flags(src, hyp),
                               "id": st.session_state.get("res", {}).get("id", 0) + 1}

res = st.session_state.get("res")
if res:
    ops = diff_ops(res["src"], res["hyp"])
    changes = [i for i, o in enumerate(ops) if o["changed"]]
    st.write(f"**{len(changes)} correction(s)** in {res['secs']:.1f} s")
    left, right = render_html(ops)
    c1, c2 = st.columns(2)
    c1.markdown("**Original**"); c1.markdown(f'<div class="rs-box">{left}</div>', unsafe_allow_html=True)
    c2.markdown("**Corrected**"); c2.markdown(f'<div class="rs-box">{right}</div>', unsafe_allow_html=True)

    accepted = {}
    if changes:
        st.subheader("Corrections")
        for i in changes:
            a, b = ops[i]["a"].strip() or "∅", ops[i]["b"].strip() or "(delete)"
            accepted[i] = st.checkbox(f"{a}  →  {b}", value=True, key=f"acc_{res['id']}_{i}")
    picks = {}
    if res["flags"]:
        st.subheader("Possible misspellings")
        st.caption("Words the model left unchanged that are not in the dictionary. Suggestions are not applied unless you pick one.")
        for f in res["flags"]:
            pick = st.selectbox(f["word"], ["keep"] + f["suggestions"], key=f"sp_{res['id']}_{f['word']}")
            if pick != "keep":
                picks[f["word"]] = pick
    final = apply_choices(ops, accepted)
    for w, r in picks.items():
        final = re.sub(rf"\b{re.escape(w)}\b", r, final)
    st.subheader("Final text")
    st.code(final, language=None, wrap_lines=True)