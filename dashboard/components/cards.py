"""
Metric card and callout box components rendered via st.markdown.
All HTML uses explicit foreground colours to work regardless of Streamlit theme.
"""

import streamlit as st

FAU_BLUE   = "#003865"
FAU_ORANGE = "#FF6B35"

# Colour map for callout styles: (bg, text, border)
_STYLE_COLORS = {
    "info":    ("#EFF6FF", "#1e40af", "#BFDBFE"),
    "warn":    ("#FFFBEB", "#92400e", "#FDE68A"),
    "success": ("#F0FDF4", "#166534", "#BBF7D0"),
    "danger":  ("#FEF2F2", "#991b1b", "#FECACA"),
}


def metric_cards_row(items: list) -> None:
    """
    Render a row of metric cards.
    Each item: {label, value, sub (optional), color (optional)}
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        color = item.get("color", FAU_BLUE)
        sub   = item.get("sub", "")
        col.markdown(f"""
<div style="background:#ffffff;border-left:4px solid {color};border-radius:8px;
            padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.08)">
  <div style="font-size:11px;color:#6B7280;font-weight:700;letter-spacing:.6px;
              text-transform:uppercase">{item['label']}</div>
  <div style="font-size:24px;font-weight:800;color:{color};margin:6px 0 4px">{item['value']}</div>
  <div style="font-size:12px;color:#64748b">{sub}</div>
</div>""", unsafe_allow_html=True)


def finding_box(text: str, icon: str = "💡", style: str = "info") -> None:
    """Render a styled finding callout box."""
    bg, text_color, border = _STYLE_COLORS.get(style, _STYLE_COLORS["info"])
    st.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-radius:8px;
            padding:12px 16px;margin:8px 0;font-size:14px;line-height:1.6;color:{text_color}">
  {icon}&nbsp; {text}
</div>""", unsafe_allow_html=True)


def caveat_box(text: str) -> None:
    finding_box(text, icon="⚠️", style="warn")


def section_header(title: str, subtitle: str = "") -> None:
    sub_html = (f'<p style="color:#64748b;font-size:13px;margin:4px 0 0">{subtitle}</p>'
                if subtitle else "")
    st.markdown(f"""
<div style="margin:24px 0 12px;padding-bottom:8px;border-bottom:2px solid #003865">
  <h3 style="color:#003865;margin:0;font-size:17px;font-weight:700">{title}</h3>
  {sub_html}
</div>""", unsafe_allow_html=True)


def significance_note(p_val: float = None, note: str = "") -> None:
    if p_val is not None and p_val < 0.001:
        st.caption(f"★ Welch's t-test: p < 0.001 — statistically significant. {note}")
    elif note:
        st.caption(note)
