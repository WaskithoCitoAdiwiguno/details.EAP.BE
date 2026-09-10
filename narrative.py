"""
details.EAP/backend/narrative.py

Generates an HR narrative from Groq using the same risk-tier logic as the
notebook, but with the active Groq key (default or UI-supplied).
"""

from __future__ import annotations

from typing import Any

from groq import Groq, AuthenticationError, APIError


def _risk_tier(proba: float) -> str:
    if proba < 0.30:
        return "RENDAH"
    if proba < 0.60:
        return "SEDANG"
    return "TINGGI"


def _build_prompt(probability: float, top_factors: list[dict[str, Any]], employee_info: dict[str, Any]) -> str:
    proba = float(probability)
    tier = _risk_tier(proba)

    if tier == "RENDAH":
        instruction = (
            "Karena risiko RENDAH (kepuasan tinggi), jelaskan: "
            "(a) mengapa nilai resign ini bisa rendah berdasarkan data yang ada, dan "
            "(b) hal-hal apa yang harus DIPERTAHANKAN agar kondisi ini tetap terjaga."
        )
    elif tier == "SEDANG":
        instruction = (
            "Karena risiko SEDANG, jelaskan faktor yang mulai menunjukkan tanda ketidakpuasan dan "
            "langkah pencegahan dini yang bisa diambil sebelum risiko meningkat."
        )
    else:
        instruction = (
            "Karena risiko TINGGI (kepuasan rendah), jelaskan: "
            "(a) mengapa nilai resign ini bisa tinggi berdasarkan data, dan "
            "(b) faktor apa yang paling mempengaruhi serta rekomendasi improvisasi konkret."
        )

    factor_lines = []
    for item in top_factors:
        feature = str(item.get("feature", ""))
        contribution = item.get("contribution", 0)
        direction = str(item.get("direction", ""))
        factor_lines.append(f"- {feature}: kontribusi SHAP {contribution:.3f} ({direction})")
    factors_text = "\n".join(factor_lines)

    employee_lines = []
    for key in ["JobRole", "Department", "Age", "YearsAtCompany", "OverTime"]:
        if key in employee_info:
            employee_lines.append(f"- {key}: {employee_info[key]}")
    employee_text = "\n".join(employee_lines) if employee_lines else "- (informasi karyawan tidak tersedia)"

    return f"""
Kamu adalah HR Business Partner yang berpengalaman menganalisis risiko attrition karyawan.
Klasifikasi risiko ini mengikuti kerangka risk-tiering dari Pavithran & Vadivel (2026, Frontiers in Big Data):
Rendah (<30%), Sedang (30-60%), Tinggi (>60%).

Data karyawan:
{employee_text}

Probabilitas resign: {proba*100:.1f}% -> Kategori risiko: {tier}

Faktor-faktor pendorong utama (dari analisis SHAP):
{factors_text}

Tugas kamu untuk bagian "Ringkasan":
{instruction}

Untuk bagian "Faktor Utama": sebutkan 2-3 faktor paling dominan dan jelaskan maknanya secara bisnis.

Untuk bagian "Rekomendasi": berikan 2-3 rekomendasi tindakan konkret sesuai kategori risiko
(pertahankan untuk Rendah, pantau untuk Sedang, intervensi aktif untuk Tinggi).

Format jawaban dengan heading: Ringkasan, Faktor Utama, Rekomendasi.
""".strip()


def generate_hr_narrative(
    probability: float,
    top_factors: list[dict[str, Any]],
    employee_info: dict[str, Any],
    *,
    api_key: str,
) -> str:
    """Call Groq with the active key and return the HR narrative text."""
    if not api_key:
        return "[Groq API key tidak tersedia. Harap setel kunci API di UI.]"

    client = Groq(api_key=api_key)
    prompt = _build_prompt(probability, top_factors, employee_info)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "Kamu adalah HR Business Partner berpengalaman yang menjawab dalam Bahasa Indonesia.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content or "[Narasi kosong dari Groq]"
