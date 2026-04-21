"""
MICRA H SENSE V3 — Rule-Based Classification Engine
=====================================================
Implements the exact flowcharts from:
  "Micra H Sense Blood-Based Hormone Algorithm Reference"
  Huawei Innovation Track

Key design decisions:
  - Pure rule-based flowcharts (no neural network)
  - Accepts a time-ordered sequence of readings for recovery checks
  - Insulin confound always checked first; outputs METABOLIC WARNING
    alongside (not instead of) the emotion result
  - Sex-specific mandatory rules enforced per the document
  - Confidence graded: Low-Moderate / Moderate / Moderate-High / High / Very High
"""

import os
import csv
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# NORMAL REFERENCE RANGES  (from Section 1 of the document)
# ─────────────────────────────────────────────────────────────────────────────

NORMAL = {
    'cortisol_nmolL':      (138, 635),
    'epinephrine_pgmL':    (10,  60),
    'norepinephrine_pgmL': (80,  520),
    'serotonin_ngmL':      (50,  200),
    'dopamine_pgmL':       (0,   30),
    'oxytocin_pgmL':       (1,   5),
    'bdnf_ngmL':           (20,  40),
    'acth_pgmL':           (10,  60),
    'il6_pgmL':            (0,   7),
    'gaba_nmolmL':         (0.15, 0.25),
    'vasopressin_pgmL':    (0,   4),
    'insulin_uIUmL':       (2,   20),
}

# Sex-specific normal ranges
NORMAL_SEX = {
    'prolactin_ngmL':    {'M': (2, 18),     'F': (2, 29)},
    'testosterone_ngdL': {'M': (300, 1000), 'F': (15, 70)},
    'estradiol_pgmL':    {'M': (10, 40),    'F': (20, 150)},
    'progesterone_ngmL': {'M': (0.1, 0.3),  'F': (0.1, 0.7)},
    'leptin_ngmL':       {'M': (1, 10),     'F': (3, 25)},
}


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    emotion:            str
    confidence:         str
    markers_matched:    list = field(default_factory=list)
    warnings:           list = field(default_factory=list)
    notes:              list = field(default_factory=list)
    metabolic_confound: bool = False
    reclassified_from:  Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _val(reading: dict, key: str) -> Optional[float]:
    """Safely extract a float from a reading dict. Returns None if missing."""
    v = reading.get(key)
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _sex(reading: dict) -> str:
    """Return 'M' or 'F' from a reading (defaults to 'M' if missing)."""
    return str(reading.get('sex', 'M')).strip().upper()


def _conf_grade(n: int, scale: list) -> str:
    """Map a confirmatory hit count to a confidence label using a provided scale list."""
    idx = min(n, len(scale) - 1)
    return scale[idx]


def _in_normal(key: str, val: float, s: str) -> bool:
    """Return True if val is within the normal range for this key and sex."""
    if key in NORMAL:
        lo, hi = NORMAL[key]
        return lo <= val <= hi
    if key in NORMAL_SEX:
        lo, hi = NORMAL_SEX[key][s]
        return lo <= val <= hi
    return True  # unknown key → assume normal


def _hormones_recovered(readings: list, keys: list, within_minutes: float) -> Optional[bool]:
    """
    Check if the specified hormones returned to normal in any follow-up reading
    taken within `within_minutes` of the first reading.

    Returns:
        True   — recovered
        False  — did NOT recover within window
        None   — no follow-up readings available (cannot determine)
    """
    if len(readings) < 2:
        return None

    t0 = float(readings[0].get('timestamp_min', 0))
    s  = _sex(readings[0])

    for r in readings[1:]:
        t       = float(r.get('timestamp_min', 0))
        elapsed = t - t0
        if elapsed <= 0 or elapsed > within_minutes:
            continue

        recovered = all(
            _val(r, k) is None or _in_normal(k, _val(r, k), s)
            for k in keys
        )
        if recovered:
            return True

    return False  # window passed, no reading showed recovery


# ─────────────────────────────────────────────────────────────────────────────
# CONFOUND PRE-CHECK  (always run first — document Section 2.1)
# ─────────────────────────────────────────────────────────────────────────────

def _check_insulin_confound(reading: dict) -> Optional[str]:
    """
    Returns a warning string if Insulin > 40 μIU/mL.
    Per the document: reactive hypoglycemia produces an IDENTICAL catecholamine
    profile to genuine Fear/Panic or Acute Stress.
    User choice: emit warning alongside result, not suppress classification.
    """
    insulin = _val(reading, 'insulin_uIUmL')
    if insulin is not None and insulin > 40:
        return (
            f"⚠  METABOLIC CONFOUND DETECTED — Insulin = {insulin:.1f} μIU/mL (threshold: 40). "
            f"Reactive hypoglycaemia can produce an identical Epinephrine/Norepinephrine/Cortisol "
            f"surge as genuine Fear/Panic or Acute Stress. "
            f"Emotion classification below is TENTATIVE. Advise blood glucose check."
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FLOWCHART 1 — FEAR / PANIC  (Section 2.8)
# Highest catecholamine values of all 8 states. Epi dominates over NE.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_fear_panic(reading: dict) -> Optional[ClassificationResult]:
    epi    = _val(reading, 'epinephrine_pgmL')
    ne     = _val(reading, 'norepinephrine_pgmL')
    cort   = _val(reading, 'cortisol_nmolL')
    acth   = _val(reading, 'acth_pgmL')
    vaso   = _val(reading, 'vasopressin_pgmL')
    ne_epi = _val(reading, 'ne_epi_ratio')
    epi_ne = _val(reading, 'epi_ne_ratio')

    # Step 2: All 3 mandatory
    if not (epi is not None and epi > 500
            and ne   is not None and ne  > 800
            and cort is not None and cort > 600):
        return None

    # Ratio check — if NE/Epi > 3, this is Anger not Fear
    if ne_epi is not None and ne_epi > 3:
        return None

    markers = [
        f"Epinephrine ↑↑↑ = {epi:.1f} pg/mL  (mandatory > 500)",
        f"Norepinephrine ↑↑↑ = {ne:.1f} pg/mL  (mandatory > 800)",
        f"Cortisol ↑↑ = {cort:.1f} nmol/L  (mandatory > 600)",
    ]
    notes  = []
    conf   = 0

    # Epi/NE ratio confirms Fear signature
    if epi_ne is not None and epi_ne > 0.5:
        markers.append(f"Epi/NE ratio = {epi_ne:.2f}  (> 0.5 — Epinephrine dominant ✓ Fear signature)")
        conf += 1

    # Step 4: Confirmatory
    if acth is not None and acth > 80:
        markers.append(f"ACTH ↑↑ = {acth:.1f} pg/mL  (> 80) [confirmatory]")
        conf += 1
    if vaso is not None and vaso > 4:
        markers.append(f"Vasopressin ↑ = {vaso:.1f} pg/mL  (> 4) [confirmatory]")
        conf += 1

    # Severity grading by Epinephrine level (document Section 2.8 bottom)
    if epi > 2000:
        notes.append("Severity: FULL PANIC ATTACK — Epinephrine > 2000 pg/mL")
        confidence = "Very High"
    elif epi > 500:
        notes.append("Severity: Fear confirmed — Epinephrine 500–2000 pg/mL")
        confidence = "High" if conf >= 1 else "Moderate-High"
    else:
        notes.append("Severity: Fear / high anxiety — Epinephrine 200–500 pg/mL")
        confidence = "Moderate"

    return ClassificationResult(
        emotion="Fear_Panic",
        confidence=confidence,
        markers_matched=markers,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FLOWCHART 2 — ACUTE STRESS  (Section 2.7)
# Fastest onset. SAM axis fires within seconds. Resolves < 90 min.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_acute_stress(reading: dict, seq: list) -> Optional[ClassificationResult]:
    epi  = _val(reading, 'epinephrine_pgmL')
    ne   = _val(reading, 'norepinephrine_pgmL')
    acth = _val(reading, 'acth_pgmL')
    cort = _val(reading, 'cortisol_nmolL')
    prol = _val(reading, 'prolactin_ngmL')

    # Step 1: All 3 mandatory
    if not (epi  is not None and epi  > 200
            and ne   is not None and ne   > 500
            and acth is not None and acth > 100):
        return None

    markers = [
        f"Epinephrine ↑↑↑ = {epi:.1f} pg/mL  (mandatory > 200)",
        f"Norepinephrine ↑↑↑ = {ne:.1f} pg/mL  (mandatory > 500)",
        f"ACTH ↑↑ = {acth:.1f} pg/mL  (mandatory > 100)",
    ]
    notes = []
    conf  = 0

    # Step 2: Confirmatory
    if cort is not None and cort > 600:
        markers.append(f"Cortisol ↑↑ = {cort:.1f} nmol/L  (> 600, peaks at 15–30 min) [confirmatory]")
        conf += 1
    if prol is not None and prol > 15:
        markers.append(f"Prolactin ↑ = {prol:.1f} ng/mL  (> 15) [confirmatory]")
        conf += 1

    # Step 3: Time check — must resolve < 90 min to be Acute Stress (not Anxiety)
    recovery_keys = ['epinephrine_pgmL', 'norepinephrine_pgmL', 'cortisol_nmolL']
    recovered = _hormones_recovered(seq, recovery_keys, within_minutes=90)

    if recovered is None:
        notes.append("⏱ No follow-up readings provided — time-based recovery check skipped.")
    elif recovered:
        notes.append("⏱ Hormones recovered within 90 min → Acute Stress confirmed (not Anxiety).")
        conf += 1
    else:
        # Did not recover → reclassify as Anxiety
        notes.append("⏱ Hormones did NOT recover within 90 min → Reclassifying as ANXIETY.")
        return None

    confidence = _conf_grade(conf, ["Moderate", "Moderate-High", "High", "Very High"])

    return ClassificationResult(
        emotion="Acute_Stress",
        confidence=confidence,
        markers_matched=markers,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FLOWCHART 3 — ANGER  (Section 2.3)
# NE dominant over Epi (NE/Epi > 3).
# ─────────────────────────────────────────────────────────────────────────────

def _classify_anger(reading: dict, seq: list) -> Optional[ClassificationResult]:
    cort   = _val(reading, 'cortisol_nmolL')
    epi    = _val(reading, 'epinephrine_pgmL')
    ne     = _val(reading, 'norepinephrine_pgmL')
    ne_epi = _val(reading, 'ne_epi_ratio')
    test   = _val(reading, 'testosterone_ngdL')
    sero   = _val(reading, 'serotonin_ngmL')
    estr   = _val(reading, 'estradiol_pgmL')
    prog   = _val(reading, 'progesterone_ngmL')
    s      = _sex(reading)

    # Step 1: All 3 mandatory
    if not (cort is not None and cort > 500
            and epi  is not None and epi  > 200
            and ne   is not None and ne   > 600):
        return None

    markers = [
        f"Cortisol ↑↑ = {cort:.1f} nmol/L  (mandatory > 500)",
        f"Epinephrine ↑↑ = {epi:.1f} pg/mL  (mandatory > 200)",
        f"Norepinephrine ↑↑ = {ne:.1f} pg/mL  (mandatory > 600)",
    ]
    notes = []
    conf  = 0

    # Step 2: NE/Epi ratio (separates Anger from Fear)
    if ne_epi is not None:
        if ne_epi < 0.5:
            return None  # Fear/Panic territory
        if ne_epi > 3:
            markers.append(f"NE/Epi ratio = {ne_epi:.2f}  (> 3 — NE dominant ✓ Anger signature)")
            conf += 1
        else:
            notes.append(f"NE/Epi ratio = {ne_epi:.2f}  (not > 3 — Anger signal weaker)")

    # Step 3: Confirmatory moderate evidence
    if test is not None:
        baseline = 650.0 if s == 'M' else 42.0   # midpoints of normal range
        if test > baseline * 1.20:
            markers.append(
                f"Testosterone ↑ = {test:.1f} ng/dL  (> 20% above baseline {baseline:.0f}) [confirmatory]"
            )
            conf += 1
    if sero is not None and sero < 70:
        markers.append(f"Serotonin ↓ = {sero:.1f} ng/mL  (< 70) [confirmatory]")
        conf += 1

    # Step 4: Sex-specific confirmatory
    if s == 'F':
        if estr is not None and estr < 60:
            markers.append(
                f"Estradiol ↓ = {estr:.1f} pg/mL  (< 60, pre-menstrual low = anger trigger) [sex-specific]"
            )
            conf += 1
        if prog is not None and prog < 0.5:
            markers.append(
                f"Progesterone ↓ = {prog:.2f} ng/mL  (< 0.5, GABA buffer lost) [sex-specific]"
            )
            conf += 1

    # Step 5: Time check — if resolves < 90 min → it's Acute Stress, not Anger
    recovery_keys = ['cortisol_nmolL', 'epinephrine_pgmL', 'norepinephrine_pgmL']
    recovered = _hormones_recovered(seq, recovery_keys, within_minutes=90)

    if recovered is None:
        notes.append("⏱ No follow-up readings — time check skipped.")
    elif recovered:
        notes.append("⏱ Hormones resolved within 90 min → Reclassifying as ACUTE STRESS.")
        return None
    else:
        notes.append("⏱ Hormones sustained > 90 min → Anger pattern confirmed.")
        conf += 1

    confidence = _conf_grade(conf, ["Moderate", "Moderate", "Moderate-High", "High", "Very High"])

    return ClassificationResult(
        emotion="Anger",
        confidence=confidence,
        markers_matched=markers,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FLOWCHART 4 — ANXIETY  (Section 2.6)
# Overlaps with Anger/Stress; key separators are GABA and 90-min duration.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_anxiety(reading: dict, seq: list) -> Optional[ClassificationResult]:
    cort   = _val(reading, 'cortisol_nmolL')
    ne     = _val(reading, 'norepinephrine_pgmL')
    epi    = _val(reading, 'epinephrine_pgmL')
    gaba   = _val(reading, 'gaba_nmolmL')
    acth   = _val(reading, 'acth_pgmL')
    epi_ne = _val(reading, 'epi_ne_ratio')
    estr   = _val(reading, 'estradiol_pgmL')
    prog   = _val(reading, 'progesterone_ngmL')
    s      = _sex(reading)

    # Step 1: All 3 mandatory
    if not (cort is not None and cort > 350
            and ne  is not None and ne   > 500
            and epi is not None and epi  > 100):
        return None

    markers = [
        f"Cortisol ↑↑ = {cort:.1f} nmol/L  (mandatory > 350)",
        f"Norepinephrine ↑↑ = {ne:.1f} pg/mL  (mandatory > 500)",
        f"Epinephrine ↑↑ = {epi:.1f} pg/mL  (mandatory > 100)",
    ]
    notes = []
    conf  = 0

    # Step 2: GABA check — normal in Anger/Stress, suppressed in Anxiety
    if gaba is not None and gaba < 0.12:
        markers.append(
            f"GABA ↓ = {gaba:.3f} nmol/mL  (< 0.12 — rules out Anger/Acute Stress) [key differentiator]"
        )
        conf += 1
    if acth is not None and acth > 60:
        markers.append(f"ACTH ↑ = {acth:.1f} pg/mL  (> 60) [confirmatory]")
        conf += 1

    # Step 3: Time check — sustained > 90 min confirms Anxiety over Acute Stress
    recovery_keys = ['cortisol_nmolL', 'epinephrine_pgmL', 'norepinephrine_pgmL']
    recovered = _hormones_recovered(seq, recovery_keys, within_minutes=90)

    if recovered is None:
        notes.append("⏱ No follow-up readings — sustained duration check skipped.")
    elif recovered:
        notes.append("⏱ Hormones recovered within 90 min → Reclassifying as ACUTE STRESS.")
        return None
    else:
        notes.append("⏱ Hormones sustained > 90 min → Anxiety confirmed (not Acute Stress).")
        conf += 1

    # Step 4: Separate from Fear/Panic — NE should dominate in Anxiety
    if epi_ne is not None and epi_ne > 0.5:
        notes.append(
            f"⚠ Epi/NE ratio = {epi_ne:.2f}  (> 0.5 — Epinephrine approaching dominance, possible Fear/Panic overlap)"
        )

    # Step 5: Sex-specific High Evidence
    if s == 'F':
        if estr is not None and estr < 40:
            markers.append(
                f"Estradiol ↓ = {estr:.1f} pg/mL  (< 40 — lowers GABA-A sensitivity) [sex-specific High Evidence]"
            )
            conf += 1
        if prog is not None and prog < 1.0:
            markers.append(
                f"Progesterone ↓ = {prog:.2f} ng/mL  (< 1.0 — allopregnanolone ↓, GABA buffer lost) [sex-specific High Evidence]"
            )
            conf += 1

    confidence = _conf_grade(conf, ["Moderate", "Moderate", "Moderate-High", "High", "Very High"])

    return ClassificationResult(
        emotion="Anxiety",
        confidence=confidence,
        markers_matched=markers,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FLOWCHART 5 — DEPRESSION  (Section 2.2)
# Unique fingerprint: all 3 of Serotonin/Dopamine/BDNF suppressed simultaneously.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_depression(reading: dict) -> Optional[ClassificationResult]:
    sero = _val(reading, 'serotonin_ngmL')
    dopa = _val(reading, 'dopamine_pgmL')
    bdnf = _val(reading, 'bdnf_ngmL')
    cort = _val(reading, 'cortisol_nmolL')
    il6  = _val(reading, 'il6_pgmL')
    acth = _val(reading, 'acth_pgmL')
    estr = _val(reading, 'estradiol_pgmL')
    prog = _val(reading, 'progesterone_ngmL')
    test = _val(reading, 'testosterone_ngdL')
    s    = _sex(reading)

    # Step 1: All 3 mandatory (unique Depression fingerprint)
    if not (sero is not None and sero < 50
            and dopa is not None and dopa < 10
            and bdnf is not None and bdnf < 20):
        return None

    markers = [
        f"Serotonin ↓↓ = {sero:.1f} ng/mL  (mandatory < 50; anhedonia: < 5)",
        f"Dopamine ↓↓ = {dopa:.1f} pg/mL  (mandatory < 10)",
        f"BDNF ↓↓ = {bdnf:.1f} ng/mL  (mandatory < 20; severe MDD: < 15)",
    ]
    notes = []
    conf  = 0

    # Note severe sub-thresholds
    if dopa < 5:
        notes.append(f"Dopamine = {dopa:.1f} pg/mL — anhedonia threshold (< 5 pg/mL) reached.")
    if bdnf < 15:
        notes.append(f"BDNF = {bdnf:.1f} ng/mL — severe MDD threshold (< 15 ng/mL) reached.")

    # Step 2: Cortisol pattern (all-day elevation, blunted diurnal rhythm)
    if cort is not None and cort > 500:
        markers.append(
            f"Cortisol ↑↑ = {cort:.1f} nmol/L  (> 500, chronic all-day elevation — HPA dysregulation) [High Evidence]"
        )
        conf += 1

    # Step 3: Additional High Evidence
    if il6 is not None and il6 > 7:
        markers.append(f"IL-6 ↑ = {il6:.1f} pg/mL  (> 7 — neuroinflammation) [High Evidence]")
        conf += 1
    if acth is not None and acth > 60:
        markers.append(f"ACTH ↑ = {acth:.1f} pg/mL  (> 60 — HPA hyperactivation) [High Evidence]")
        conf += 1

    # Step 4: Sex-specific High Evidence
    if s == 'F':
        if estr is not None and estr < 50:
            markers.append(
                f"Estradiol ↓↓ = {estr:.1f} pg/mL  (< 50, perimenopausal) [sex-specific High Evidence]"
            )
            conf += 1
        if prog is not None and prog < 0.5:
            markers.append(
                f"Progesterone ↓↓ = {prog:.2f} ng/mL  (< 0.5) [sex-specific High Evidence]"
            )
            conf += 1
    elif s == 'M':
        if test is not None and test < 300:
            markers.append(
                f"Testosterone ↓ = {test:.1f} ng/dL  (< 300) [sex-specific Moderate Evidence]"
            )
            conf += 1

    # Confidence grading from document Section 2.2
    if conf == 0:
        confidence = "Low-Moderate"
        notes.append("Mandatory markers only — possible early depression.")
    elif conf <= 2:
        confidence = "Moderate-High"
        notes.append("Likely depression.")
    elif conf == 3:
        confidence = "High"
        notes.append("Probable depression.")
    else:
        confidence = "Very High"
        notes.append("Strong depression classification.")

    return ClassificationResult(
        emotion="Depression",
        confidence=confidence,
        markers_matched=markers,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FLOWCHART 6 — SADNESS  (Section 2.5)
# No High Evidence anchors. Key marker: episodic Prolactin spike + time recovery.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_sadness(reading: dict, seq: list) -> Optional[ClassificationResult]:
    prol = _val(reading, 'prolactin_ngmL')
    sero = _val(reading, 'serotonin_ngmL')
    dopa = _val(reading, 'dopamine_pgmL')
    cort = _val(reading, 'cortisol_nmolL')

    # Step 1: Mandatory — episodic prolactin spike > 20 ng/mL
    if prol is None or prol <= 20:
        return None

    markers = [f"Prolactin ↑ (episodic spike) = {prol:.1f} ng/mL  (mandatory > 20)"]
    notes   = []
    conf    = 0

    # Step 2: Confirm with ANY 2 of 3 moderate markers
    sero_ok = sero is not None and 40 <= sero <= 80
    dopa_ok = dopa is not None and 5  <= dopa <= 12
    cort_ok = cort is not None and 200 <= cort <= 450

    if sero_ok:
        markers.append(f"Serotonin ↓ = {sero:.1f} ng/mL  (mild 40–80, not severely low like Depression) [confirmatory]")
        conf += 1
    if dopa_ok:
        markers.append(f"Dopamine ↓ = {dopa:.1f} pg/mL  (mild 5–12) [confirmatory]")
        conf += 1
    if cort_ok:
        markers.append(f"Cortisol mildly ↑ = {cort:.1f} nmol/L  (200–450, transient) [confirmatory]")
        conf += 1

    if conf < 2:
        notes.append(f"Only {conf}/2 required confirmatory markers present — Sadness classification weak.")
        if conf == 0:
            return None

    # Step 3: CRITICAL time check — must recover within 60 min, otherwise → Depression
    recovery_keys = ['prolactin_ngmL', 'serotonin_ngmL', 'dopamine_pgmL']
    recovered = _hormones_recovered(seq, recovery_keys, within_minutes=60)

    if recovered is None:
        notes.append("⏱ No follow-up readings — recovery check skipped (cannot confirm vs. Depression).")
    elif recovered:
        notes.append("⏱ Values recovered within 60 min → Sadness confirmed (not Depression).")
        conf += 1
    else:
        notes.append("⏱ Values did NOT recover within 60 min → Reclassifying as DEPRESSION.")
        return None

    confidence = "Moderate-High" if conf >= 3 else "Moderate"

    return ClassificationResult(
        emotion="Sadness",
        confidence=confidence,
        markers_matched=markers,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FLOWCHART 7 — HAPPINESS / JOY  (Section 2.4)
# Only state where Serotonin + Dopamine + Oxytocin all go UP simultaneously.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_happiness(reading: dict) -> Optional[ClassificationResult]:
    sero = _val(reading, 'serotonin_ngmL')
    dopa = _val(reading, 'dopamine_pgmL')
    oxyt = _val(reading, 'oxytocin_pgmL')
    cort = _val(reading, 'cortisol_nmolL')
    bdnf = _val(reading, 'bdnf_ngmL')
    il6  = _val(reading, 'il6_pgmL')
    estr = _val(reading, 'estradiol_pgmL')
    gaba = _val(reading, 'gaba_nmolmL')
    s    = _sex(reading)

    # Step 1: All 3 mandatory
    if not (sero is not None and sero > 130
            and dopa is not None and dopa > 12
            and oxyt is not None and oxyt > 3):
        return None

    markers = [
        f"Serotonin ↑ = {sero:.1f} ng/mL  (mandatory > 130)",
        f"Dopamine ↑ = {dopa:.1f} pg/mL  (mandatory > 12; active happiness: 15–30)",
        f"Oxytocin ↑ = {oxyt:.1f} pg/mL  (mandatory > 3)",
    ]
    notes = []
    conf  = 0

    # Step 2: Separation from Calm
    # GABA elevated + Dopamine moderate → reclassify as Calm
    if gaba is not None and gaba > 0.20 and dopa is not None and dopa <= 15:
        notes.append(
            f"GABA elevated ({gaba:.3f}) + Dopamine moderate ({dopa:.1f} ≤ 15) → Reclassifying as CALM."
        )
        return None

    if dopa is not None and dopa > 15:
        markers.append(f"Dopamine actively elevated = {dopa:.1f} pg/mL  (> 15 — active Happiness, not Calm)")
        conf += 1
    if cort is not None and cort < 250:
        markers.append(f"Cortisol ↓ = {cort:.1f} nmol/L  (< 250) [confirmatory]")
        conf += 1

    # Step 3: Additional confirmatory
    if bdnf is not None and bdnf > 28:
        markers.append(f"BDNF ↑ = {bdnf:.1f} ng/mL  (> 28) [confirmatory]")
        conf += 1
    if il6 is not None and il6 < 5:
        markers.append(f"IL-6 ↓ = {il6:.1f} pg/mL  (< 5) [confirmatory]")
        conf += 1

    # Step 4: Sex-specific (strong confidence booster for females)
    if s == 'F' and estr is not None and 80 <= estr <= 200:
        markers.append(
            f"Estradiol ↑ = {estr:.1f} pg/mL  (80–200, follicular peak = peak positive mood) [sex-specific High Evidence]"
        )
        conf += 1

    confidence = _conf_grade(conf, ["Moderate", "Moderate-High", "High", "Very High"])

    return ClassificationResult(
        emotion="Happiness",
        confidence=confidence,
        markers_matched=markers,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FLOWCHART 8 — CALM / RELAXED  (Section 2.9)
# Simplest state: all stress hormones simultaneously low.
# Female users: Progesterone is an ADDITIONAL MANDATORY marker.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_calm(reading: dict) -> Optional[ClassificationResult]:
    cort = _val(reading, 'cortisol_nmolL')
    epi  = _val(reading, 'epinephrine_pgmL')
    ne   = _val(reading, 'norepinephrine_pgmL')
    prog = _val(reading, 'progesterone_ngmL')
    gaba = _val(reading, 'gaba_nmolmL')
    dopa = _val(reading, 'dopamine_pgmL')
    sero = _val(reading, 'serotonin_ngmL')
    oxyt = _val(reading, 'oxytocin_pgmL')
    s    = _sex(reading)

    # Step 1: All 3 mandatory (both sexes)
    if not (cort is not None and cort < 250
            and epi  is not None and epi  < 40
            and ne   is not None and ne   < 300):
        return None

    markers = [
        f"Cortisol ↓ = {cort:.1f} nmol/L  (mandatory < 250)",
        f"Epinephrine ↓ = {epi:.1f} pg/mL  (mandatory < 40)",
        f"Norepinephrine ↓ = {ne:.1f} pg/mL  (mandatory < 300)",
    ]
    notes = []
    conf  = 0

    # Step 2: Additional MANDATORY for females
    if s == 'F':
        if prog is None or prog <= 3:
            prog_str = f"{prog:.2f} ng/mL" if prog is not None else "not provided"
            notes.append(
                f"⚠ FEMALE MANDATORY marker absent: Progesterone ↑ > 3 ng/mL "
                f"(current: {prog_str}). Calm classification suppressed for this female profile."
            )
            return None
        else:
            markers.append(
                f"Progesterone ↑ = {prog:.2f} ng/mL  (> 3, mid-luteal — GABA-A activation) [FEMALE mandatory]"
            )
            conf += 1

    # Step 3: Separation from Happiness
    # Dopamine > 15 + Oxytocin ↑ + Serotonin > 150 → reclassify as Happiness
    if (dopa is not None and dopa > 15
            and oxyt is not None and oxyt > 3
            and sero is not None and sero > 150):
        notes.append(
            f"Dopamine {dopa:.1f} > 15 + Oxytocin {oxyt:.1f} > 3 + Serotonin {sero:.1f} > 150 "
            f"→ Reclassifying as HAPPINESS."
        )
        return None

    if gaba is not None and gaba > 0.20:
        markers.append(
            f"GABA ↑ = {gaba:.3f} nmol/mL  (> 0.20 — confirms Calm over Happiness)"
        )
        conf += 1
    if dopa is not None and dopa <= 15:
        markers.append(f"Dopamine moderate = {dopa:.1f} pg/mL  (≤ 15, not actively elevated)")
        conf += 1
    if sero is not None and 100 <= sero <= 180:
        markers.append(f"Serotonin stable = {sero:.1f} ng/mL  (100–180)")
        conf += 1

    confidence = _conf_grade(conf, ["Moderate", "Moderate-High", "High", "Very High"])

    return ClassificationResult(
        emotion="Calm",
        confidence=confidence,
        markers_matched=markers,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CLASSIFIER  (Section 2.1 — Master Overview)
# ─────────────────────────────────────────────────────────────────────────────

def classify(readings_sequence: list) -> ClassificationResult:
    """
    Main entry point.

    Args:
        readings_sequence: A time-ordered list of reading dicts.
            Each dict must contain hormone values keyed as in the CSV
            (e.g. 'cortisol_nmolL', 'epinephrine_pgmL', etc.) plus:
                'timestamp_min' — minutes since first reading (0 for first)
                'sex'           — 'M' or 'F'
            At minimum, one reading is required.
            Additional readings enable time-based recovery checks.

    Returns:
        ClassificationResult with emotion, confidence, matched markers,
        notes, and any metabolic confound warnings.
    """
    if not readings_sequence:
        raise ValueError("At least one reading is required.")

    reading  = readings_sequence[0]
    warnings = []

    # ── CONFOUND PRE-CHECK — always first, non-negotiable ──────────────────
    confound_msg      = _check_insulin_confound(reading)
    metabolic_confound = confound_msg is not None
    if confound_msg:
        warnings.append(confound_msg)

    # ── FLOWCHART PRIORITY ORDER ────────────────────────────────────────────
    # States with the highest/most extreme thresholds are checked first to
    # prevent lower-threshold states from claiming a match first.
    #   1. Fear/Panic   — highest catecholamine levels (Epi > 500, NE > 800)
    #   2. Acute Stress — high catecholamines + ACTH; resolves < 90 min
    #   3. Anger        — same catecholamine range as Stress; NE/Epi ratio key
    #   4. Anxiety      — overlaps Anger/Stress; GABA + 90-min duration key
    #   5. Depression   — monoamine depletion fingerprint (Sero+Dopa+BDNF ↓↓)
    #   6. Sadness      — Prolactin episodic spike + 60-min recovery
    #   7. Happiness    — all positive triad simultaneously elevated
    #   8. Calm         — everything quiet; NE/Epi/Cortisol all suppressed

    result = (
        _classify_fear_panic(reading)
        or _classify_acute_stress(reading, readings_sequence)
        or _classify_anger(reading, readings_sequence)
        or _classify_anxiety(reading, readings_sequence)
        or _classify_depression(reading)
        or _classify_sadness(reading, readings_sequence)
        or _classify_happiness(reading)
        or _classify_calm(reading)
    )

    # Fallback — no flowchart matched
    if result is None:
        result = ClassificationResult(
            emotion="Unclassified",
            confidence="None",
            notes=["No flowchart rules were satisfied by the provided hormone profile."],
        )

    result.warnings.extend(warnings)
    result.metabolic_confound = metabolic_confound
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING — run classifier on an entire CSV dataset
# ─────────────────────────────────────────────────────────────────────────────

def batch_classify_csv(filepath: str, max_rows: int = None) -> dict:
    """
    Read the CSV dataset, classify each row as a single-reading sequence,
    and return accuracy + per-class statistics.

    NOTE: Single-row classification cannot use time-based recovery checks.
    All time-based notes will say 'No follow-up readings provided'.
    """
    correct   = 0
    total     = 0
    per_class = {}

    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows and total >= max_rows:
                break

            true_label = row.get('mental_health_state', '').strip()
            if not true_label:
                continue

            # Build reading dict — map CSV column names directly
            reading = {k: v for k, v in row.items()}
            reading['timestamp_min'] = 0

            try:
                result = classify([reading])
            except Exception:
                continue

            predicted = result.emotion
            match     = (predicted == true_label)

            if match:
                correct += 1
            total += 1

            if true_label not in per_class:
                per_class[true_label] = {'correct': 0, 'total': 0}
            per_class[true_label]['total'] += 1
            if match:
                per_class[true_label]['correct'] += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    return {
        'accuracy':  accuracy,
        'correct':   correct,
        'total':     total,
        'per_class': per_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: ClassificationResult, label: str = ""):
    W = 65
    print("═" * W)
    if label:
        print(f"  TEST: {label}")
        print("─" * W)

    if result.metabolic_confound:
        print("  ⚠  METABOLIC CONFOUND DETECTED")
        for w in result.warnings:
            # Word-wrap warning
            words = w.split()
            line  = "     "
            for word in words:
                if len(line) + len(word) > W - 2:
                    print(line)
                    line = "     "
                line += word + " "
            print(line)
        print()

    print(f"  Emotion     :  {result.emotion}")
    print(f"  Confidence  :  {result.confidence}")
    if result.reclassified_from:
        print(f"  Reclassified from: {result.reclassified_from}")

    if result.markers_matched:
        print()
        print("  Markers matched:")
        for m in result.markers_matched:
            print(f"    ✓  {m}")

    if result.notes:
        print()
        print("  Notes:")
        for n in result.notes:
            print(f"    →  {n}")

    print("═" * W)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — four test cases covering the key document scenarios
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── TEST 1: Fear/Panic with metabolic confound warning ──────────────────
    print_result(classify([{
        "timestamp_min": 0, "sex": "M", "age": 34, "bmi": 24.5,
        "cortisol_nmolL": 720,   "epinephrine_pgmL": 900,
        "norepinephrine_pgmL": 1300, "ne_epi_ratio": 1.44, "epi_ne_ratio": 0.69,
        "acth_pgmL": 190,  "vasopressin_pgmL": 6.0,
        "insulin_uIUmL": 48,          # ← CONFOUND > 40
        "serotonin_ngmL": 55,  "dopamine_pgmL": 14,  "oxytocin_pgmL": 3.5,
        "bdnf_ngmL": 22,   "il6_pgmL": 5,    "gaba_nmolmL": 0.19,
        "prolactin_ngmL": 14,  "leptin_ngmL": 5,
        "testosterone_ngdL": 500,  "estradiol_pgmL": 25,
        "progesterone_ngmL": 0.2,  "melatonin_pgmL": 100,
    }]), label="Fear/Panic + Metabolic Confound")

    # ── TEST 2: Female Anxiety sustained > 90 min ───────────────────────────
    print_result(classify([
        {   # T=0 — initial reading
            "timestamp_min": 0,  "sex": "F",  "age": 28,  "bmi": 22.0,
            "cortisol_nmolL": 510,  "epinephrine_pgmL": 260,
            "norepinephrine_pgmL": 820,  "ne_epi_ratio": 3.15, "epi_ne_ratio": 0.32,
            "acth_pgmL": 95,   "gaba_nmolmL": 0.08,
            "insulin_uIUmL": 18,
            "serotonin_ngmL": 75,  "dopamine_pgmL": 8,  "oxytocin_pgmL": 1.5,
            "bdnf_ngmL": 18,  "il6_pgmL": 8,   "prolactin_ngmL": 12,
            "leptin_ngmL": 8,  "testosterone_ngdL": 40,
            "estradiol_pgmL": 35,   # < 40 — sex-specific High Evidence
            "progesterone_ngmL": 0.7, "vasopressin_pgmL": 2, "melatonin_pgmL": 90,
        },
        {   # T=100 min — still elevated (no recovery) → confirms Anxiety, not Stress
            "timestamp_min": 100, "sex": "F",
            "cortisol_nmolL": 490, "epinephrine_pgmL": 240, "norepinephrine_pgmL": 790,
        },
    ]), label="Female Anxiety (sustained > 90 min)")

    # ── TEST 3: Sadness — prolactin spike, recovers within 60 min ───────────
    print_result(classify([
        {   # T=0
            "timestamp_min": 0,  "sex": "M",  "age": 22,  "bmi": 23.0,
            "cortisol_nmolL": 310,  "epinephrine_pgmL": 45,
            "norepinephrine_pgmL": 190,  "ne_epi_ratio": 4.2, "epi_ne_ratio": 0.24,
            "prolactin_ngmL": 35,   # ← episodic spike > 20
            "serotonin_ngmL": 62,   # mild reduction 40–80
            "dopamine_pgmL": 9,     # mild reduction 5–12
            "acth_pgmL": 35,   "gaba_nmolmL": 0.20,  "insulin_uIUmL": 10,
            "oxytocin_pgmL": 2, "bdnf_ngmL": 22, "il6_pgmL": 6,
            "testosterone_ngdL": 600,  "estradiol_pgmL": 20,
            "progesterone_ngmL": 0.2,  "vasopressin_pgmL": 1,
            "leptin_ngmL": 5,  "melatonin_pgmL": 110,
        },
        {   # T=45 min — values back to normal → Sadness confirmed
            "timestamp_min": 45,  "sex": "M",
            "prolactin_ngmL": 10,   # recovered
            "serotonin_ngmL": 120,  # recovered
            "dopamine_pgmL": 18,    # recovered
        },
    ]), label="Sadness (recovers within 60 min)")

    # ── TEST 4: Female Depression (perimenopausal) ───────────────────────────
    print_result(classify([{
        "timestamp_min": 0,  "sex": "F",  "age": 48,  "bmi": 27.0,
        "cortisol_nmolL": 660,  "epinephrine_pgmL": 30,
        "norepinephrine_pgmL": 145,  "ne_epi_ratio": 4.8, "epi_ne_ratio": 0.21,
        "serotonin_ngmL": 18,   # ↓↓ mandatory
        "dopamine_pgmL": 3,     # ↓↓ mandatory (anhedonia)
        "bdnf_ngmL": 11,        # ↓↓ mandatory (severe MDD)
        "acth_pgmL": 105,  "il6_pgmL": 16,  "gaba_nmolmL": 0.18,
        "prolactin_ngmL": 14,  "insulin_uIUmL": 26,  "leptin_ngmL": 10,
        "testosterone_ngdL": 30,
        "estradiol_pgmL": 16,   # < 50 — perimenopausal
        "progesterone_ngmL": 0.3, "vasopressin_pgmL": 1.5,
        "oxytocin_pgmL": 1.2,  "melatonin_pgmL": 50,
    }]), label="Female Depression (perimenopausal, Very High confidence)")

    # ── OPTIONAL: Batch evaluate on dataset if CSV is present ───────────────
    CSV_PATH = os.path.join(os.path.dirname(__file__), "micra_h_sense_dataset.csv")
    if os.path.exists(CSV_PATH):
        print("\n" + "═" * 65)
        print("  BATCH EVALUATION ON DATASET")
        print("═" * 65)
        stats = batch_classify_csv(CSV_PATH)
        print(f"  Overall Accuracy : {stats['accuracy']:.1f}%  "
              f"({stats['correct']}/{stats['total']} correct)")
        print()
        print("  Per-Class Results:")
        for cls, d in sorted(stats['per_class'].items()):
            acc = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
            print(f"    {cls:<18} {acc:5.1f}%  ({d['correct']}/{d['total']})")
        print("═" * 65)
