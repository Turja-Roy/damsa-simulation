#!/usr/bin/env python3
"""Build APS talk slides and speaker notes for DAMSA.

Outputs:
  - APS_DAMSA_Talk.pptx
  - APS_DAMSA_Talk_speaker_notes.md

The deck uses repo figures where possible and draws a lightweight geometry
schematic directly on the title/concept slides.
"""

from __future__ import annotations

import os
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


ROOT = Path(__file__).resolve().parent.parent
PLOT = ROOT / "plots"
OUT_PPTX = ROOT / "APS_DAMSA_Talk.pptx"
OUT_NOTES = ROOT / "APS_DAMSA_Talk_speaker_notes.md"

TITLE = "DAMSA Simulation and Geometry Optimization"
SUBTITLE = "APS 2026 | Turja Roy | The University of Texas at Arlington | April 10, 2026"


def add_notes(notes, text):
    notes.append(text.strip())
    notes.append("")


def set_bg(slide, color=(248, 249, 251)):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(*color)


def add_title(slide, title, subtitle=None):
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.25), Inches(12.2), Inches(0.6))
    tf = box.text_frame
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.color.rgb = RGBColor(20, 24, 33)
    if subtitle:
        box2 = slide.shapes.add_textbox(Inches(0.45), Inches(0.78), Inches(12.2), Inches(0.35))
        tf2 = box2.text_frame
        p2 = tf2.paragraphs[0]
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.size = Pt(12)
        r2.font.color.rgb = RGBColor(90, 96, 110)


def add_footer(slide, text="DAMSA | APS 2026"):
    box = slide.shapes.add_textbox(Inches(11.6), Inches(7.0), Inches(1.2), Inches(0.2))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = text
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(120, 126, 136)


def add_bullets(slide, bullets, left, top, width, height, font_size=20):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(2)
    tf.margin_right = Pt(2)
    tf.margin_top = Pt(1)
    tf.margin_bottom = Pt(1)
    first = True
    for bullet in bullets:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(font_size)
        p.font.color.rgb = RGBColor(35, 39, 47)
        p.space_after = Pt(4)
    return box


def add_image(slide, path, left, top, width=None, height=None):
    if width is not None and height is not None:
        return slide.shapes.add_picture(str(path), left, top, width=width, height=height)
    if width is not None:
        return slide.shapes.add_picture(str(path), left, top, width=width)
    if height is not None:
        return slide.shapes.add_picture(str(path), left, top, height=height)
    return slide.shapes.add_picture(str(path), left, top)


def add_callout(slide, text, left, top, width, height, color=(255, 250, 235), line=(208, 160, 52)):
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*color)
    shape.line.color.rgb = RGBColor(*line)
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = RGBColor(70, 55, 20)
    return shape


def add_geometry_schematic(slide, left, top, width, height):
    # Simple schematic: target -> VDC -> magnet -> calo
    y = top + height * 0.45
    h = height * 0.22
    colors = {
        "target": RGBColor(180, 71, 62),
        "vdc": RGBColor(44, 120, 198),
        "magnet": RGBColor(97, 97, 97),
        "calo": RGBColor(220, 140, 30),
    }
    blocks = [
        (0.02, 0.14, "W target", "target"),
        (0.23, 0.18, "VDC", "vdc"),
        (0.47, 0.16, "Magnet", "magnet"),
        (0.72, 0.22, "CsI calo", "calo"),
    ]
    for rel_x, rel_w, label, key in blocks:
        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
            Inches(left + width * rel_x),
            Inches(y),
            Inches(width * rel_w),
            Inches(h),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = colors[key]
        shape.line.color.rgb = RGBColor(255, 255, 255)
        tf = shape.text_frame
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = label
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
    # arrows
    for rel_x in [0.16, 0.41, 0.64]:
        line = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.CHEVRON,
            Inches(left + width * rel_x),
            Inches(y + h * 0.15),
            Inches(width * 0.05),
            Inches(h * 0.7),
        )
        line.fill.solid()
        line.fill.fore_color.rgb = RGBColor(80, 80, 90)
        line.line.color.rgb = RGBColor(80, 80, 90)
    # labels
    for rel_x, text in [(0.00, "8 GeV e-"), (0.26, "~1 m short baseline"), (0.74, "a -> gamma gamma")]:
        box = slide.shapes.add_textbox(Inches(left + width * rel_x), Inches(top + height * 0.02), Inches(width * 0.22), Inches(height * 0.18))
        p = box.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = text
        r.font.size = Pt(11)
        r.font.color.rgb = RGBColor(70, 75, 85)


def slide_title(prs, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, TITLE, SUBTITLE)
    add_geometry_schematic(slide, 0.55, 1.5, 12.0, 1.6)
    add_callout(slide, "Ultra-short-baseline ALP search: catch decays before they escape the detector.", Inches(0.8), Inches(3.6), Inches(11.2), Inches(0.6))
    add_bullets(slide, ["Bremsstrahlung photons from an 8 GeV electron beam", "Geant4 beamline + analytic ALP optimization", "Particle-physics audience: background, acceptance, and sensitivity"], Inches(0.9), Inches(4.5), Inches(11.2), Inches(1.5), font_size=17)
    add_footer(slide)
    add_notes(notes, "Open with the central thesis: DAMSA is designed to exploit an ultra-short baseline so short-lived ALPs still decay within the detector acceptance. Briefly point to the schematic and say the talk goes from motivation to simulation to geometry optimization.")


def slide_motivation(prs, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Motivation: the low-mass dark matter / dark-sector gap")
    add_image(slide, PLOT / "sensitivity_curve.png", Inches(0.55), Inches(1.25), width=Inches(5.7))
    add_image(slide, PLOT / "opening_angles.png", Inches(6.45), Inches(1.25), width=Inches(6.0))
    add_callout(slide, "DAMSA targets the MeV-to-sub-GeV regime where direct detection loses reach and long baselines lose short-lived particles.", Inches(0.75), Inches(6.25), Inches(11.6), Inches(0.6))
    add_footer(slide)
    add_notes(notes, "Set up the gap in a way a particle physicist will recognize immediately: the relevant open space is low mass, low coupling, and beam-dump style experiments hit a baseline ceiling for short-lived states. Use the exclusion plot as the anchor and do not over-explain the cosmology pie chart if the room is mixed.")


def slide_concept(prs, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "DAMSA concept")
    add_geometry_schematic(slide, 0.8, 1.2, 11.6, 1.5)
    bullets = [
        "Primakoff production from bremsstrahlung photons in tungsten",
        "Signal topology: a -> gamma gamma inside the VDC, both photons reaching CsI",
        "Short baseline is the point: long-baseline setups lose these decays before detection",
        "Baseline geometry used here: 10 cm target, 35 cm VDC, 12x12 cm2 calo",
    ]
    add_bullets(slide, bullets, Inches(0.8), Inches(3.1), Inches(6.1), Inches(2.8), font_size=18)
    add_callout(slide, "The design problem is geometric: preserve signal acceptance while suppressing forward EM backgrounds.", Inches(7.1), Inches(3.35), Inches(5.2), Inches(0.9))
    add_footer(slide)
    add_notes(notes, "Describe the detector in one pass: electron beam on W, a short vacuum gap, a magnet, and a compact CsI calorimeter. Emphasize that the signal is fully reconstructable only if both photons survive to the calo face, so the geometry enters directly in the acceptance.")


def slide_simulation(prs, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Geant4 simulation and scoring strategy")
    add_image(slide, PLOT / "png/summary/energy_angle_count_grid.png", Inches(0.55), Inches(1.2), width=Inches(6.3))
    bullets = [
        "8 GeV e- beam on tungsten with EM shower development in Geant4",
        "Physics focus: bremsstrahlung photons, electrons, positrons, neutrons",
        "Multi-stage scoring at target exit, VDC entrance, and calorimeter face",
        "Output fluxes feed the ALP pipeline and the geometry optimization",
    ]
    add_bullets(slide, bullets, Inches(7.0), Inches(1.35), Inches(5.2), Inches(2.7), font_size=17)
    add_callout(slide, "The shower is the source term. Once it saturates, target-length changes become mostly analytic.", Inches(7.0), Inches(4.15), Inches(5.2), Inches(0.75))
    add_footer(slide)
    add_notes(notes, "Explain that Geant4 is used to establish the background phase space at a few detector planes, not to brute-force every geometry. Mention that the bremsstrahlung spectrum is the key input to the ALP generator and the downstream optimization.")


def slide_background(prs, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Background characterization")
    add_image(slide, PLOT / "png/TargetExit/overlay.png", Inches(0.45), Inches(1.18), width=Inches(4.05))
    add_image(slide, PLOT / "png/TargetExit/angle_gamma.png", Inches(4.6), Inches(1.18), width=Inches(4.0))
    add_image(slide, PLOT / "png/TargetExit/energy_gamma.png", Inches(8.7), Inches(1.18), width=Inches(4.0))
    add_callout(slide, "Forward photons dominate; the background is strongly collimated along the beam axis.", Inches(0.75), Inches(6.05), Inches(11.4), Inches(0.55))
    add_footer(slide)
    add_notes(notes, "Keep this slide tight. The important physics message is that the dominant background is forward EM radiation, mostly photons, and this justifies why both the VDC and calorimeter size matter. You do not need to dwell on species-by-species details unless asked.")


def slide_signal_kinematics(prs, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "ALP signal kinematics")
    add_image(slide, PLOT / "opening_angles.png", Inches(0.55), Inches(1.15), width=Inches(7.0))
    table = slide.shapes.add_table(4, 3, Inches(7.85), Inches(1.4), Inches(4.1), Inches(2.2)).table
    headers = ["m_a", "12 cm calo", "20 cm calo"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
    rows = [
        ["50-100 MeV", "~98-100%", "~100%"],
        ["200 MeV", "64%", "82%"],
        ["Takeaway", "size-limited", "acceptance recovers"],
    ]
    for r, row in enumerate(rows, start=1):
        for c, txt in enumerate(row):
            table.cell(r, c).text = txt
    add_callout(slide, "For heavier ALPs, the opening angle grows and calorimeter size becomes a first-order design parameter.", Inches(7.9), Inches(4.0), Inches(4.0), Inches(0.85))
    add_footer(slide)
    add_notes(notes, "Use the opening-angle plot to make the geometry intuition obvious. State that light ALPs are almost always contained, but by 200 MeV the acceptance penalty is large for a 12 cm calo, which is exactly why the calorimeter optimization matters.")


def slide_optimization_strategy(prs, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Optimization strategy")
    add_image(slide, ROOT / "build" / "gap_scan_20260313_095752_analysis.png", Inches(0.6), Inches(1.25), width=Inches(5.5))
    bullets = [
        "Scan parameters: target length, VDC length, and calo XY size",
        "Key simplification: target shower saturates by 10 cm",
        "VDC length and calo size can be optimized analytically after the Geant4 baseline",
        "Objective: minimize background while maximizing separable signal fraction",
    ]
    add_bullets(slide, bullets, Inches(6.4), Inches(1.35), Inches(6.0), Inches(2.8), font_size=17)
    add_callout(slide, "This reduces the optimization to a tractable Pareto search instead of repeated Geant4 reruns.", Inches(6.45), Inches(4.35), Inches(5.95), Inches(0.65))
    add_footer(slide)
    add_notes(notes, "Frame this as the methodological payoff. Once the target shower saturates, you can treat target-length changes analytically and spend simulation effort where it matters: background propagation and signal geometry.")


def slide_joint_pareto(prs, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Joint VDC x calorimeter optimization")
    add_image(slide, PLOT / "pareto" / "pareto_front.png", Inches(0.55), Inches(1.2), width=Inches(6.2))
    add_image(slide, PLOT / "pareto" / "pareto_gap_curves.png", Inches(6.95), Inches(1.2), width=Inches(5.55))
    add_callout(slide, "Result: VDC = 30 cm and Calo = 20x20 cm2 dominates across masses in the scanned range.", Inches(0.85), Inches(6.0), Inches(11.0), Inches(0.55))
    add_footer(slide)
    add_notes(notes, "Tell the audience that the joint scan is monotonic enough that the Pareto front is easy to interpret: shorter VDC reduces spread, larger calo recovers heavy-ALP acceptance. The optimum is not a delicate local extremum; it is a robust corner of the design space.")


def slide_global_pareto(prs, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Global 3D Pareto front")
    add_image(slide, PLOT / "global_pareto" / "global_pareto_front.png", Inches(0.35), Inches(1.15), width=Inches(7.4))
    add_image(slide, PLOT / "snr" / "snr_vs_mass.png", Inches(8.0), Inches(1.35), width=Inches(4.0))
    add_callout(slide, "Target-length gains are analytically projected; the VDC and calo optimum is the robust part of the result.", Inches(0.8), Inches(6.05), Inches(11.2), Inches(0.5))
    add_footer(slide)
    add_notes(notes, "Use this as the bridge between the two-parameter scan and the full three-parameter conclusion. Be explicit that the target-length trend is projection-based, while the VDC/calo result is directly anchored by the joint scan.")


def slide_sensitivity(prs, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Sensitivity projection and takeaways")
    add_image(slide, PLOT / "sensitivity_curve.png", Inches(0.55), Inches(1.15), width=Inches(6.3))
    bullets = [
        "Projected 90% CL reach at the optimized geometry",
        "The excluded band follows the geometry choice established by the Pareto search",
        "Optimal working point used here: 30 cm VDC, 20x20 cm2 calo, 10 cm target",
        "Next step: hardware-design studies at the optimized geometry and validation of target-length scaling",
    ]
    add_bullets(slide, bullets, Inches(7.0), Inches(1.35), Inches(5.3), Inches(2.9), font_size=16)
    add_callout(slide, "DAMSA closes the presentation loop: motivation, background, optimization, and a concrete sensitivity projection.", Inches(7.0), Inches(4.35), Inches(5.15), Inches(0.85))
    add_footer(slide)
    add_notes(notes, "End by tying the sensitivity curve back to the original motivation slide. The audience should leave with a clean picture: the optimized geometry is now fixed well enough to support the next design phase.")


def slide_future_work(prs, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Future work")
    bullets = [
        "Geant4 validation of the target-length background scaling",
        "Shielding and veto design for the forward photon flux",
        "Detector-response studies: energy resolution, angular thresholds, and reconstruction",
        "Refine the APS story into a hardware-credible baseline design",
    ]
    add_bullets(slide, bullets, Inches(0.85), Inches(1.45), Inches(7.0), Inches(3.0), font_size=20)
    add_callout(slide, "Use this slide to make the program feel like a live design effort, not a finished endpoint.", Inches(7.6), Inches(1.8), Inches(4.7), Inches(0.95))
    add_callout(slide, "Questions?", Inches(8.7), Inches(4.25), Inches(2.5), Inches(0.65), color=(230, 240, 255), line=(80, 120, 180))
    add_footer(slide)
    add_notes(notes, "If time permits, mention that the remaining work is engineering-facing rather than conceptual: the optimized geometry is already pinned down, so now the question is how to realize the veto, shielding, and readout cleanly.")


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    notes = []
    slide_title(prs, notes)
    slide_motivation(prs, notes)
    slide_concept(prs, notes)
    slide_simulation(prs, notes)
    slide_background(prs, notes)
    slide_signal_kinematics(prs, notes)
    slide_optimization_strategy(prs, notes)
    slide_joint_pareto(prs, notes)
    slide_global_pareto(prs, notes)
    slide_sensitivity(prs, notes)
    slide_future_work(prs, notes)

    prs.save(str(OUT_PPTX))
    OUT_NOTES.write_text("\n".join(f"## Slide {i+1}\n{note}" for i, note in enumerate(notes)), encoding="utf-8")
    print(f"Saved {OUT_PPTX}")
    print(f"Saved {OUT_NOTES}")


if __name__ == "__main__":
    build_deck()
