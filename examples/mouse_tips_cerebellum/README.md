# Mouse — TIPS at 1 MHz on the middle of the cerebellum

The first mouse case on this branch: a whole-skull transparency map for a **cerebellum**
target in the Maga (UW 4K) dry-skull microCT, and the **TIPS** bowl (Philips Therapeutic
Imaging Probe and Sonication; 80 mm radius of curvature, 20.5–46 mm annular aperture,
8 rings) seated on it at **1 MHz**.

```bash
GPU=0 python run_mouse_tips_cerebellum.py        # solve + extract + place + report
```

Inputs come from the TUBA mouse pipeline (`tuba.species.mouse`), which already stages this
skull and its Allen CCFv3 registration — nothing new is scanned or registered here.

## The target

The middle of the cerebellum is the **centroid of the Allen CCFv3 cerebellum** (every
descendant of `CB`, id 512, in the Allen structure graph) warped into the skull's own
frame, at **maga RAS (−0.06, −11.60, 1.41) mm**: midline, 4.5 mm caudal of the brain
center the whole-skull baseline uses. The warped annotation recovers 40.8 mm³ of
cerebellum, and its whole-brain centroid agrees with the independently curated
endocranial-cavity centroid to **0.05 mm**, which is the check that the registration is
sound. The script recomputes the centroid at run time and asserts it against the recorded
constant.

## Two things a mouse changes

**Points per wavelength is set by bone thickness, not by the wavelength.** At 1 MHz the
water wavelength is 1.54 mm, so the human default of 6 ppw would give dx = 0.257 mm — but
the mouse calvaria is only ~0.2–0.5 mm thick, i.e. *one voxel*. This case runs **12 ppw
(dx = 0.128 mm)**, resolving the bone in 2–4 voxels; the domain is small enough
(252 × 292 × 212) that the solve still takes seconds. Human runs need no such bump because
their skull is ~7 mm thick.

**The placement footprint comes from the cone half-angle, not the aperture radius.**
`TransducerSpec.to_bowl_constraints` sets `bowl_radius_mm = aperture/2`, which is right for
a human — the skull sits about one focal length away, so the beam cone crosses it at
roughly the aperture radius — and wrong by more than an order of magnitude here:

| | radius on the skull | footprint |
|---|---|---|
| naive aperture radius | 46 mm | **100 %** of the skull |
| cone-derived, `r·sin(θ½)` | 3.4 mm | 6 % of the skull |

The mouse skull sits ~6 mm from the cerebellum while the TIPS focal length is 80 mm, so
the cone crosses it at 3.4 mm, not 46 mm. With the naive radius every candidate window
contains the entire 25 mm head, the window score goes flat, and the "optimum" it returns is
arbitrary (it lands on the opposite side of the head at a worse 32° incidence). The script
derives the footprint with `footprint_radius_mm()`; the packaged default is left alone so
human results do not move.

## Result

| | |
|---|---|
| window | maga RAS (−1.6, −12.9, 2.0) mm — interparietal/occipital bone, directly over the cerebellum |
| window-to-target | 2.1 mm (the mouse brain nearly touches the skull) |
| incidence | 24.6° (inside the 35° acceptance angle) |
| access | 54 % of 4π of incidence-legal, well-transmitting skull |
| placement tolerance | the ≥90 %-objective lobe extends 4 mm |
| transparency spread | 18.3 dB between the 5th and 95th percentile patch |

The 18 dB spread is the point worth keeping: the mouse skull is *not* uniformly transparent
at 1 MHz even though the bone is far thinner than a wavelength (λ_bone = 2.9 mm, bone
≈ 0.3 mm) — sutures, the interparietal plate, and the basicranium still differ by more than
an order of magnitude in delivered intensity.

## Caveats

- **The focus is longer than the cerebellum.** At f/0.87 and 1 MHz the TIPS focal spot is
  ~1.3 mm laterally but ~8 mm axially, against a cerebellum that spans 5.6 mm
  antero-posteriorly. Lateral targeting is selective; axial is not.
- **The central hole is not modelled.** `TransducerSpec` describes the TIPS as a filled
  annular bowl; the 20.5 mm inner radius affects the drive and side-lobe structure, not the
  window search (which depends only on the cone half-angle).
- **Dry skull.** The specimen has no scalp or soft tissue, and the intensity ramp maps the
  (air-filled) cranial cavity to water — the standard choice for this dataset, but it means
  the model is a skull-in-water problem, not an in-vivo one.
- The report writes coordinates in the skull's own frame (`maga_aligned_native_ras_mm`) and
  omits the EEG 10-20 overlay, which is human-only; see the frame gating in `report.py`.
