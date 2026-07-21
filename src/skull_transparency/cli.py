"""``skull-transparency`` — turn a skull map + target + transducer into a
transparency map, a positioning score, and a placement.json. The user-facing glue
over the library.

Subcommands::

  skull-transparency prepare --c-map c.npy --affine A.npy --target 30,30,30 \
      --transducer ctx500.json --approach 0,0,1 --out run/        # -> sim tree (.dat inputs)
  skull-transparency prepare --c-map c.npy --center \
      --transducer ctx500.json --out run/                         # brain-center whole-skull (no target)

  # then run the CUDA solve on the sim tree (see README), e.g.:
  #   python -m skull_transparency.sim outward --sim run --out run --run
  skull-transparency extract --run run/outward --sim run --out run/bundle   # -> Field Bundle

  skull-transparency place        --bundle run/bundle --out result/   # -> the 3 placement outputs
  skull-transparency transparency --bundle run/bundle --out map.png   # -> whole-skull transparency figure
  skull-transparency position     --bundle run/bundle --out fig.png   # -> placement preview (--interactive: napari)
  skull-transparency run ...      # = prepare, then prints the solve/extract/place chain to run next

``prepare`` needs no GPU (it only writes solver inputs). ``place`` consumes a
post-solve Field Bundle and emits ``surface_map.npz`` (transparency map),
``score.json`` (positioning score), and ``placement.json``; ``transparency`` renders
the 1/r^2-corrected whole-skull map (the brain-center baseline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---- small loaders ---------------------------------------------------------

def _parse_vec(s):
    return np.array([float(x) for x in str(s).replace(",", " ").split()], float)


def _load_volume(path):
    """Return (array, affine_or_None). ``.npy`` -> (array, None); ``.nii``/``.nii.gz``
    -> (data, sform) via nibabel (lazy import)."""
    p = str(path)
    if p.endswith(".npy"):
        return np.load(p), None
    if p.endswith(".nii") or p.endswith(".nii.gz"):
        import nibabel as nib                       # optional; only for NIfTI input
        img = nib.load(p)
        return np.asarray(img.dataobj, dtype=float), np.asarray(img.affine, float)
    raise ValueError(f"unsupported map format: {p} (use .npy or .nii/.nii.gz)")


def _resolve_affine(args, vol_affine):
    if args.affine:
        return np.load(args.affine)
    if vol_affine is not None:
        return vol_affine
    raise SystemExit("no affine: pass --affine A.npy, or give a NIfTI c-map (which carries one).")


def _load_transducer(spec_arg):
    """Build a TransducerSpec from a JSON file/string. ``{"preset": "ctx500", ...}``
    dispatches to ``TransducerSpec.ctx500(**rest)``; otherwise the dict is the kwargs."""
    from .transducer_spec import TransducerSpec
    if Path(spec_arg).exists():
        text = Path(spec_arg).read_text()
    elif spec_arg.strip().startswith("{"):
        text = spec_arg                                   # inline JSON object string
    else:
        raise SystemExit(f"--transducer: not a readable file and not inline JSON: {spec_arg!r} "
                         "(pass a path to a .json file, or a JSON object string).")
    d = json.loads(text)
    preset = d.pop("preset", None)
    if preset == "ctx500":
        return TransducerSpec.ctx500(**d)
    if preset:
        raise SystemExit(f"unknown transducer preset {preset!r}")
    return TransducerSpec(**d)


# ---- subcommands -----------------------------------------------------------

def _cmd_prepare(args):
    from .sim.prepare import build_run_from_medium, build_brain_center_run
    c, aff = _load_volume(args.c_map)
    rho = _load_volume(args.rho_map)[0] if args.rho_map else None
    alpha = _load_volume(args.alpha_map)[0] if args.alpha_map else None
    affine = _resolve_affine(args, aff)
    spec = _load_transducer(args.transducer)
    if args.center:
        bc_kwargs = {}
        if args.bone_threshold is not None:
            bc_kwargs["bone_threshold"] = args.bone_threshold        # non-human/thin-bone speed
        if args.center_mm:
            bc_kwargs["center_phys_mm"] = _parse_vec(args.center_mm)  # explicit center (skip atlas/hole-fill)
        if args.surround_mm is not None:
            bc_kwargs["surround_mm"] = args.surround_mm               # water margin -> shrinks the grid/GPU memory
        out = build_brain_center_run(
            c, affine, spec, args.out, rho_map=rho, alpha_map=alpha,
            input_frame=args.input_frame, **bc_kwargs)
        print(f"wrote brain-center sim tree {out}  (omnidirectional source at the brain center; "
              f"solve, extract, then `transparency` the bundle)")
        return 0
    if not args.target:
        raise SystemExit("prepare needs --target (world mm) [+ --approach], or --center for a "
                         "brain-center whole-skull run.")
    target = _parse_vec(args.target)
    approach = _parse_vec(args.approach) if args.approach else None
    out = build_run_from_medium(
        c, affine, target, spec, args.out, rho_map=rho, alpha_map=alpha,
        input_frame=args.input_frame, approach=approach,
        standoff_mm=args.standoff_mm,
        surround_mm=(90.0 if args.surround_mm is None else args.surround_mm))
    print(f"wrote sim tree {out}  (now run the solver, then `place` the bundle)")
    return 0


def _cmd_transparency(args):
    import skull_transparency as st
    bundle = st.load_bundle(args.bundle)
    thr = (args.bone_threshold if args.bone_threshold is not None
           else float(bundle.physics.get("bone_threshold", 2200.0)))   # bundle carries the medium's cutoff
    opts = st.TransparencyOptions(distance_correct=not args.no_distance_correct, bone_threshold=thr)
    tmap = st.compute_transparency_map(bundle, options=opts, log=(print if args.verbose else None))
    if args.save_npz:
        tmap.to_npz(args.save_npz)
    out = args.out or "transparency.png"
    st.render_transparency_surface(tmap, out, title=args.title)
    dc = "raw (no 1/r^2)" if args.no_distance_correct else "1/r^2-corrected"
    print(f"transparency [{dc}]: {len(tmap.surf_vox)} patches, peak {tmap.Ipk_Wcm2.max():.3g} W/cm^2; "
          f"wrote {out}")
    return 0


def _cmd_place(args):
    import skull_transparency as st
    from .score import PositioningScore
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bundle = st.load_bundle(args.bundle)
    tmap = st.compute_transparency_map(bundle, log=(print if args.verbose else None))
    tmap.to_npz(out / "surface_map.npz")

    if args.transducer:
        bc = _load_transducer(args.transducer).to_bowl_constraints(focal_length_mm=args.focal_length)
    else:
        bc = st.BowlConstraints(focal_length_mm=args.focal_length or 60.0)
    pl = st.place_bowl(tmap, bc)

    world_frame = getattr(bundle.registration, "world_frame", "mni_ras_mm")
    (out / "placement.json").write_text(
        json.dumps(st.to_placement_dict(pl, target_name=args.target_name,
                                        world_frame=world_frame), indent=1))
    score = PositioningScore.from_placement(pl, target_name=args.target_name or "target")
    score.to_json(out / "score.json")

    print(f"transparency: {len(tmap.surf_vox)} patches, peak {tmap.Ipk_Wcm2.max():.3f} W/cm^2")
    print(f"score {score.normalized:.3f} (focal-pressure proxy {score.focal_pressure_proxy:.3g}), "
          f"incidence {score.incidence_deg:.1f} deg")
    print(f"wrote {out/'surface_map.npz'}, {out/'score.json'}, {out/'placement.json'}")
    return 0


def _cmd_extract(args):
    from .sim.extract import extract_bundle
    kw = {"n_out": args.n_out}
    if args.bone_threshold is not None:
        kw["bone_threshold"] = args.bone_threshold          # recorded into bundle physics
    out = extract_bundle(args.run, args.out, args.sim, **kw)
    print(f"wrote Field Bundle {out}")
    return 0


def _cmd_position(args):
    import skull_transparency as st
    from .position_tool import preview_placement, view_napari
    bundle = st.load_bundle(args.bundle)
    tmap = st.compute_transparency_map(bundle, log=(print if args.verbose else None))
    bc = (_load_transducer(args.transducer).to_bowl_constraints(focal_length_mm=args.focal_length)
          if args.transducer else st.BowlConstraints(focal_length_mm=args.focal_length or 60.0))
    pl = st.place_bowl(tmap, bc)
    if args.interactive:
        view_napari(tmap, pl)
        return 0
    out = args.out or "placement_preview.png"
    preview_placement(tmap, pl, out_png=out, title=args.target_name or "placement")
    print(f"wrote {out}  (score {pl.transparency_score:.3f}, incidence {pl.incidence_deg:.1f} deg)")
    return 0


def _cmd_run(args):
    rc = _cmd_prepare(args)
    if rc:
        return rc
    o = args.out
    print("\nNext: run the (GPU) CUDA solve, extract a Field Bundle, then place it:\n"
          f"  python -m skull_transparency.sim outward --sim {o} --out {o} --run\n"
          f"  skull-transparency extract --run {o}/outward --sim {o} --out {o}/bundle\n"
          f"  skull-transparency place --bundle {o}/bundle --out {o}/result\n"
          "(the solve is GPU-bound; see README.)")
    return 0


# ---- parser ----------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="skull-transparency", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_prepare(sp):
        sp.add_argument("--c-map", required=True, help="sound-speed map (.npy or .nii)")
        sp.add_argument("--rho-map", help="optional density map (.npy/.nii)")
        sp.add_argument("--alpha-map", help="optional attenuation map dB/cm/MHz (.npy/.nii); auto-enables attenuation")
        sp.add_argument("--affine", help="4x4 voxel->world-mm .npy (else taken from a NIfTI c-map)")
        sp.add_argument("--target", help="target in world mm, 'x,y,z' (omit with --center)")
        sp.add_argument("--center", action="store_true",
                        help="brain-center whole-skull run: one omnidirectional source at the brain "
                             "center, cube sized to the whole head (no --target/--approach needed)")
        sp.add_argument("--center-mm", help="with --center: explicit brain center in world mm 'x,y,z' "
                        "(else atlas CoM for an MNI frame, or the image-only intracranial centroid)")
        sp.add_argument("--bone-threshold", type=float, default=None,
                        help="bone sound-speed cutoff m/s (default 2200, the human value); lower it for "
                             "a medium whose bone is slower (a value above water/soft tissue, below bone)")
        sp.add_argument("--transducer", required=True, help="TransducerSpec JSON file or string")
        sp.add_argument("--approach", help="aim unit vector target->skin, 'x,y,z' (required until auto)")
        sp.add_argument("--input-frame", default="ras_mm", help="provenance label for the world frame")
        sp.add_argument("--standoff-mm", type=float, default=20.0)
        sp.add_argument("--surround-mm", type=float, default=None,
                        help="water margin around the head in mm; sizes the grid (smaller = less GPU "
                             "memory). Default 90 (targeted) / 25 (--center). Safe to lower to ~8-12.")
        sp.add_argument("--out", required=True, help="output sim-tree directory")

    sp = sub.add_parser("prepare", help="skull map -> sim tree (.dat solver inputs); no GPU")
    add_prepare(sp); sp.set_defaults(func=_cmd_prepare)

    sp = sub.add_parser("place", help="Field Bundle -> surface_map.npz + score.json + placement.json")
    sp.add_argument("--bundle", required=True, help="Field Bundle directory (post-solve)")
    sp.add_argument("--transducer", help="TransducerSpec JSON for the window constraints")
    sp.add_argument("--focal-length", type=float, default=None, help="bowl focal length mm (default ROC or 60)")
    sp.add_argument("--target-name", default=None)
    sp.add_argument("--verbose", action="store_true")
    sp.add_argument("--out", required=True, help="output directory")
    sp.set_defaults(func=_cmd_place)

    sp = sub.add_parser("extract", help="solved run (genout_mod.dat) + sim tree -> Field Bundle")
    sp.add_argument("--run", required=True, help="solved outward run dir (holds genout_mod.dat)")
    sp.add_argument("--sim", required=True, help="producer sim tree (meta.json + c.f32 + registration.json)")
    sp.add_argument("--n-out", type=int, default=None, help="outward frame count (default: all recorded)")
    sp.add_argument("--bone-threshold", type=float, default=None,
                    help="bone sound-speed cutoff m/s recorded into the bundle (default 2200; "
                         "use the producer's value for a non-human/thin-bone medium)")
    sp.add_argument("--out", required=True, help="output Field Bundle directory")
    sp.set_defaults(func=_cmd_extract)

    sp = sub.add_parser("position", help="Field Bundle -> placement preview figure (or --interactive napari)")
    sp.add_argument("--bundle", required=True, help="Field Bundle directory")
    sp.add_argument("--transducer", help="TransducerSpec JSON for the window constraints")
    sp.add_argument("--focal-length", type=float, default=None)
    sp.add_argument("--target-name", default=None)
    sp.add_argument("--interactive", action="store_true", help="open the napari viewer (needs a display + [viz])")
    sp.add_argument("--verbose", action="store_true")
    sp.add_argument("--out", default=None, help="output PNG (default placement_preview.png)")
    sp.set_defaults(func=_cmd_position)

    sp = sub.add_parser("transparency",
                        help="Field Bundle -> whole-skull transparency figure (1/r^2-corrected)")
    sp.add_argument("--bundle", required=True, help="Field Bundle directory (e.g. a brain-center run)")
    sp.add_argument("--out", default=None, help="output PNG (default transparency.png)")
    sp.add_argument("--title", default=None)
    sp.add_argument("--save-npz", default=None, help="also write the TransparencyMap as .npz")
    sp.add_argument("--bone-threshold", type=float, default=None,
                    help="bone sound-speed cutoff m/s for the calvarial surface "
                         "(default: the bundle's physics.bone_threshold)")
    sp.add_argument("--no-distance-correct", action="store_true",
                    help="raw peak intensity, no 1/r^2 spreading correction")
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=_cmd_transparency)

    sp = sub.add_parser("run", help="prepare (+ the solve/extract/place chain to run next)")
    add_prepare(sp); sp.set_defaults(func=_cmd_run)
    return p


def main(argv=None):
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
