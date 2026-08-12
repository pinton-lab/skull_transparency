"""napari plugin (npe2) contributions: open Field Bundles as layers, and a gallery
widget that fetches a precomputed target map and shows it with its placement.

Everything here is desktop-side post-processing on a precomputed bundle — no GPU and no
solver. napari/magicgui are imported lazily so the base package stays import-light.
"""
from __future__ import annotations

from pathlib import Path


def _bundle_dir_for(path) -> Path | None:
    """Resolve a reader path (bundle dir, bundle.json, or *.skullbundle.zip) to a
    Field Bundle directory, or None if it is not one of ours."""
    from .gallery import ZIP_SUFFIX, _unzip_bundle, cache_dir
    p = Path(path)
    if p.name == "bundle.json":
        return p.parent
    if p.is_dir() and (p / "bundle.json").exists():
        return p
    if p.name.endswith(ZIP_SUFFIX) and p.exists():
        return _unzip_bundle(p, cache_dir() / p.name[: -len(ZIP_SUFFIX)])
    return None


def napari_get_reader(path):
    """npe2 reader hook: accept Field Bundle directories / bundle.json / bundle zips."""
    if isinstance(path, (list, tuple)):
        path = path[0]
    return _read_bundle if _bundle_dir_for(path) is not None else None


def _read_bundle(path):
    """Load a bundle, compute its transparency map + default placement, return layers."""
    import skull_transparency as st
    from .position_tool import transparency_layers
    bdir = _bundle_dir_for(path if not isinstance(path, (list, tuple)) else path[0])
    tmap = st.compute_transparency_map(st.load_bundle(bdir))
    try:
        pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=60.0))
    except Exception:                      # a map is still viewable without a placement
        pl = None
    return transparency_layers(tmap, pl, name=Path(bdir).parent.name or "transparency")


def explore_widget():
    """npe2 widget: pick a named gallery target, fetch its precomputed map, show it."""
    from magicgui import magicgui
    from . import gallery

    names = sorted(gallery.list_targets())

    @magicgui(call_button="Open map", target={"choices": names},
              focal_length_mm={"min": 20.0, "max": 150.0})
    def open_gallery_target(viewer: "napari.Viewer",  # noqa: F821 - napari injects it
                            target: str = names[0] if names else "",
                            focal_length_mm: float = 60.0):
        import skull_transparency as st
        from .position_tool import transparency_layers
        bdir = gallery.fetch(target)
        tmap = st.compute_transparency_map(st.load_bundle(bdir))
        pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=focal_length_mm))
        for data, kwargs, _kind in transparency_layers(tmap, pl, name=target):
            viewer.add_points(data, **kwargs)
        viewer.dims.ndisplay = 3

    return open_gallery_target
