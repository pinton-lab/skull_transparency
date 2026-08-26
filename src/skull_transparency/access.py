"""Which surface patches a transducer may actually be placed on.

:func:`place_bowl` maximises delivered coupling over every patch of the transparency map.
That is correct physics and the wrong answer on its own, because the map covers bone a
transducer can never be seated against: the rim of the foramen magnum, the skull base under
the pharynx and neck, the inside of an orbit, and any window whose beam would first cross
the mandible or a zygomatic arch. This module builds the boolean ``legal_mask`` that
:class:`~skull_transparency.placement.BowlConstraints` accepts, from criteria measured on
the subject's own sound-speed volume rather than hand-drawn per species.

Three criteria, each independently switchable:

**1. Beam layer count** (``max_layers``) -- march outward from the target along the patch's
own radial to the transducer standoff and count DISTINCT bone layers. A clean calvarial
window crosses exactly one. Two or more means something else is in front of the window --
mandible, zygomatic arch, tympanic bulla, the far table of a pneumatised sinus -- so the
bowl would fire through it. This is what rules out the ventral approaches on a small skull,
where the vault-shaped intuition silently fails.

**2. Minimum bone** (``min_bone_mm``) -- a patch whose radial crosses almost no bone is
sitting on the lip of an opening; the reciprocity argument still credits it with the energy
that came straight out of the hole. This is the classic foramen leak.

**3. Open-aperture exclusion** (``open_pad_deg``, ``neck_cone_deg``) -- the leak above is
angular, not per-patch: the map has NO patch inside the foramen magnum (there is no bone
there to carry one), so a per-patch test alone cannot see it. :func:`escape_directions`
casts rays over the whole sphere and finds the solid angle that leaves the head without
meeting bone at all -- the foramen magnum, the orbital fissures, the basicranial gap the
neck occupies. Patches within ``open_pad_deg`` of any escaping ray are dropped (the rim),
and ``neck_cone_deg`` drops a cone about the axis of EVERY opening bigger than
``neck_min_fraction`` of the sphere -- what lies beyond an opening is the animal's body, so
none of them can be approached through. Guarding only the largest is not enough: this
Saimiri skull has the foramen magnum caudally (1.4% of the sphere) AND a ventral
basicranial gap under the pharynx (1.2%), and a bowl aimed through the second one scores
well while pointing straight at the animal's throat. Nothing here is species-specific: the
openings are measured, not assumed.

**4. Cap clearance** (``cap_roc_mm``, ``cap_aperture_mm``) -- seat the actual dish at the
standoff, aimed at the target, and require that no point of it lands in bone. A window can be optically ideal and still unreachable because the bowl would collide
with a zygomatic arch or the far side of the head. On a small skull the dish is a large
fraction of the head, so this is usually the binding constraint; on a human vault it costs
nothing. This is the criterion that ``cap in-bone`` reports live in the positioning tool.

Typical use::

    mask, info = access_mask(tmap, bundle, standoff_mm=35.0, neck_cone_deg=40.0,
                             cap_roc_mm=35.0, cap_aperture_mm=30.0)
    pl = place_bowl(tmap, BowlConstraints(..., legal_mask=mask))
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _bundle_bits(source, bone_threshold=None):
    """(c volume, target voxel, dx_mm, bone_threshold, R_world_from_grid) from a bundle."""
    from .bundle import FieldBundle, load_bundle
    b = source if isinstance(source, FieldBundle) else load_bundle(source)
    reg = b.registration
    thr = float(bone_threshold if bone_threshold is not None
                else b.physics.get("bone_threshold", 2200.0))
    return (np.asarray(b.skull_c()), np.asarray(reg.target_fullres_voxel, float),
            float(reg.dx_mm), thr, np.asarray(reg.R_mni_to_sim, float))


def _march(c, thr, origin, dirs, n_steps):
    """Sample ``c > thr`` along ``dirs`` from ``origin``. Returns a (n_steps, M) bool array;
    samples outside the grid count as non-bone (the ray has left the head)."""
    shp = np.asarray(c.shape)
    dirs = np.atleast_2d(np.asarray(dirs, float))
    steps = np.arange(n_steps, dtype=float)[:, None, None]
    P = np.asarray(origin, float)[None, None, :] + steps * dirs[None, :, :]
    q = np.rint(P).astype(np.int32)
    inside = np.all((q >= 0) & (q < shp), axis=2)
    np.clip(q, 0, shp - 1, out=q)
    return (c[q[..., 0], q[..., 1], q[..., 2]] > thr) & inside


def _profile(bone):
    """(bone_steps, n_layers) per column of a (n_steps, M) bool ray-sample array."""
    n_layers = (np.diff(bone.astype(np.int8), axis=0) == 1).sum(0) + bone[0].astype(np.int64)
    return bone.sum(0), n_layers


def fibonacci_directions(n: int = 4000) -> np.ndarray:
    """``n`` near-uniform unit directions on the sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    gold = np.pi * (1.0 + 5.0 ** 0.5)
    return np.stack([np.cos(gold * i) * np.sin(phi),
                     np.sin(gold * i) * np.sin(phi), np.cos(phi)], axis=1)


@dataclass
class Opening:
    """A connected patch of solid angle that leaves the head without crossing bone."""
    axis_grid: np.ndarray        # (3,) mean unit direction from the target, grid frame
    axis_world: np.ndarray       # (3,) the same direction in the bundle's world frame
    n_dirs: int                  # sampled rays in this cluster
    solid_angle_sr: float        # 4*pi * n_dirs / n_total
    fraction: float              # n_dirs / n_total
    half_angle_deg: float        # angular radius: max angle of a member from the axis


def escape_directions(source, *, bone_threshold=None, n_dirs: int = 4000,
                      reach_mm: float = 45.0, link_deg: float = 6.0):
    """Find the solid angle that escapes the head from the target without meeting bone.

    Casts ``n_dirs`` rays from the bundle's target out to ``reach_mm`` and keeps those that
    never sample ``c > bone_threshold``. Escaping rays are clustered by angular
    single-linkage (``link_deg``) into :class:`Opening` s, largest first -- on a skull in
    anatomical pose the largest is the foramen magnum.

    Returns ``(dirs, escaped, openings)``: the sampled directions (grid frame), the boolean
    escape flag, and the cluster list."""
    c, tgt, dx_mm, thr, R = _bundle_bits(source, bone_threshold)
    dirs = fibonacci_directions(n_dirs)
    bone = _march(c, thr, tgt, dirs, int(reach_mm / dx_mm) + 1)
    escaped = ~bone.any(axis=0)

    openings: list[Opening] = []
    idx = np.where(escaped)[0]
    if len(idx):
        D = dirs[idx]
        cos_link = np.cos(np.deg2rad(link_deg))
        adj = (D @ D.T) >= cos_link                      # angular single-linkage graph
        seen = np.zeros(len(idx), bool)
        for s in range(len(idx)):
            if seen[s]:
                continue
            comp, stack = [], [s]
            seen[s] = True
            while stack:                                  # flood fill
                k = stack.pop()
                comp.append(k)
                nxt = np.where(adj[k] & ~seen)[0]
                seen[nxt] = True
                stack.extend(nxt.tolist())
            comp = np.asarray(comp)
            ax = D[comp].mean(0)
            ax = ax / (np.linalg.norm(ax) or 1.0)
            half = float(np.degrees(np.arccos(np.clip(D[comp] @ ax, -1.0, 1.0))).max())
            openings.append(Opening(axis_grid=ax, axis_world=R.T @ ax, n_dirs=len(comp),
                                    solid_angle_sr=4.0 * np.pi * len(comp) / n_dirs,
                                    fraction=len(comp) / n_dirs, half_angle_deg=half))
        openings.sort(key=lambda o: -o.n_dirs)
    return dirs, escaped, openings


def cap_clearance(tmap, source, *, standoff_mm: float, roc_mm: float, aperture_mm: float,
                  bone_threshold=None, density: float = 0.06, max_bone_pts: int = 0,
                  allow_off_grid: bool = True, candidates=None):
    """Can the physical bowl actually SIT on each patch?

    Seats the cap at ``standoff_mm`` along each patch's radial, aimed at the target, and
    counts cap points that fall inside bone or outside the grid. A window can be optically
    perfect and still be unreachable because the dish would collide with a zygomatic arch,
    the mandible or the other side of the head -- the constraint that actually rules out
    ventral approaches on a small skull, where the bowl is a large fraction of the head.

    Only cap points landing in BONE count as a collision. Points that fall outside the grid
    are free space, not obstruction: the domain is sized to the head plus a small surround,
    so a dish seated a focal length out routinely overhangs it, and counting that as a
    collision would reject windows for the shape of the simulation box rather than for
    anatomy. Set ``allow_off_grid=False`` to require the dish inside the modelled domain.

    ``candidates`` (indices) restricts the test to patches that passed the cheaper criteria.
    Returns ``(ok, n_bone, n_off_grid)`` over ALL patches (untested ones are ``True``/0)."""
    from .transducer import cap_directions
    c, tgt, dx_mm, thr, _R = _bundle_bits(source, bone_threshold)
    rhat = np.asarray(tmap.rhat, float)
    n_pat = len(rhat)
    idx = np.arange(n_pat) if candidates is None else np.asarray(candidates, int)

    half = float(np.degrees(np.arcsin((aperture_mm / 2.0) / roc_mm)))
    roc_vox = roc_mm / dx_mm
    # canonical cap about +z: points relative to the FOCUS, so apex = focus - roc*aim
    tmpl = roc_vox * cap_directions(np.array([0.0, 0.0, 1.0]), roc_vox, half, density)

    ok = np.ones(n_pat, bool)
    n_bone = np.zeros(n_pat, np.int64)
    n_oob = np.zeros(n_pat, np.int64)
    shp = np.asarray(c.shape)
    for s in range(0, len(idx), 2000):
        sub = idx[s:s + 2000]
        aim = -rhat[sub]                                   # apex -> target
        # The cap axis points OUTWARD (= -aim = rhat), matching build_cap's
        # ``cap_directions(-aim, ...)``. Building it about +aim instead puts the dish on the
        # far side of the head, where it silently reports clearance for the wrong window.
        z = rhat[sub]
        a = np.tile(np.array([0.0, 0.0, 1.0]), (len(sub), 1))
        flip = np.abs(z[:, 2]) > 0.9
        a[flip] = np.array([1.0, 0.0, 0.0])
        x = np.cross(a, z); x /= np.linalg.norm(x, axis=1, keepdims=True)
        y = np.cross(z, x)
        Rm = np.stack([x, y, z], axis=2)                   # (B,3,3), columns = new basis
        focus = tgt[None, :] + (standoff_mm - roc_mm) / dx_mm * (-aim)   # apex + roc*aim
        P = focus[:, None, :] + np.einsum("bij,kj->bki", Rm, tmpl)
        q = np.rint(P).astype(np.int32)
        inside = np.all((q >= 0) & (q < shp), axis=2)
        np.clip(q, 0, shp - 1, out=q)
        hit = (c[q[..., 0], q[..., 1], q[..., 2]] > thr) & inside
        n_bone[sub] = hit.sum(1)
        n_oob[sub] = (~inside).sum(1)
    bad = n_bone if allow_off_grid else n_bone + n_oob
    ok[idx] = bad[idx] <= int(max_bone_pts)
    return ok, n_bone, n_oob


@dataclass
class AccessInfo:
    """What each criterion cost, for the report and for sanity."""
    n_total: int
    n_legal: int
    dropped_layers: int = 0
    dropped_thin: int = 0
    dropped_open: int = 0
    dropped_neck: int = 0
    dropped_cap: int = 0
    openings: list = field(default_factory=list)
    neck_openings: list = field(default_factory=list)
    escape_fraction: float = 0.0
    bone_mm: np.ndarray | None = None
    n_layers: np.ndarray | None = None
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        o = (f"; largest opening {self.openings[0].fraction*100:.1f}% of solid angle "
             f"(half-angle {self.openings[0].half_angle_deg:.0f} deg)") if self.openings else ""
        return (f"{self.n_legal}/{self.n_total} patches accessible "
                f"({100.0*self.n_legal/max(self.n_total,1):.1f}%) — dropped "
                f"{self.dropped_layers} multi-layer, {self.dropped_thin} thin-bone, "
                f"{self.dropped_open} open-aperture rim, {self.dropped_neck} neck cone, "
                f"{self.dropped_cap} no cap clearance"
                f"{o}")


def access_mask(tmap, source, *, standoff_mm: float, bone_threshold=None,
                max_layers: int | None = 1, min_bone_mm: float | None = 0.3,
                open_pad_deg: float | None = 10.0, neck_cone_deg: float | None = None,
                cap_roc_mm: float | None = None, cap_aperture_mm: float | None = None,
                cap_density: float = 0.06, cap_max_bone_pts: int = 0,
                cap_allow_off_grid: bool = True, neck_min_fraction: float = 0.005,
                n_dirs: int = 4000, reach_mm: float | None = None, link_deg: float = 6.0):
    """Boolean ``legal_mask`` over ``tmap``'s patches, plus an :class:`AccessInfo`.

    ``standoff_mm`` is how far out the transducer face sits (the bowl ROC for a
    focus-on-target seating); the layer count is taken along the patch's radial out to that
    distance, which is the path the beam would take. Pass ``None`` to any criterion to skip
    it. See the module docstring for what each one rules out."""
    c, tgt, dx_mm, thr, R = _bundle_bits(source, bone_threshold)
    rhat = np.asarray(tmap.rhat, float)
    n_pat = len(rhat)
    reach = float(reach_mm if reach_mm is not None else standoff_mm)

    bone = _march(c, thr, tgt, rhat, int(reach / dx_mm) + 1)
    bone_steps, n_layers = _profile(bone)
    bone_mm = bone_steps * dx_mm

    legal = np.ones(n_pat, bool)
    info = AccessInfo(n_total=n_pat, n_legal=n_pat, bone_mm=bone_mm, n_layers=n_layers,
                      params=dict(standoff_mm=standoff_mm, max_layers=max_layers,
                                  min_bone_mm=min_bone_mm, open_pad_deg=open_pad_deg,
                                  neck_cone_deg=neck_cone_deg, bone_threshold=thr,
                                  neck_min_fraction=neck_min_fraction,
                                  cap_roc_mm=cap_roc_mm, cap_aperture_mm=cap_aperture_mm,
                                  reach_mm=reach, n_dirs=n_dirs, link_deg=link_deg))

    if max_layers is not None:
        drop = n_layers > int(max_layers)
        info.dropped_layers = int((drop & legal).sum())
        legal &= ~drop
    if min_bone_mm is not None:
        drop = bone_mm < float(min_bone_mm)
        info.dropped_thin = int((drop & legal).sum())
        legal &= ~drop

    if open_pad_deg is not None or neck_cone_deg is not None:
        dirs, escaped, openings = escape_directions(
            source, bone_threshold=thr, n_dirs=n_dirs, reach_mm=reach, link_deg=link_deg)
        info.openings = openings
        info.escape_fraction = float(escaped.mean())
        if escaped.any():
            if open_pad_deg is not None:
                # nearest escaping ray to each patch direction, in chunks (n_pat x n_esc)
                E = dirs[escaped]
                cos_pad = np.cos(np.deg2rad(open_pad_deg))
                near = np.zeros(n_pat, bool)
                for s in range(0, n_pat, 20000):
                    near[s:s + 20000] = (rhat[s:s + 20000] @ E.T).max(1) >= cos_pad
                info.dropped_open = int((near & legal).sum())
                legal &= ~near
            if neck_cone_deg is not None and openings:
                # A cone about EVERY significant opening, not just the largest. What lies
                # beyond an opening is the animal's body: the neck through the foramen
                # magnum, the pharynx and throat through the basicranial gap. Guarding only
                # the biggest one leaves the second route wide open -- on this Saimiri skull
                # the caudal foramen (1.4% of the sphere) and the ventral basicranial gap
                # (1.2%) are separate clusters, and the window escapes through the second.
                big = [o for o in openings if o.fraction >= float(neck_min_fraction)]
                drop = np.zeros(n_pat, bool)
                for o in big:
                    drop |= (rhat @ o.axis_grid) >= np.cos(np.deg2rad(neck_cone_deg))
                info.neck_openings = big
                info.dropped_neck = int((drop & legal).sum())
                legal &= ~drop

    if cap_roc_mm and cap_aperture_mm:
        ok, _n_bone, _n_oob = cap_clearance(tmap, source, standoff_mm=standoff_mm,
                                            roc_mm=cap_roc_mm, aperture_mm=cap_aperture_mm,
                                            bone_threshold=thr, density=cap_density,
                                            max_bone_pts=cap_max_bone_pts,
                                            allow_off_grid=cap_allow_off_grid,
                                            candidates=np.where(legal)[0])
        info.dropped_cap = int((~ok & legal).sum())
        legal &= ok

    info.n_legal = int(legal.sum())
    return legal, info
