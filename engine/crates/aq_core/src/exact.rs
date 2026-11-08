//! Exact geometric predicates (Shewchuk, through the `robust` crate). Every crate uses these
//! wrappers so the sign conventions are fixed in one place.

use crate::mesh::{V2, V3};

/// Positive when a, b, c are in CCW order, negative when CW, zero when collinear (exact sign).
pub fn orient2d(a: V2, b: V2, c: V2) -> f64 {
    robust::orient2d(
        robust::Coord { x: a[0], y: a[1] },
        robust::Coord { x: b[0], y: b[1] },
        robust::Coord { x: c[0], y: c[1] },
    )
}

/// Shewchuk orient3d: positive when d lies below the plane through a, b, c, where "below"
/// means a, b, c appear CCW when seen from above, i.e. the sign of `(a-d).((b-d) x (c-d))`.
/// Exact sign.
pub fn orient3d(a: V3, b: V3, c: V3, d: V3) -> f64 {
    robust::orient3d(
        robust::Coord3D { x: a[0], y: a[1], z: a[2] },
        robust::Coord3D { x: b[0], y: b[1], z: b[2] },
        robust::Coord3D { x: c[0], y: c[1], z: c[2] },
        robust::Coord3D { x: d[0], y: d[1], z: d[2] },
    )
}

/// Exact sign of [`orient2d`] as -1, 0 or +1.
pub fn orient2d_sign(a: V2, b: V2, c: V2) -> i8 {
    let o = orient2d(a, b, c);
    if o > 0.0 {
        1
    } else if o < 0.0 {
        -1
    } else {
        0
    }
}
