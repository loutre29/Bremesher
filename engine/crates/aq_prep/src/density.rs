//! Density map transfer onto the proxy (DESIGN 6.4 API, 6.5 rho).

use aq_core::NONE;

use crate::{ComponentMesh, Proxy};

/// Lower and upper clamp of an input density multiplier.
pub const DENSITY_CLAMP: (f64, f64) = (1e-3, 1e3);

/// Sets `px.density[v] = rho(v)^strength`, where rho is the user density map (`density`, one
/// multiplier per **input** vertex, reached through `src.vert_src`) interpolated with
/// `px.src_bary` in the component face `px.src_face[v]`. Multipliers are clamped to
/// [1e-3, 1e3]; a non-finite or non-positive value, a missing entry, an out-of-range source
/// face or a missing barycentric entry (a short `px.src_bary`) reads as 1. `None`, or a
/// non-finite strength, gives 1 everywhere. The exponent is
/// applied here, so aq_curv must use `px.density` as rho without raising it again.
pub fn transfer_density(px: &mut Proxy, src: &ComponentMesh, density: Option<&[f32]>, strength: f64) {
    let nv = px.mesh.pos.len();
    px.density = vec![1.0; nv];
    let Some(d) = density else { return };
    if !strength.is_finite() {
        return;
    }
    let read = |cv: u32| -> f64 {
        let iv = src.vert_src.get(cv as usize).copied().unwrap_or(NONE);
        match d.get(iv as usize) {
            Some(&x) if x.is_finite() && x > 0.0 => (x as f64).clamp(DENSITY_CLAMP.0, DENSITY_CLAMP.1),
            _ => 1.0,
        }
    };
    for v in 0..nv {
        let f = px.src_face.get(v).copied().unwrap_or(NONE);
        let Some(t) = src.mesh.tris.get(f as usize) else { continue };
        let Some(&b) = px.src_bary.get(v) else { continue };
        let mut rho = 0.0;
        let mut w = 0.0;
        for k in 0..3 {
            let bk = b[k].max(0.0);
            rho += bk * read(t[k]);
            w += bk;
        }
        let rho = if w > 0.0 { rho / w } else { 1.0 };
        px.density[v] = rho.powf(strength);
    }
}
