//! Fill-reducing orderings for [`crate::Chol::analyze`].
//!
//! * AMD through faer.
//! * A nested dissection for large mesh-like graphs: recursive bisection by BFS level-set
//!   vertex separators from a pseudo-peripheral vertex (George's automatic nested
//!   dissection), separators thinned to the vertices that really separate, leaves ordered by
//!   AMD. No recursion: an explicit task stack (DESIGN 3.4).
//! * [`chol_flops`]: elimination tree and exact column counts of L for a given ordering, used
//!   to keep the cheaper of the two orderings.
//!
//! Everything is deterministic (sorted, no hashing).

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::SymbolicSparseColMatRef;
use faer::sparse::linalg::amd;

use aq_core::alloc::{try_vec, try_with_capacity};
use aq_core::{AqError, Ctl, ErrKind};

use crate::sparse::{Csc, err};

/// Symmetric adjacency (no diagonal) in CSR form.
pub(crate) struct Graph {
    pub xadj: Vec<usize>,
    pub adj: Vec<u32>,
}

impl Graph {
    pub fn from_lower(a: &Csc) -> Result<Graph, AqError> {
        let n = a.n();
        let (cp, ri) = (a.col_ptr(), a.row_idx());
        let mut deg = try_vec(n + 1, 0usize)?;
        for j in 0..n {
            for &i in &ri[cp[j]..cp[j + 1]] {
                if i != j {
                    deg[i + 1] += 1;
                    deg[j + 1] += 1;
                }
            }
        }
        for j in 0..n {
            deg[j + 1] += deg[j];
        }
        let mut adj = try_vec(deg[n], 0u32)?;
        let mut fill = try_vec(n + 1, 0usize)?;
        fill.copy_from_slice(&deg);
        for j in 0..n {
            for &i in &ri[cp[j]..cp[j + 1]] {
                if i != j {
                    adj[fill[i]] = j as u32;
                    fill[i] += 1;
                    adj[fill[j]] = i as u32;
                    fill[j] += 1;
                }
            }
        }
        Ok(Graph { xadj: deg, adj })
    }

    fn nbrs(&self, v: usize) -> &[u32] {
        &self.adj[self.xadj[v]..self.xadj[v + 1]]
    }
}

/// AMD ordering of the pattern of `a` (lower storage), as (fwd, inv): `fwd[k]` is the
/// original index eliminated k-th.
pub(crate) fn amd_order(a: &Csc) -> Result<(Vec<usize>, Vec<usize>), AqError> {
    let n = a.n();
    amd_of_pattern(n, a.col_ptr(), a.row_idx())
}

fn amd_of_pattern(
    n: usize,
    cp: &[usize],
    ri: &[usize],
) -> Result<(Vec<usize>, Vec<usize>), AqError> {
    let mut fwd = try_vec(n, 0usize)?;
    let mut inv = try_vec(n, 0usize)?;
    if n == 0 {
        return Ok((fwd, inv));
    }
    let s = SymbolicSparseColMatRef::new_checked(n, n, cp, None, ri);
    let req = amd::order_maybe_unsorted_scratch::<usize>(n, ri.len());
    let mut mem =
        MemBuffer::try_new(req).map_err(|_| err(ErrKind::Budget, "amd: out of memory"))?;
    amd::order_maybe_unsorted(
        &mut fwd,
        &mut inv,
        s,
        amd::Control::default(),
        MemStack::new(&mut mem),
    )
    .map_err(|e| err(ErrKind::Budget, format!("amd: {e:?}")))?;
    Ok((fwd, inv))
}

/// Columns between two `ctl.check()` calls in [`chol_flops`].
const CTL_COLS: usize = 4096;

/// Extra BFS sweeps of the pseudo-peripheral vertex search.
const PP_SWEEPS: usize = 2;

/// Nested dissection ordering (fwd, inv) of the graph, leaves of at most `leaf` vertices
/// ordered by AMD. `ctl` is checked once per task (each task costs at most a few passes over
/// its vertices and edges, well under 10 ms at 200k vertices).
///
/// A disconnected task is split into all of its components in one pass (O(task)), and
/// components of at most `leaf` vertices are batched into leaf tasks, so fragmented patterns
/// (isolated unknowns, identity rows, small blocks) cost O(n) in total.
pub(crate) fn nested_dissection(
    g: &Graph,
    leaf: usize,
    ctl: Option<&Ctl>,
) -> Result<(Vec<usize>, Vec<usize>), AqError> {
    let n = g.xadj.len() - 1;
    let leaf = leaf.max(1);
    let mut fwd = try_vec(n, usize::MAX)?;
    let mut task_mark = try_vec(n, 0u32)?; // vertex belongs to the current task
    let mut seen = try_vec(n, 0u32)?; // BFS stamp
    let mut comp_mark = try_vec(n, 0u32)?; // component split stamp (= task id)
    let mut level = try_vec(n, 0u32)?;
    let mut side = try_vec(n, 0u8)?;
    let mut loc = try_vec(n, 0usize)?;
    let (mut task_id, mut bfs_id) = (0u32, 0u32);
    let mut all: Vec<u32> = try_with_capacity(n)?;
    all.extend(0..n as u32);
    let mut stack: Vec<(Vec<u32>, usize)> = vec![(all, 0)];
    let mut queue: Vec<u32> = Vec::new();

    // BFS restricted to the current task from `root`: fills `queue` in BFS order and `level`,
    // returns the number of levels.
    let bfs = |root: u32,
               task_id: u32,
               bfs_id: &mut u32,
               task_mark: &[u32],
               seen: &mut [u32],
               level: &mut [u32],
               queue: &mut Vec<u32>|
     -> u32 {
        *bfs_id = bfs_id.wrapping_add(1);
        if *bfs_id == 0 {
            seen.fill(0);
            *bfs_id = 1;
        }
        queue.clear();
        queue.push(root);
        seen[root as usize] = *bfs_id;
        level[root as usize] = 0;
        let mut head = 0;
        let mut nlev = 1;
        while head < queue.len() {
            let v = queue[head] as usize;
            head += 1;
            let lv = level[v];
            for &u in g.nbrs(v) {
                let ui = u as usize;
                if task_mark[ui] == task_id && seen[ui] != *bfs_id {
                    seen[ui] = *bfs_id;
                    level[ui] = lv + 1;
                    nlev = nlev.max(lv + 2);
                    queue.push(u);
                }
            }
        }
        nlev
    };

    while let Some((verts, start)) = stack.pop() {
        if let Some(c) = ctl {
            c.check()?;
        }
        task_id = task_id.wrapping_add(1);
        if task_id == 0 {
            task_mark.fill(0);
            comp_mark.fill(0);
            task_id = 1;
        }
        for &v in &verts {
            task_mark[v as usize] = task_id;
        }
        let nv = verts.len();
        if nv == 0 {
            continue;
        }
        if nv <= leaf {
            order_leaf(g, &verts, start, &task_mark, task_id, &mut loc, &mut fwd)?;
            continue;
        }
        // Connected component of verts[0]; when connected, this BFS is also the first sweep
        // of the pseudo-peripheral search.
        let mut nlev = bfs(
            verts[0],
            task_id,
            &mut bfs_id,
            &task_mark,
            &mut seen,
            &mut level,
            &mut queue,
        );
        if queue.len() < nv {
            let split = Split {
                g,
                task_id,
                task_mark: &task_mark,
                leaf,
            };
            split.push_components(&verts, start, &queue, &mut comp_mark, &mut stack)?;
            continue;
        }
        // Pseudo-peripheral root: BFS again from a minimum-degree vertex of the last level
        // while the eccentricity grows (at most PP_SWEEPS more sweeps).
        let mut root = verts[0];
        for _ in 0..PP_SWEEPS {
            let last = nlev - 1;
            let cand = queue
                .iter()
                .rev()
                .take_while(|&&v| level[v as usize] == last)
                .copied()
                .min_by_key(|&v| {
                    (
                        g.nbrs(v as usize)
                            .iter()
                            .filter(|&&u| task_mark[u as usize] == task_id)
                            .count(),
                        v,
                    )
                });
            let Some(c) = cand else { break };
            let nl = bfs(
                c,
                task_id,
                &mut bfs_id,
                &task_mark,
                &mut seen,
                &mut level,
                &mut queue,
            );
            if nl < nlev {
                // Strictly worse: restore the level structure of the previous root.
                nlev = bfs(
                    root,
                    task_id,
                    &mut bfs_id,
                    &task_mark,
                    &mut seen,
                    &mut level,
                    &mut queue,
                );
                break;
            }
            let grew = nl > nlev;
            root = c;
            nlev = nl;
            if !grew {
                break;
            }
        }
        if nlev < 3 {
            order_leaf(g, &verts, start, &task_mark, task_id, &mut loc, &mut fwd)?;
            continue;
        }
        // Level sizes and the separating level: the smallest level leaving at least 30% of
        // the vertices on each side, else the median level.
        let mut cnt = try_vec(nlev as usize, 0usize)?;
        for &v in &queue {
            cnt[level[v as usize] as usize] += 1;
        }
        let mut below = 0usize;
        let mut best: Option<(usize, usize)> = None; // (size, level)
        let mut median = 1usize;
        let mut median_set = false;
        for (l, &c) in cnt.iter().enumerate() {
            let above = nv - below - c;
            if l > 0 && l + 1 < cnt.len() {
                if 10 * below >= 3 * nv && 10 * above >= 3 * nv && best.is_none_or(|b| c < b.0) {
                    best = Some((c, l));
                }
                if !median_set && 2 * (below + c) >= nv {
                    median = l;
                    median_set = true;
                }
            }
            below += c;
        }
        let sep = best.map_or(median, |b| b.1) as u32;
        // Sides: 0 = A (below), 1 = B (above), 2 = separator.
        for &v in &verts {
            let l = level[v as usize];
            side[v as usize] = if l < sep {
                0
            } else if l > sep {
                1
            } else {
                2
            };
        }
        // Thin the separator: a vertex with no neighbour in B joins A, then one with no
        // neighbour in A joins B.
        for target in [0u8, 1u8] {
            let other = 1 - target;
            for &v in &verts {
                if side[v as usize] != 2 {
                    continue;
                }
                let touches = g
                    .nbrs(v as usize)
                    .iter()
                    .any(|&u| task_mark[u as usize] == task_id && side[u as usize] == other);
                if !touches {
                    side[v as usize] = target;
                }
            }
        }
        let (mut na, mut nb) = (0usize, 0usize);
        for &v in &verts {
            match side[v as usize] {
                0 => na += 1,
                1 => nb += 1,
                _ => {}
            }
        }
        if na == 0 || nb == 0 {
            order_leaf(g, &verts, start, &task_mark, task_id, &mut loc, &mut fwd)?;
            continue;
        }
        let mut a: Vec<u32> = try_with_capacity(na)?;
        let mut b: Vec<u32> = try_with_capacity(nb)?;
        let mut pos = start + na + nb;
        for &v in &verts {
            match side[v as usize] {
                0 => a.push(v),
                1 => b.push(v),
                _ => {
                    fwd[pos] = v as usize;
                    pos += 1;
                }
            }
        }
        let sb = start + na;
        stack.push((b, sb));
        stack.push((a, start));
    }
    let mut inv = try_vec(n, usize::MAX)?;
    for (k, &v) in fwd.iter().enumerate() {
        if v >= n || inv[v] != usize::MAX {
            return Err(err(
                ErrKind::Internal,
                "nested_dissection: not a permutation",
            ));
        }
        inv[v] = k;
    }
    Ok((fwd, inv))
}

/// Component split of a disconnected nested dissection task.
struct Split<'a> {
    g: &'a Graph,
    task_id: u32,
    task_mark: &'a [u32],
    leaf: usize,
}

impl Split<'_> {
    /// Splits the task `verts` (positions `start..start + verts.len()`) into its connected
    /// components in one O(task) pass and pushes them as tasks. `first` is the component of
    /// `verts[0]`, already found. A component larger than `leaf` becomes its own task; smaller
    /// runs are batched, in order, into tasks of at most `leaf` consecutive positions (a leaf
    /// task is ordered by AMD on the union, which handles several components at once).
    fn push_components(
        &self,
        verts: &[u32],
        start: usize,
        first: &[u32],
        comp_mark: &mut [u32],
        stack: &mut Vec<(Vec<u32>, usize)>,
    ) -> Result<(), AqError> {
        let (g, task_id, task_mark, leaf) = (self.g, self.task_id, self.task_mark, self.leaf);
        let nv = verts.len();
        // All components back to back in `flat`, with their end offsets in `ends`.
        let mut flat: Vec<u32> = try_with_capacity(nv)?;
        let mut ends: Vec<usize> = Vec::new();
        for &v in first {
            comp_mark[v as usize] = task_id;
        }
        flat.extend_from_slice(first);
        ends.push(flat.len());
        for &r in verts {
            if comp_mark[r as usize] == task_id {
                continue;
            }
            comp_mark[r as usize] = task_id;
            let mut head = flat.len();
            flat.push(r);
            while head < flat.len() {
                let v = flat[head] as usize;
                head += 1;
                for &u in g.nbrs(v) {
                    let ui = u as usize;
                    if task_mark[ui] == task_id && comp_mark[ui] != task_id {
                        comp_mark[ui] = task_id;
                        flat.push(u);
                    }
                }
            }
            ends.push(flat.len());
        }
        if flat.len() != nv {
            return Err(err(
                ErrKind::Internal,
                "nested_dissection: component split lost vertices",
            ));
        }
        let mut batch: Vec<u32> = Vec::new();
        let mut batch_start = start;
        let mut pos = start;
        let mut s = 0usize;
        for &e in &ends {
            let comp = &flat[s..e];
            let c = comp.len();
            if c > leaf {
                // A batch covers consecutive positions: close it before this component.
                if !batch.is_empty() {
                    stack.push((std::mem::take(&mut batch), batch_start));
                }
                let mut own: Vec<u32> = try_with_capacity(c)?;
                own.extend_from_slice(comp);
                stack.push((own, pos));
            } else {
                if batch.len() + c > leaf {
                    stack.push((std::mem::take(&mut batch), batch_start));
                }
                if batch.is_empty() {
                    batch = try_with_capacity(leaf)?;
                    batch_start = pos;
                }
                batch.extend_from_slice(comp);
            }
            pos += c;
            s = e;
        }
        if !batch.is_empty() {
            stack.push((batch, batch_start));
        }
        Ok(())
    }
}

/// AMD on the subgraph induced by `verts`, written to `fwd[start..start + len]`.
fn order_leaf(
    g: &Graph,
    verts: &[u32],
    start: usize,
    task_mark: &[u32],
    task_id: u32,
    loc: &mut [usize],
    fwd: &mut [usize],
) -> Result<(), AqError> {
    let nv = verts.len();
    for (k, &v) in verts.iter().enumerate() {
        loc[v as usize] = k;
    }
    let mut cp = Vec::with_capacity(nv + 1);
    let mut ri = Vec::new();
    cp.push(0);
    for (j, &v) in verts.iter().enumerate() {
        let s = ri.len();
        ri.push(j);
        for &u in g.nbrs(v as usize) {
            if task_mark[u as usize] == task_id && loc[u as usize] > j {
                ri.push(loc[u as usize]);
            }
        }
        ri[s..].sort_unstable();
        cp.push(ri.len());
    }
    let (lf, _) = amd_of_pattern(nv, &cp, &ri)?;
    for (k, &l) in lf.iter().enumerate() {
        fwd[start + k] = verts[l] as usize;
    }
    Ok(())
}

/// Floating-point operation count proxy `sum_j c_j^2` of the Cholesky factor of `P A P^T`,
/// with `c_j` the exact column counts of L (diagonal included), and `nnz(L)`. The cost is
/// O(nnz(L)); `ctl` is checked every [`CTL_COLS`] columns.
pub(crate) fn chol_flops(
    a: &Csc,
    inv: &[usize],
    ctl: Option<&Ctl>,
) -> Result<(f64, usize), AqError> {
    let n = a.n();
    let check = |j: usize| -> Result<(), AqError> {
        match ctl {
            Some(c) if j.is_multiple_of(CTL_COLS) => c.check(),
            _ => Ok(()),
        }
    };
    // Permuted pattern, upper storage by column: for new column j, rows i < j.
    let (cp, ri) = (a.col_ptr(), a.row_idx());
    let mut cnt = try_vec(n + 1, 0usize)?;
    for j in 0..n {
        for &i in &ri[cp[j]..cp[j + 1]] {
            if i != j {
                let (pi, pj) = (inv[i], inv[j]);
                cnt[pi.max(pj) + 1] += 1;
            }
        }
    }
    for j in 0..n {
        cnt[j + 1] += cnt[j];
    }
    let mut up = try_vec(cnt[n], 0usize)?;
    let mut fill = try_vec(n + 1, 0usize)?;
    fill.copy_from_slice(&cnt);
    for j in 0..n {
        for &i in &ri[cp[j]..cp[j + 1]] {
            if i != j {
                let (pi, pj) = (inv[i], inv[j]);
                let (r, c) = if pi < pj { (pi, pj) } else { (pj, pi) };
                up[fill[c]] = r;
                fill[c] += 1;
            }
        }
    }
    // Elimination tree (Liu) with path compression.
    let mut parent = try_vec(n, usize::MAX)?;
    let mut ancestor = try_vec(n, usize::MAX)?;
    for j in 0..n {
        check(j)?;
        for &i0 in &up[cnt[j]..cnt[j + 1]] {
            let mut i = i0;
            while i != usize::MAX && i < j {
                let next = ancestor[i];
                ancestor[i] = j;
                if next == usize::MAX {
                    parent[i] = j;
                    break;
                }
                i = next;
            }
        }
    }
    // Row subtrees: row j of L = union of etree paths from each i (A[i][j] != 0, i < j) up
    // to j. Each visited node adds one to its column count.
    let mut colcount = try_vec(n, 1usize)?; // diagonal
    let mut mark = try_vec(n, usize::MAX)?;
    for j in 0..n {
        check(j)?;
        mark[j] = j;
        for &i0 in &up[cnt[j]..cnt[j + 1]] {
            let mut i = i0;
            while i != usize::MAX && mark[i] != j {
                mark[i] = j;
                colcount[i] += 1;
                i = parent[i];
            }
        }
    }
    let flops = colcount.iter().map(|&c| (c as f64) * (c as f64)).sum();
    let nnz = colcount.iter().sum();
    Ok((flops, nnz))
}
