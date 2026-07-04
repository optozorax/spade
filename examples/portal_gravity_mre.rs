//! MRE for the spatially-varying-sizing refinement feature (see `refinement.md` in the
//! repository root). Self-contained: all geometry is hardcoded, no external dependencies.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example portal_gravity_mre
//! ```
//!
//! # What this is
//!
//! Real triangulation input extracted from the motivating downstream project (a 2D FEM
//! Poisson solver with portals). The domain of interest is the unit square `[0,1]²`; the
//! full meshed domain is `[-10, 11]²` to emulate open-space boundary conditions. Two
//! portal polylines (horizontal segments at y = 0.2 and y = 0.8, x ∈ [0.4, 0.6]) are
//! inserted as constraint edges whose discretization must survive refinement exactly
//! (`keep_constraint_edges`).
//!
//! Two real parameter sets, both from the same scene:
//!
//! * `explore` — interactive quality, `edge_length = 0.1`. Reference (downstream
//!   project's machine): 2153 triangles total, inner ≈ 2.8 ms + outer ≈ 0.48 ms.
//! * `render` — final render quality, `edge_length = 0.01`. Reference: 46439 triangles
//!   total, inner ≈ 67 ms + outer ≈ 4 ms.
//!
//! # What the baseline does (current downstream workaround)
//!
//! Because `with_max_allowed_area` is a single global constant, the project builds TWO
//! triangulations per case and stitches them (stitching not reproduced here — it is
//! irrelevant to spade):
//!
//! 1. *Inner*: the hardcoded points (unit-square corners + portal polylines), portal
//!    edges as constraints, refined with `keep_constraint_edges` +
//!    `with_max_allowed_area(0.4 · edge_length²)`.
//! 2. *Outer*: the inner triangulation's convex-hull vertices + 4 far corners at ±10,
//!    hull ring as constraints, refined with `keep_constraint_edges` only (angle
//!    quality, no area bound). Faces whose centroid falls strictly inside `(0,1)²` are
//!    discarded (that region is covered by the inner triangulation).
//!
//! `run_baseline` reproduces this pipeline call-for-call and checks the triangle totals
//! match the downstream project bit-for-bit. **These totals must not change** when you
//! modify spade — the constant-area path is required to stay byte-identical (backwards
//! compatibility, requirement 3 of refinement.md).
//!
//! # What you should implement and test here
//!
//! Fill in [`build_single_triangulation_with_sizing`]: ONE triangulation of the whole
//! `[-10, 11]²` domain using your new sizing-function API, replacing both baseline
//! triangulations. `main` then verifies it (constraints preserved, area bound honored
//! inside the unit square, sane triangle count) and prints timings next to the baseline
//! numbers. Success criteria:
//!
//! * all checks print `ok` for both cases (exit code 0);
//! * total time is in the same ballpark as the baseline sum for each case (same order
//!   of magnitude is fine, faster is better);
//! * triangle count inside the unit square is comparable to the baseline inner count.

use std::time::{Duration, Instant};

use spade::handles::FixedVertexHandle;
use spade::{ConstrainedDelaunayTriangulation, Point2, RefinementParameters, Triangulation};
use tiny_skia::{Color, FillRule, LineCap, Paint, PathBuilder, Pixmap, Stroke, StrokeDash, Transform};

type Cdt = ConstrainedDelaunayTriangulation<Point2<f64>>;

/// The outer domain is `[0,1]²` grown by this much on every side: `[-10, 11]²`.
const OUTER_LAYER_SIZE: f64 = 10.0;
/// Hard stop for refinement, same value the downstream project uses.
const MAX_ADDITIONAL_VERTICES: usize = 1_000_000;

struct Case {
    name: &'static str,
    /// Target edge length inside the unit square; the area bound is derived from it.
    edge_length: f64,
    /// Unit-square corners + portal polyline points, in exact downstream order.
    points: &'static [(f64, f64)],
    /// Portal polyline segments (indices into `points`), inserted as constraint edges.
    edges: &'static [[usize; 2]],
    /// Baseline total (inner + kept outer) on unmodified spade 2.15.1. Must not drift.
    expected_total_triangles: usize,
    /// Reference baseline timings on the downstream project's machine (for eyeballing).
    reference_inner_ms: f64,
    reference_outer_ms: f64,
}

const CASES: &[Case] = &[
    Case {
        name: "explore (edge_length = 0.1)",
        edge_length: 0.1,
        points: EXPLORE_POINTS,
        edges: EXPLORE_EDGES,
        expected_total_triangles: 2153,
        reference_inner_ms: 2.8,
        reference_outer_ms: 0.48,
    },
    Case {
        name: "render (edge_length = 0.01)",
        edge_length: 0.01,
        points: RENDER_POINTS,
        edges: RENDER_EDGES,
        expected_total_triangles: 46439,
        reference_inner_ms: 67.0,
        reference_outer_ms: 4.0,
    },
];

/// Max allowed triangle area inside the unit square for a given target edge length.
fn max_allowed_area(edge_length: f64) -> f64 {
    0.4 * edge_length * edge_length
}

fn inside_unit_square_strict(x: f64, y: f64) -> bool {
    x > 0.0 && x < 1.0 && y > 0.0 && y < 1.0
}

fn centroid([a, b, c]: [Point2<f64>; 3]) -> (f64, f64) {
    ((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0)
}

fn triangle_area([a, b, c]: [Point2<f64>; 3]) -> f64 {
    0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs()
}

/// Snapshot of the constraint edges (as vertex-index pairs) before refinement, so we can
/// assert afterwards that refinement did not split or drop any of them. Vertex indices
/// are stable across refinement: spade only appends vertices.
fn collect_constraint_vertex_pairs(cdt: &Cdt) -> Vec<(FixedVertexHandle, FixedVertexHandle)> {
    cdt.undirected_edges()
        .filter(|edge| edge.is_constraint_edge())
        .map(|edge| {
            let [a, b] = edge.vertices();
            (a.fix(), b.fix())
        })
        .collect()
}

/// Panics if any pre-refinement constraint edge no longer exists as a direct constraint.
fn assert_constraints_preserved(
    cdt: &Cdt,
    expected: &[(FixedVertexHandle, FixedVertexHandle)],
    context: &str,
) {
    for &(a, b) in expected {
        assert!(
            cdt.exists_constraint(a, b),
            "{context}: constraint edge {} <-> {} was split or lost by refinement",
            a.index(),
            b.index(),
        );
    }
}

/// Builds and refines the fine inner triangulation exactly like the downstream project.
fn build_inner(case: &Case) -> (Cdt, Duration) {
    let points: Vec<Point2<f64>> = case
        .points
        .iter()
        .map(|&(x, y)| Point2::new(x, y))
        .collect();

    let start = Instant::now();

    let mut cdt = Cdt::bulk_load_cdt(points, vec![]).expect("inner bulk load failed");
    for &[a, b] in case.edges {
        // bulk_load_cdt preserves vertex order (there are no duplicate points), so
        // pre-load indices are valid handles.
        cdt.add_constraint_and_split(
            FixedVertexHandle::from_index(a),
            FixedVertexHandle::from_index(b),
            |v| v,
        );
    }

    let expected_constraints = collect_constraint_vertex_pairs(&cdt);

    let parameters = RefinementParameters::<f64>::new()
        .keep_constraint_edges()
        .with_max_allowed_area(max_allowed_area(case.edge_length))
        .with_max_additional_vertices(MAX_ADDITIONAL_VERTICES);
    cdt.refine(parameters);

    let elapsed = start.elapsed();

    assert_constraints_preserved(&cdt, &expected_constraints, "inner");
    (cdt, elapsed)
}

/// Builds and refines the coarse outer triangulation exactly like the downstream
/// project: convex-hull vertices of the refined inner triangulation + 4 far corners,
/// hull ring as constraint edges, angle-quality refinement only. Returns the
/// triangulation, the number of *kept* faces (centroid not strictly inside the unit
/// square), and the build time.
fn build_outer(inner: &Cdt) -> (Cdt, usize, Duration) {
    let start = Instant::now();

    let hull_edges: Vec<(usize, usize)> = inner
        .convex_hull()
        .map(|hull| (hull.from().index(), hull.to().index()))
        .collect();
    let mut hull_vertices = Vec::with_capacity(hull_edges.len());
    hull_vertices.push(hull_edges[0].0);
    for &(_, to) in &hull_edges {
        hull_vertices.push(to);
    }
    if hull_vertices.last().copied() == Some(hull_vertices[0]) {
        hull_vertices.pop();
    }
    let hull_len = hull_vertices.len();

    let mut points: Vec<Point2<f64>> = hull_vertices
        .iter()
        .map(|&i| inner.vertex(FixedVertexHandle::from_index(i)).position())
        .collect();
    let s = OUTER_LAYER_SIZE;
    points.push(Point2::new(-s, -s));
    points.push(Point2::new(-s, 1.0 + s));
    points.push(Point2::new(1.0 + s, -s));
    points.push(Point2::new(1.0 + s, 1.0 + s));

    let mut cdt = Cdt::bulk_load_cdt(points, vec![]).expect("outer bulk load failed");
    for i in 0..hull_len {
        cdt.add_constraint_and_split(
            FixedVertexHandle::from_index(i),
            FixedVertexHandle::from_index((i + 1) % hull_len),
            |v| v,
        );
    }

    let expected_constraints = collect_constraint_vertex_pairs(&cdt);

    let parameters = RefinementParameters::<f64>::new().keep_constraint_edges();
    cdt.refine(parameters);

    let elapsed = start.elapsed();

    assert_constraints_preserved(&cdt, &expected_constraints, "outer");

    // The outer triangulation covers the full convex hull, including the unit-square
    // region already meshed by the inner triangulation; those faces are discarded.
    let kept = cdt
        .inner_faces()
        .filter(|face| {
            let (cx, cy) = centroid(face.positions());
            !inside_unit_square_strict(cx, cy)
        })
        .count();

    (cdt, kept, elapsed)
}

struct BaselineResult {
    total_triangles: usize,
    inner_triangles: usize,
    total_time: Duration,
}

fn run_baseline(case: &Case, failed: &mut bool) -> BaselineResult {
    let (inner, inner_time) = build_inner(case);
    let (_outer, outer_kept, outer_time) = build_outer(&inner);

    let inner_triangles = inner.inner_faces().count();
    let total = inner_triangles + outer_kept;

    println!("  baseline (two triangulations, current workaround):");
    println!(
        "    inner: {} triangles in {:.2?} (reference machine: ~{} ms)",
        inner_triangles, inner_time, case.reference_inner_ms
    );
    println!(
        "    outer: {} kept triangles in {:.2?} (reference machine: ~{} ms)",
        outer_kept, outer_time, case.reference_outer_ms
    );
    println!(
        "    total: {} triangles in {:.2?}",
        total,
        inner_time + outer_time
    );

    if total == case.expected_total_triangles {
        println!("    reproduction check: ok (matches downstream project exactly)");
    } else {
        println!(
            "    reproduction check: FAIL — expected {} triangles, got {}.",
            case.expected_total_triangles, total
        );
        println!("    If you modified spade: the constant-area path must stay byte-identical");
        println!("    (backwards compatibility, requirement 3 of refinement.md).");
        *failed = true;
    }

    BaselineResult {
        total_triangles: total,
        inner_triangles,
        total_time: inner_time + outer_time,
    }
}

/// =====================================================================================
/// IMPLEMENT YOUR SOLUTION HERE.
///
/// Build ONE triangulation of the whole `[-10, 11]²` domain that replaces both baseline
/// triangulations, using your new spatially-varying-sizing API. Return `Some(cdt)`; the
/// caller runs the checks. Keep the rest of this file unchanged so the baseline stays a
/// faithful reproduction of the downstream project.
///
/// The intended shape (adapt to whatever API you actually design):
///
/// ```ignore
/// let mut points: Vec<Point2<f64>> =
///     case.points.iter().map(|&(x, y)| Point2::new(x, y)).collect();
/// let s = OUTER_LAYER_SIZE;
/// points.push(Point2::new(-s, -s));
/// points.push(Point2::new(-s, 1.0 + s));
/// points.push(Point2::new(1.0 + s, -s));
/// points.push(Point2::new(1.0 + s, 1.0 + s));
///
/// let mut cdt = Cdt::bulk_load_cdt(points, vec![]).unwrap();
/// for &[a, b] in case.edges {
///     cdt.add_constraint_and_split(
///         FixedVertexHandle::from_index(a),
///         FixedVertexHandle::from_index(b),
///         |v| v,
///     );
/// }
///
/// let bound = max_allowed_area(case.edge_length);
/// let parameters = RefinementParameters::<f64>::new()
///     .keep_constraint_edges()
///     .with_max_allowed_area_fn(move |p: Point2<f64>| {
///         // Small triangles inside the unit square, unbounded outside. Sharp jump —
///         // the angle criterion provides the grading.
///         if inside_unit_square_strict(p.x, p.y) { bound } else { f64::INFINITY }
///     })
///     .with_max_additional_vertices(MAX_ADDITIONAL_VERTICES);
/// cdt.refine(parameters);
/// Some(cdt)
/// ```
/// =====================================================================================
/// Builds ONE triangulation of the whole `[-10, 11]²` domain using
/// `refine_with_sizing`: the max-area bound applies to every triangle whose bounding box
/// overlaps the unit square (so huge triangles covering the square are caught as well),
/// everything else is only refined by the angle criterion.
fn build_single_triangulation_with_sizing(case: &Case) -> Option<Cdt> {
    let mut points: Vec<Point2<f64>> = case
        .points
        .iter()
        .map(|&(x, y)| Point2::new(x, y))
        .collect();
    let s = OUTER_LAYER_SIZE;
    points.push(Point2::new(-s, -s));
    points.push(Point2::new(-s, 1.0 + s));
    points.push(Point2::new(1.0 + s, -s));
    points.push(Point2::new(1.0 + s, 1.0 + s));

    let mut cdt = Cdt::bulk_load_cdt(points, vec![]).expect("single bulk load failed");
    for &[a, b] in case.edges {
        cdt.add_constraint_and_split(
            FixedVertexHandle::from_index(a),
            FixedVertexHandle::from_index(b),
            |v| v,
        );
    }

    let bound = max_allowed_area(case.edge_length);
    let parameters = RefinementParameters::<f64>::new()
        .keep_constraint_edges()
        .with_max_additional_vertices(MAX_ADDITIONAL_VERTICES);
    cdt.refine_with_sizing(parameters, |[a, b, c]: [Point2<f64>; 3]| {
        let overlaps_unit_square = a.x.max(b.x).max(c.x) > 0.0
            && a.x.min(b.x).min(c.x) < 1.0
            && a.y.max(b.y).max(c.y) > 0.0
            && a.y.min(b.y).min(c.y) < 1.0;
        if overlaps_unit_square {
            bound
        } else {
            f64::INFINITY
        }
    });

    Some(cdt)
}

fn run_solution(case: &Case, baseline: &BaselineResult, failed: &mut bool) {
    // Constraint handles are the original portal-edge index pairs: the input points are
    // distinct and bulk_load_cdt preserves their order, and the 4 far corners are
    // appended after them, so indices in `case.edges` stay valid in the merged point set.
    let start = Instant::now();
    let Some(cdt) = build_single_triangulation_with_sizing(case) else {
        println!("    NOT IMPLEMENTED YET (fill in build_single_triangulation_with_sizing)");
        return;
    };
    let elapsed = start.elapsed();

    let mut ok = true;

    for &[a, b] in case.edges {
        if !cdt.exists_constraint(
            FixedVertexHandle::from_index(a),
            FixedVertexHandle::from_index(b),
        ) {
            println!("    constraint check: FAIL — portal edge {a} <-> {b} was split or lost");
            ok = false;
            break;
        }
    }
    if ok {
        println!("    constraint check: ok (all portal edges survived refinement)");
    }

    // Every face lying inside the unit square must satisfy the area bound. Evaluated at
    // the centroid; if your sizing function samples elsewhere (e.g. circumcenter),
    // loosen this check accordingly and document the choice.
    let bound = max_allowed_area(case.edge_length);
    let mut inside_count = 0usize;
    let mut worst_violation: f64 = 0.0;
    for face in cdt.inner_faces() {
        let (cx, cy) = centroid(face.positions());
        if inside_unit_square_strict(cx, cy) {
            inside_count += 1;
            let area = triangle_area(face.positions());
            if area > bound {
                worst_violation = worst_violation.max(area / bound);
            }
        }
    }
    if worst_violation > 1.0 {
        println!(
            "    area-bound check: FAIL — worst triangle inside the unit square exceeds \
             the bound by {worst_violation:.2}x"
        );
        ok = false;
    } else {
        println!(
            "    area-bound check: ok (all {inside_count} unit-square triangles within bound)"
        );
    }

    let total = cdt.inner_faces().count();
    println!(
        "    triangles: {} total, {} inside unit square (baseline: {} total, {} inner)",
        total, inside_count, baseline.total_triangles, baseline.inner_triangles
    );
    println!(
        "    time: {:.2?} (baseline both triangulations: {:.2?})",
        elapsed, baseline.total_time
    );

    // Vertex-count sanity: the whole point of the feature is that the huge outer region
    // stays coarse. Allow generous slack for grading, but catch runaway refinement.
    if inside_count > 0 && total > baseline.total_triangles * 3 {
        println!(
            "    triangle-count check: FAIL — {}x more triangles than baseline; the outer \
             region is probably being over-refined",
            total / baseline.total_triangles.max(1)
        );
        ok = false;
    } else {
        println!("    triangle-count check: ok");
    }

    render_case_images(case, &cdt);

    if !ok {
        *failed = true;
    }
}

fn main() {
    let mut failed = false;
    for case in CASES {
        println!("case: {}", case.name);
        let baseline = run_baseline(case, &mut failed);
        println!("  single triangulation with sizing function (the goal):");
        run_solution(case, &baseline, &mut failed);
        println!();
    }
    if failed {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
    println!("RESULT: ok");
}

// =========================================================================================
// Visualization: renders the single-triangulation solution to PNG files in `mre_images/`.
//
// Visual encoding:
//   * triangle fill — log(area) on a single-hue sequential ramp: near-white blue for the
//     largest triangles, deep blue for the smallest, so the grading is visible at a glance;
//   * thin slate lines — triangulation edges (edges shorter than ~2.5px are not stroked,
//     otherwise the fine region would degrade into a solid ink blob at full-domain zoom);
//   * thick red lines — constraint edges (the portal polylines);
//   * dashed green outline — the unit square, i.e. the boundary of the fine-sizing region.
//
// Two views per case, each at full size and as a small preview:
//   * `<case>_full.png` — the whole `[-10, 11]²` domain;
//   * `<case>_unit_square.png` — closeup of the unit square plus a margin.
//
// Set the environment variable `MRE_NO_RENDER` to skip rendering (e.g. for quick timing
// runs).
// =========================================================================================

/// Side length (in pixels) of the full-size renders.
const RENDER_SIZE: u32 = 10_000;
/// Side length (in pixels) of the preview renders.
const PREVIEW_SIZE: u32 = 1_500;

#[derive(Clone, Copy)]
struct Viewport {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl Viewport {
    fn intersects_bbox(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> bool {
        max_x >= self.min_x && min_x <= self.max_x && max_y >= self.min_y && min_y <= self.max_y
    }
}

/// Sequential single-hue ramp: `t = 0` (largest area) -> near-white blue,
/// `t = 1` (smallest area) -> deep blue.
fn area_fill_color(t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let light = (240.0, 244.0, 250.0);
    let deep = (43.0, 108.0, 176.0);
    (
        (light.0 + (deep.0 - light.0) * t) as u8,
        (light.1 + (deep.1 - light.1) * t) as u8,
        (light.2 + (deep.2 - light.2) * t) as u8,
    )
}

fn render_cdt_png(cdt: &Cdt, vp: Viewport, size: u32, out_path: &str) {
    let mut pixmap = Pixmap::new(size, size).expect("failed to allocate pixmap");
    pixmap.fill(Color::from_rgba8(253, 253, 251, 255));

    let scale = size as f64 / (vp.max_x - vp.min_x);
    let to_px = |p: Point2<f64>| -> (f32, f32) {
        (
            ((p.x - vp.min_x) * scale) as f32,
            ((vp.max_y - p.y) * scale) as f32,
        )
    };
    // Stroke widths are chosen for the 10000px render and scale with the actual size.
    let px_scale = size as f64 / RENDER_SIZE as f64;

    // ---- triangle fills, colored by log(area) ----
    let mut visible = Vec::new();
    let mut log_min = f64::INFINITY;
    let mut log_max = f64::NEG_INFINITY;
    for face in cdt.inner_faces() {
        let positions = face.positions();
        let [a, b, c] = positions;
        if !vp.intersects_bbox(
            a.x.min(b.x).min(c.x),
            a.y.min(b.y).min(c.y),
            a.x.max(b.x).max(c.x),
            a.y.max(b.y).max(c.y),
        ) {
            continue;
        }
        let log_area = triangle_area(positions).max(f64::MIN_POSITIVE).ln();
        log_min = log_min.min(log_area);
        log_max = log_max.max(log_area);
        visible.push((positions, log_area));
    }
    let log_range = (log_max - log_min).max(1e-12);

    let mut fill_paint = Paint::default();
    fill_paint.anti_alias = true;
    for &(positions, log_area) in &visible {
        let t = 1.0 - (log_area - log_min) / log_range;
        let (r, g, b) = area_fill_color(t);
        fill_paint.set_color_rgba8(r, g, b, 255);

        let [pa, pb, pc] = positions.map(to_px);
        let mut path = PathBuilder::new();
        path.move_to(pa.0, pa.1);
        path.line_to(pb.0, pb.1);
        path.line_to(pc.0, pc.1);
        path.close();
        if let Some(path) = path.finish() {
            pixmap.fill_path(&path, &fill_paint, FillRule::Winding, Transform::identity(), None);
        }
    }

    // ---- regular triangulation edges (thin slate) ----
    let mut edge_paint = Paint::default();
    edge_paint.anti_alias = true;
    edge_paint.set_color_rgba8(70, 74, 82, 150);
    let edge_stroke = Stroke {
        width: ((1.6 * px_scale).max(0.7)) as f32,
        ..Default::default()
    };

    // Edges are accumulated into chunked multi-segment paths - one stroke call per edge
    // would be needlessly slow at these edge counts.
    let mut chunk = PathBuilder::new();
    let mut chunk_len = 0usize;
    for edge in cdt.undirected_edges() {
        if edge.is_constraint_edge() {
            continue;
        }
        let [p0, p1] = edge.positions();
        if !vp.intersects_bbox(
            p0.x.min(p1.x),
            p0.y.min(p1.y),
            p0.x.max(p1.x),
            p0.y.max(p1.y),
        ) {
            continue;
        }
        let (x0, y0) = to_px(p0);
        let (x1, y1) = to_px(p1);
        if ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt() < 2.5 {
            continue;
        }
        chunk.move_to(x0, y0);
        chunk.line_to(x1, y1);
        chunk_len += 1;
        if chunk_len == 4096 {
            if let Some(path) = std::mem::replace(&mut chunk, PathBuilder::new()).finish() {
                pixmap.stroke_path(&path, &edge_paint, &edge_stroke, Transform::identity(), None);
            }
            chunk_len = 0;
        }
    }
    if let Some(path) = chunk.finish() {
        pixmap.stroke_path(&path, &edge_paint, &edge_stroke, Transform::identity(), None);
    }

    // ---- constraint edges (thick red, always drawn) ----
    let mut constraint_paint = Paint::default();
    constraint_paint.anti_alias = true;
    constraint_paint.set_color_rgba8(214, 40, 40, 255);
    let constraint_stroke = Stroke {
        width: ((6.0 * px_scale).max(1.4)) as f32,
        line_cap: LineCap::Round,
        ..Default::default()
    };
    let mut constraints = PathBuilder::new();
    for edge in cdt.undirected_edges().filter(|e| e.is_constraint_edge()) {
        let [p0, p1] = edge.positions();
        if !vp.intersects_bbox(
            p0.x.min(p1.x),
            p0.y.min(p1.y),
            p0.x.max(p1.x),
            p0.y.max(p1.y),
        ) {
            continue;
        }
        let (x0, y0) = to_px(p0);
        let (x1, y1) = to_px(p1);
        constraints.move_to(x0, y0);
        constraints.line_to(x1, y1);
    }
    if let Some(path) = constraints.finish() {
        pixmap.stroke_path(
            &path,
            &constraint_paint,
            &constraint_stroke,
            Transform::identity(),
            None,
        );
    }

    // ---- fine-region boundary: the unit square, dashed green ----
    let mut boundary_paint = Paint::default();
    boundary_paint.anti_alias = true;
    boundary_paint.set_color_rgba8(46, 125, 50, 230);
    let boundary_stroke = Stroke {
        width: ((3.5 * px_scale).max(1.0)) as f32,
        dash: StrokeDash::new(
            vec![(30.0 * px_scale) as f32, (18.0 * px_scale) as f32],
            0.0,
        ),
        ..Default::default()
    };
    let corners = [
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(1.0, 1.0),
        Point2::new(0.0, 1.0),
    ];
    let mut boundary = PathBuilder::new();
    let (x0, y0) = to_px(corners[0]);
    boundary.move_to(x0, y0);
    for corner in &corners[1..] {
        let (x, y) = to_px(*corner);
        boundary.line_to(x, y);
    }
    boundary.close();
    if let Some(path) = boundary.finish() {
        pixmap.stroke_path(
            &path,
            &boundary_paint,
            &boundary_stroke,
            Transform::identity(),
            None,
        );
    }

    pixmap.save_png(out_path).expect("failed to save png");
}

fn render_case_images(case: &Case, cdt: &Cdt) {
    if std::env::var_os("MRE_NO_RENDER").is_some() {
        return;
    }
    std::fs::create_dir_all("mre_images").expect("failed to create mre_images/");
    let slug = case.name.split_whitespace().next().unwrap_or("case");

    let pad = 0.5;
    let s = OUTER_LAYER_SIZE;
    let full = Viewport {
        min_x: -s - pad,
        min_y: -s - pad,
        max_x: 1.0 + s + pad,
        max_y: 1.0 + s + pad,
    };
    let closeup = Viewport {
        min_x: -0.15,
        min_y: -0.15,
        max_x: 1.15,
        max_y: 1.15,
    };

    for (view_name, vp) in [("full", full), ("unit_square", closeup)] {
        for (suffix, size) in [("", RENDER_SIZE), ("_preview", PREVIEW_SIZE)] {
            let out_path = format!("mre_images/{slug}_{view_name}{suffix}.png");
            let start = Instant::now();
            render_cdt_png(cdt, vp, size, &out_path);
            println!("    wrote {out_path} ({size}px) in {:.2?}", start.elapsed());
        }
    }
}

// =========================================================================================
// Hardcoded geometry, extracted verbatim from the downstream project (scene `up_down.ron`,
// patch `user.Explore`). Layout per case: points 0-3 are the unit-square corners
// (0,0), (1,0), (1,1), (0,1); the rest are two portal polylines (y = 0.2 and y = 0.8,
// x from 0.6 down to 0.4). `*_EDGES` are the polyline segments; both polylines must keep
// exactly this discretization after refinement — the downstream FEM pairs the two sides
// point-by-point.
// =========================================================================================

#[rustfmt::skip]
const EXPLORE_POINTS: &[(f64, f64)] = &[
    (0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
    (0.0, 1.0), (0.6, 0.2), (0.5909090909090909, 0.2),
    (0.5818181818181818, 0.2), (0.5727272727272728, 0.2), (0.5636363636363636, 0.2),
    (0.5545454545454546, 0.2), (0.5454545454545454, 0.2), (0.5363636363636364, 0.2),
    (0.5272727272727272, 0.2), (0.5181818181818182, 0.2), (0.509090909090909, 0.2),
    (0.5, 0.2), (0.4909090909090909, 0.2), (0.4818181818181818, 0.2),
    (0.4727272727272727, 0.2), (0.4636363636363636, 0.2), (0.45454545454545453, 0.2),
    (0.44545454545454544, 0.2), (0.43636363636363634, 0.2), (0.42727272727272725, 0.2),
    (0.4181818181818182, 0.2), (0.40909090909090906, 0.2), (0.4, 0.2),
    (0.6, 0.8), (0.5909090909090909, 0.8), (0.5818181818181818, 0.8),
    (0.5727272727272728, 0.8), (0.5636363636363636, 0.8), (0.5545454545454546, 0.8),
    (0.5454545454545454, 0.8), (0.5363636363636364, 0.8), (0.5272727272727272, 0.8),
    (0.5181818181818182, 0.8), (0.509090909090909, 0.8), (0.5, 0.8),
    (0.4909090909090909, 0.8), (0.4818181818181818, 0.8), (0.4727272727272727, 0.8),
    (0.4636363636363636, 0.8), (0.45454545454545453, 0.8), (0.44545454545454544, 0.8),
    (0.43636363636363634, 0.8), (0.42727272727272725, 0.8), (0.4181818181818182, 0.8),
    (0.40909090909090906, 0.8), (0.4, 0.8),
];

#[rustfmt::skip]
const EXPLORE_EDGES: &[[usize; 2]] = &[
    [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
    [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20],
    [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [27, 28], [28, 29],
    [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 37],
    [37, 38], [38, 39], [39, 40], [40, 41], [41, 42], [42, 43], [43, 44], [44, 45],
    [45, 46], [46, 47], [47, 48], [48, 49],
];

#[rustfmt::skip]
const RENDER_POINTS: &[(f64, f64)] = &[
    (0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
    (0.0, 1.0), (0.6, 0.2), (0.5980769230769231, 0.2),
    (0.5961538461538461, 0.2), (0.5942307692307692, 0.2), (0.5923076923076923, 0.2),
    (0.5903846153846154, 0.2), (0.5884615384615385, 0.2), (0.5865384615384616, 0.2),
    (0.5846153846153846, 0.2), (0.5826923076923077, 0.2), (0.5807692307692308, 0.2),
    (0.5788461538461539, 0.2), (0.5769230769230769, 0.2), (0.575, 0.2),
    (0.573076923076923, 0.2), (0.5711538461538461, 0.2), (0.5692307692307692, 0.2),
    (0.5673076923076923, 0.2), (0.5653846153846154, 0.2), (0.5634615384615385, 0.2),
    (0.5615384615384615, 0.2), (0.5596153846153846, 0.2), (0.5576923076923077, 0.2),
    (0.5557692307692308, 0.2), (0.5538461538461539, 0.2), (0.551923076923077, 0.2),
    (0.55, 0.2), (0.5480769230769231, 0.2), (0.5461538461538462, 0.2),
    (0.5442307692307692, 0.2), (0.5423076923076923, 0.2), (0.5403846153846154, 0.2),
    (0.5384615384615384, 0.2), (0.5365384615384615, 0.2), (0.5346153846153846, 0.2),
    (0.5326923076923077, 0.2), (0.5307692307692308, 0.2), (0.5288461538461539, 0.2),
    (0.5269230769230769, 0.2), (0.525, 0.2), (0.5230769230769231, 0.2),
    (0.5211538461538462, 0.2), (0.5192307692307693, 0.2), (0.5173076923076924, 0.2),
    (0.5153846153846153, 0.2), (0.5134615384615384, 0.2), (0.5115384615384615, 0.2),
    (0.5096153846153846, 0.2), (0.5076923076923077, 0.2), (0.5057692307692307, 0.2),
    (0.5038461538461538, 0.2), (0.5019230769230769, 0.2), (0.5, 0.2),
    (0.4980769230769231, 0.2), (0.49615384615384617, 0.2), (0.49423076923076925, 0.2),
    (0.49230769230769234, 0.2), (0.49038461538461536, 0.2), (0.48846153846153845, 0.2),
    (0.48653846153846153, 0.2), (0.4846153846153846, 0.2), (0.4826923076923077, 0.2),
    (0.4807692307692308, 0.2), (0.47884615384615387, 0.2), (0.4769230769230769, 0.2),
    (0.475, 0.2), (0.47307692307692306, 0.2), (0.47115384615384615, 0.2),
    (0.46923076923076923, 0.2), (0.4673076923076923, 0.2), (0.4653846153846154, 0.2),
    (0.4634615384615385, 0.2), (0.46153846153846156, 0.2), (0.45961538461538465, 0.2),
    (0.4576923076923077, 0.2), (0.45576923076923076, 0.2), (0.45384615384615384, 0.2),
    (0.4519230769230769, 0.2), (0.45, 0.2), (0.4480769230769231, 0.2),
    (0.4461538461538461, 0.2), (0.4442307692307692, 0.2), (0.4423076923076923, 0.2),
    (0.4403846153846154, 0.2), (0.43846153846153846, 0.2), (0.43653846153846154, 0.2),
    (0.4346153846153846, 0.2), (0.4326923076923077, 0.2), (0.4307692307692308, 0.2),
    (0.4288461538461539, 0.2), (0.4269230769230769, 0.2), (0.425, 0.2),
    (0.4230769230769231, 0.2), (0.42115384615384616, 0.2), (0.41923076923076924, 0.2),
    (0.4173076923076923, 0.2), (0.41538461538461535, 0.2), (0.41346153846153844, 0.2),
    (0.4115384615384615, 0.2), (0.4096153846153846, 0.2), (0.4076923076923077, 0.2),
    (0.40576923076923077, 0.2), (0.40384615384615385, 0.2), (0.40192307692307694, 0.2),
    (0.4, 0.2), (0.6, 0.8), (0.5980769230769231, 0.8),
    (0.5961538461538461, 0.8), (0.5942307692307692, 0.8), (0.5923076923076923, 0.8),
    (0.5903846153846154, 0.8), (0.5884615384615385, 0.8), (0.5865384615384616, 0.8),
    (0.5846153846153846, 0.8), (0.5826923076923077, 0.8), (0.5807692307692308, 0.8),
    (0.5788461538461539, 0.8), (0.5769230769230769, 0.8), (0.575, 0.8),
    (0.573076923076923, 0.8), (0.5711538461538461, 0.8), (0.5692307692307692, 0.8),
    (0.5673076923076923, 0.8), (0.5653846153846154, 0.8), (0.5634615384615385, 0.8),
    (0.5615384615384615, 0.8), (0.5596153846153846, 0.8), (0.5576923076923077, 0.8),
    (0.5557692307692308, 0.8), (0.5538461538461539, 0.8), (0.551923076923077, 0.8),
    (0.55, 0.8), (0.5480769230769231, 0.8), (0.5461538461538462, 0.8),
    (0.5442307692307692, 0.8), (0.5423076923076923, 0.8), (0.5403846153846154, 0.8),
    (0.5384615384615384, 0.8), (0.5365384615384615, 0.8), (0.5346153846153846, 0.8),
    (0.5326923076923077, 0.8), (0.5307692307692308, 0.8), (0.5288461538461539, 0.8),
    (0.5269230769230769, 0.8), (0.525, 0.8), (0.5230769230769231, 0.8),
    (0.5211538461538462, 0.8), (0.5192307692307693, 0.8), (0.5173076923076924, 0.8),
    (0.5153846153846153, 0.8), (0.5134615384615384, 0.8), (0.5115384615384615, 0.8),
    (0.5096153846153846, 0.8), (0.5076923076923077, 0.8), (0.5057692307692307, 0.8),
    (0.5038461538461538, 0.8), (0.5019230769230769, 0.8), (0.5, 0.8),
    (0.4980769230769231, 0.8), (0.49615384615384617, 0.8), (0.49423076923076925, 0.8),
    (0.49230769230769234, 0.8), (0.49038461538461536, 0.8), (0.48846153846153845, 0.8),
    (0.48653846153846153, 0.8), (0.4846153846153846, 0.8), (0.4826923076923077, 0.8),
    (0.4807692307692308, 0.8), (0.47884615384615387, 0.8), (0.4769230769230769, 0.8),
    (0.475, 0.8), (0.47307692307692306, 0.8), (0.47115384615384615, 0.8),
    (0.46923076923076923, 0.8), (0.4673076923076923, 0.8), (0.4653846153846154, 0.8),
    (0.4634615384615385, 0.8), (0.46153846153846156, 0.8), (0.45961538461538465, 0.8),
    (0.4576923076923077, 0.8), (0.45576923076923076, 0.8), (0.45384615384615384, 0.8),
    (0.4519230769230769, 0.8), (0.45, 0.8), (0.4480769230769231, 0.8),
    (0.4461538461538461, 0.8), (0.4442307692307692, 0.8), (0.4423076923076923, 0.8),
    (0.4403846153846154, 0.8), (0.43846153846153846, 0.8), (0.43653846153846154, 0.8),
    (0.4346153846153846, 0.8), (0.4326923076923077, 0.8), (0.4307692307692308, 0.8),
    (0.4288461538461539, 0.8), (0.4269230769230769, 0.8), (0.425, 0.8),
    (0.4230769230769231, 0.8), (0.42115384615384616, 0.8), (0.41923076923076924, 0.8),
    (0.4173076923076923, 0.8), (0.41538461538461535, 0.8), (0.41346153846153844, 0.8),
    (0.4115384615384615, 0.8), (0.4096153846153846, 0.8), (0.4076923076923077, 0.8),
    (0.40576923076923077, 0.8), (0.40384615384615385, 0.8), (0.40192307692307694, 0.8),
    (0.4, 0.8),
];

#[rustfmt::skip]
const RENDER_EDGES: &[[usize; 2]] = &[
    [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
    [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20],
    [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 28],
    [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36],
    [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 42], [42, 43], [43, 44],
    [44, 45], [45, 46], [46, 47], [47, 48], [48, 49], [49, 50], [50, 51], [51, 52],
    [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 60],
    [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 68],
    [68, 69], [69, 70], [70, 71], [71, 72], [72, 73], [73, 74], [74, 75], [75, 76],
    [76, 77], [77, 78], [78, 79], [79, 80], [80, 81], [81, 82], [82, 83], [83, 84],
    [84, 85], [85, 86], [86, 87], [87, 88], [88, 89], [89, 90], [90, 91], [91, 92],
    [92, 93], [93, 94], [94, 95], [95, 96], [96, 97], [97, 98], [98, 99], [99, 100],
    [100, 101], [101, 102], [102, 103], [103, 104], [104, 105], [105, 106], [106, 107], [107, 108],
    [109, 110], [110, 111], [111, 112], [112, 113], [113, 114], [114, 115], [115, 116], [116, 117],
    [117, 118], [118, 119], [119, 120], [120, 121], [121, 122], [122, 123], [123, 124], [124, 125],
    [125, 126], [126, 127], [127, 128], [128, 129], [129, 130], [130, 131], [131, 132], [132, 133],
    [133, 134], [134, 135], [135, 136], [136, 137], [137, 138], [138, 139], [139, 140], [140, 141],
    [141, 142], [142, 143], [143, 144], [144, 145], [145, 146], [146, 147], [147, 148], [148, 149],
    [149, 150], [150, 151], [151, 152], [152, 153], [153, 154], [154, 155], [155, 156], [156, 157],
    [157, 158], [158, 159], [159, 160], [160, 161], [161, 162], [162, 163], [163, 164], [164, 165],
    [165, 166], [166, 167], [167, 168], [168, 169], [169, 170], [170, 171], [171, 172], [172, 173],
    [173, 174], [174, 175], [175, 176], [176, 177], [177, 178], [178, 179], [179, 180], [180, 181],
    [181, 182], [182, 183], [183, 184], [184, 185], [185, 186], [186, 187], [187, 188], [188, 189],
    [189, 190], [190, 191], [191, 192], [192, 193], [193, 194], [194, 195], [195, 196], [196, 197],
    [197, 198], [198, 199], [199, 200], [200, 201], [201, 202], [202, 203], [203, 204], [204, 205],
    [205, 206], [206, 207], [207, 208], [208, 209], [209, 210], [210, 211], [211, 212], [212, 213],
];
