//! Every CSS colour filter function, and filter lists, through the scene painter, against the
//! definitions in the Filter Effects spec.

#![cfg(feature = "filters")]

use std::sync::Arc;

use anyrender::{Filter, PaintScene, filters::FilterEffect};
use anyrender_vello_cpu::VelloCpuScenePainter;
use kurbo::{Affine, Rect};
use peniko::{Color, Fill};

const SIZE: u16 = 48;
/// The source colour, 8-bit sRGB: (0.8, 0.2, 0.1).
const SOURCE: [u8; 3] = [204, 51, 26];

fn render(filter: Filter) -> [u8; 4] {
    let mut painter = VelloCpuScenePainter::new(SIZE, SIZE);
    let full = Rect::new(0.0, 0.0, f64::from(SIZE), f64::from(SIZE));
    painter.push_layer(
        peniko::Mix::Normal,
        1.0,
        Affine::IDENTITY,
        &full,
        Some(Arc::new(filter)),
        None,
    );
    painter.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(SOURCE[0], SOURCE[1], SOURCE[2]),
        None,
        &full,
    );
    painter.pop_layer();
    let pixmap = painter.finish();
    let pixel = pixmap.sample(SIZE / 2, SIZE / 2);
    [pixel.r, pixel.g, pixel.b, pixel.a]
}

fn to_u8(rgb: [f32; 3]) -> [u8; 4] {
    let [r, g, b] = rgb.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
    [r, g, b, 255]
}

fn assert_close(got: [u8; 4], expected: [u8; 4], what: &str) {
    let close = got.iter().zip(expected).all(|(g, e)| g.abs_diff(e) <= 2);
    assert!(close, "{what}: got {got:?}, expected {expected:?}");
}

fn source() -> [f32; 3] {
    SOURCE.map(|c| f32::from(c) / 255.0)
}

fn luma([r, g, b]: [f32; 3], [lr, lg, lb]: [f32; 3]) -> f32 {
    lr * r + lg * g + lb * b
}

#[test]
fn each_function_matches_the_spec() {
    let c = source();
    let css_luma = luma(c, [0.2126, 0.7152, 0.0722]);
    let rot_luma = luma(c, [0.213, 0.715, 0.072]);
    let cases: [(&str, FilterEffect, [f32; 3]); 7] = [
        (
            "brightness(1.4)",
            FilterEffect::brightness(1.4),
            c.map(|v| v * 1.4),
        ),
        (
            "contrast(1.5)",
            FilterEffect::contrast(1.5),
            c.map(|v| 1.5 * v - 0.25),
        ),
        (
            "grayscale(0.6)",
            FilterEffect::grayscale(0.6),
            c.map(|v| 0.4 * v + 0.6 * css_luma),
        ),
        (
            "invert(0.7)",
            FilterEffect::invert(0.7),
            c.map(|v| 0.7 * (1.0 - v) + 0.3 * v),
        ),
        (
            "saturate(2)",
            FilterEffect::saturate(2.0),
            c.map(|v| rot_luma + 2.0 * (v - rot_luma)),
        ),
        (
            "sepia(1)",
            FilterEffect::sepia(1.0),
            [
                0.393 * c[0] + 0.769 * c[1] + 0.189 * c[2],
                0.349 * c[0] + 0.686 * c[1] + 0.168 * c[2],
                0.272 * c[0] + 0.534 * c[1] + 0.131 * c[2],
            ],
        ),
        (
            // A half turn: cos = -1, sin = 0.
            "hue-rotate(180deg)",
            FilterEffect::hue_rotate(core::f32::consts::PI),
            [
                -0.574 * c[0] + 1.43 * c[1] + 0.144 * c[2],
                0.426 * c[0] + 0.43 * c[1] + 0.144 * c[2],
                0.426 * c[0] + 1.43 * c[1] - 0.856 * c[2],
            ],
        ),
    ];
    for (what, effect, expected) in cases {
        assert_close(render(Filter::single(effect)), to_u8(expected), what);
    }
}

#[test]
fn opacity_scales_the_alpha() {
    let [r, g, b, a] = render(Filter::single(FilterEffect::opacity(0.5)));
    assert!(a.abs_diff(128) <= 1, "alpha {a}");
    // The pixmap is premultiplied.
    for (got, source) in [r, g, b].into_iter().zip(SOURCE) {
        assert!(
            got.abs_diff(source / 2) <= 2,
            "premultiplied colour {got} from {source}"
        );
    }
}

#[test]
fn a_filter_list_applies_left_to_right() {
    let c = source();
    let contrast = |v: f32| (1.5 * v - 0.25).clamp(0.0, 1.0);
    let invert = |v: f32| 1.0 - v;

    let list =
        Filter::linear_list([FilterEffect::contrast(1.5), FilterEffect::invert(1.0)].into_iter());
    assert_close(
        render(list),
        to_u8(c.map(|v| invert(contrast(v)))),
        "contrast(1.5) invert(1)",
    );

    let reversed =
        Filter::linear_list([FilterEffect::invert(1.0), FilterEffect::contrast(1.5)].into_iter());
    assert_close(
        render(reversed),
        to_u8(c.map(|v| contrast(invert(v)))),
        "invert(1) contrast(1.5)",
    );
}

#[test]
fn a_colour_function_and_a_blur_compose() {
    let c = source();
    // The centre of a large area is untouched by the blur.
    let list =
        Filter::linear_list([FilterEffect::contrast(1.2), FilterEffect::blur(3.0)].into_iter());
    assert_close(
        render(list),
        to_u8(c.map(|v| 1.2 * v - 0.1)),
        "contrast(1.2) blur(3px)",
    );
}

#[test]
fn effects_vello_cannot_draw_are_skipped_not_panicked_on() {
    let list = Filter::linear_list([FilterEffect::Tile, FilterEffect::grayscale(1.0)].into_iter());
    let css_luma = luma(source(), [0.2126, 0.7152, 0.0722]);
    assert_close(render(list), to_u8([css_luma; 3]), "tile then grayscale(1)");
}
