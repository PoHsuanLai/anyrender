//! Fitting images into the renderer's image atlas.
//!
//! The atlas has a maximum image size (see [`vello_hybrid::Resources::max_image_size`]). Pages can
//! draw images that are larger, so those are downscaled before upload instead of failing.

use kurbo::Affine;
use peniko::color::PremulRgba8;
use rustc_hash::FxHashMap;
use vello_common::{paint::ImageId, pixmap::Pixmap};

/// An image that is in the atlas, and how it was scaled to get there.
#[derive(Clone, Copy, Debug)]
pub struct UploadedImage {
    pub id: ImageId,
    /// Uploaded size divided by original size, per axis. `(1.0, 1.0)` if the image was not scaled.
    pub scale: (f64, f64),
}

impl UploadedImage {
    /// The transform to multiply onto the right of a paint transform, which maps original image
    /// pixels to the destination, so that it maps the uploaded image pixels to the same
    /// destination rectangle: `paint_transform * paint_correction()`.
    pub fn paint_correction(&self) -> Affine {
        Affine::scale_non_uniform(1.0 / self.scale.0, 1.0 / self.scale.1)
    }
}

/// Uploads by peniko image id. `None` records an image that could not be uploaded, so it is
/// skipped on later frames without retrying.
pub type ImageUploads = FxHashMap<u64, Option<UploadedImage>>;

/// The size to upload an image of `size` at, so that it fits in `max` and keeps its aspect ratio.
///
/// Returns `size` unchanged if it already fits. Returns `None` if there is nothing to fit
/// (an empty image or an atlas that holds no pixels).
pub(crate) fn fit_size(size: (u32, u32), max: (u32, u32)) -> Option<(u32, u32)> {
    let (width, height) = size;
    let (max_width, max_height) = max;
    if width == 0 || height == 0 || max_width == 0 || max_height == 0 {
        return None;
    }
    if width <= max_width && height <= max_height {
        return Some(size);
    }
    let (width, height) = (u64::from(width), u64::from(height));
    let (max_width, max_height) = (u64::from(max_width), u64::from(max_height));
    // Compare width / max_width with height / max_height without dividing: the larger ratio is
    // the axis that has to shrink the most, and it lands exactly on its maximum.
    let (fit_width, fit_height) = if width * max_height >= height * max_width {
        (max_width, (height * max_width / width).max(1))
    } else {
        ((width * max_height / height).max(1), max_height)
    };
    Some((
        u32::try_from(fit_width).ok()?,
        u32::try_from(fit_height).ok()?,
    ))
}

/// Downscales a pixmap to `width` x `height` with a box filter: each destination pixel is the
/// average of the source area it covers. Pixels are premultiplied, so averaging is correct.
///
/// Returns `None` if `width` or `height` is zero or larger than the source (this only shrinks).
pub(crate) fn downscale(source: &Pixmap, width: u16, height: u16) -> Option<Pixmap> {
    let (src_width, src_height) = (u32::from(source.width()), u32::from(source.height()));
    let (dst_width, dst_height) = (u32::from(width), u32::from(height));
    if dst_width == 0 || dst_height == 0 || dst_width > src_width || dst_height > src_height {
        return None;
    }

    // Horizontal pass: src_width x src_height -> dst_width x src_height.
    let rows = source.data().chunks_exact(src_width as usize);
    let mut across = Vec::with_capacity((dst_width * src_height) as usize);
    for row in rows {
        for x in 0..dst_width {
            let mut sum = [0.0_f32; 4];
            for (k, weight) in box_weights(x, src_width, dst_width) {
                add_weighted(&mut sum, row[k as usize], weight);
            }
            across.push(sum);
        }
    }

    // Vertical pass: dst_width x src_height -> dst_width x dst_height.
    let mut pixels = Vec::with_capacity((dst_width * dst_height) as usize);
    for y in 0..dst_height {
        for x in 0..dst_width {
            let mut sum = [0.0_f32; 4];
            for (k, weight) in box_weights(y, src_height, dst_height) {
                let column = across[(k * dst_width + x) as usize];
                for (sum, value) in sum.iter_mut().zip(column) {
                    *sum += value * weight;
                }
            }
            pixels.push(to_pixel(sum));
        }
    }

    Some(Pixmap::from_parts_with_opacity(
        pixels,
        width,
        height,
        source.may_have_transparency(),
    ))
}

/// The source indices and weights (summing to 1) that destination index `index` averages when
/// `src` samples are shrunk to `dst` samples.
fn box_weights(index: u32, src: u32, dst: u32) -> impl Iterator<Item = (u32, f32)> {
    let scale = f64::from(src) / f64::from(dst);
    let start = f64::from(index) * scale;
    let end = (f64::from(index) + 1.0) * scale;
    let first = start.floor() as u32;
    let last = (end.ceil() as u32).min(src);
    (first..last).map(move |k| {
        let overlap = end.min(f64::from(k) + 1.0) - start.max(f64::from(k));
        (k, (overlap / scale) as f32)
    })
}

fn add_weighted(sum: &mut [f32; 4], pixel: PremulRgba8, weight: f32) {
    let channels = [pixel.r, pixel.g, pixel.b, pixel.a];
    for (sum, channel) in sum.iter_mut().zip(channels) {
        *sum += f32::from(channel) * weight;
    }
}

fn to_pixel(sum: [f32; 4]) -> PremulRgba8 {
    let channel = |value: f32| value.round().clamp(0.0, 255.0) as u8;
    PremulRgba8 {
        r: channel(sum[0]),
        g: channel(sum[1]),
        b: channel(sum[2]),
        a: channel(sum[3]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u16, height: u16, pixel: PremulRgba8) -> Pixmap {
        let pixels = vec![pixel; usize::from(width) * usize::from(height)];
        Pixmap::from_parts(pixels, width, height)
    }

    #[test]
    fn wide_banner_is_fit_to_the_atlas_width() {
        assert_eq!(fit_size((5600, 256), (4096, 4096)), Some((4096, 187)));
    }

    #[test]
    fn tall_image_is_fit_to_the_atlas_height() {
        assert_eq!(fit_size((256, 5600), (4096, 4096)), Some((187, 4096)));
    }

    #[test]
    fn fitting_image_is_untouched() {
        assert_eq!(fit_size((4096, 4096), (4096, 4096)), Some((4096, 4096)));
        assert_eq!(fit_size((10, 20), (4096, 4096)), Some((10, 20)));
    }

    #[test]
    fn thin_image_keeps_at_least_one_pixel() {
        assert_eq!(fit_size((60000, 1), (4096, 4096)), Some((4096, 1)));
    }

    #[test]
    fn nothing_fits_an_empty_atlas() {
        assert_eq!(fit_size((10, 10), (0, 0)), None);
        assert_eq!(fit_size((0, 10), (100, 100)), None);
    }

    #[test]
    fn downscale_has_the_requested_size() {
        let pixel = PremulRgba8 {
            r: 1,
            g: 2,
            b: 3,
            a: 255,
        };
        let small = downscale(&solid(56, 3, pixel), 40, 2).unwrap();
        assert_eq!((small.width(), small.height()), (40, 2));
        assert_eq!(small.data().len(), 80);
    }

    #[test]
    fn downscale_keeps_a_solid_colour() {
        let pixel = PremulRgba8 {
            r: 200,
            g: 100,
            b: 50,
            a: 255,
        };
        let small = downscale(&solid(56, 25, pixel), 40, 18).unwrap();
        assert!(small.data().iter().all(|&p| p == pixel));
    }

    #[test]
    fn downscale_keeps_a_translucent_colour() {
        let pixel = PremulRgba8 {
            r: 60,
            g: 30,
            b: 0,
            a: 120,
        };
        let small = downscale(&solid(33, 17, pixel), 7, 5).unwrap();
        assert!(small.data().iter().all(|&p| p == pixel));
    }

    #[test]
    fn downscale_averages_covered_pixels() {
        let black = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        };
        let white = PremulRgba8 {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        };
        let source = Pixmap::from_parts(vec![black, white], 2, 1);
        let small = downscale(&source, 1, 1).unwrap();
        assert_eq!(
            small.data()[0],
            PremulRgba8 {
                r: 128,
                g: 128,
                b: 128,
                a: 255
            }
        );
    }

    #[test]
    fn downscale_never_enlarges() {
        let pixel = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        };
        assert!(downscale(&solid(4, 4, pixel), 8, 4).is_none());
        assert!(downscale(&solid(4, 4, pixel), 0, 4).is_none());
    }

    #[test]
    fn paint_transform_covers_the_same_destination() {
        let uploaded = UploadedImage {
            id: ImageId::new(0),
            scale: (0.5, 0.25),
        };
        let paint = Affine::translate((10.0, 20.0));
        let adjusted = paint * uploaded.paint_correction();
        // The far corner of the 8x8 original and of the 4x2 upload land on the same point.
        let original = paint * kurbo::Point::new(8.0, 8.0);
        let scaled = adjusted * kurbo::Point::new(4.0, 2.0);
        assert!((original - scaled).hypot() < 1e-9);
    }
}
