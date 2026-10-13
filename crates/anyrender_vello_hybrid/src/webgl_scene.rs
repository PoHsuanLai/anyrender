//! WebGL-compatible [`PaintScene`] implementation for [`vello_hybrid::Scene`].

use anyrender::{Filter, Glyph, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext};
use glifo::FontEmbolden;
use kurbo::{Affine, Diagonal2, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, StyleRef};
use vello_common::paint::PaintType;

use crate::image_fit::{ImageUploads, UploadedImage, downscale, fit_size};
use peniko::ImageBrush;
use vello_common::paint::ImageSource;

use std::sync::Arc;

const DEFAULT_TOLERANCE: f64 = 0.1;

pub struct WebGlImageManager<'a> {
    pub(crate) renderer: &'a mut vello_hybrid::WebGlRenderer,
    pub(crate) resources: &'a mut vello_hybrid::Resources,
    pub(crate) cache: &'a mut ImageUploads,
}

impl<'a> WebGlImageManager<'a> {
    pub fn new(
        renderer: &'a mut vello_hybrid::WebGlRenderer,
        resources: &'a mut vello_hybrid::Resources,
        cache: &'a mut ImageUploads,
    ) -> Self {
        Self {
            renderer,
            resources,
            cache,
        }
    }

    /// Uploads an image to the atlas, downscaling it if it is larger than the atlas allows.
    ///
    /// Returns `None` if the image cannot be uploaded; it is then not drawn.
    pub(crate) fn upload_image(&mut self, image: &peniko::ImageData) -> Option<UploadedImage> {
        let peniko_id = image.data.id();

        if let Some(uploaded) = self.cache.get(&peniko_id) {
            return *uploaded;
        }

        let uploaded = self.try_upload_image(image);
        self.cache.insert(peniko_id, uploaded);
        uploaded
    }

    fn try_upload_image(&mut self, image: &peniko::ImageData) -> Option<UploadedImage> {
        // Pixmaps are limited to u16 sizes
        if image.width > u32::from(u16::MAX) || image.height > u32::from(u16::MAX) {
            return None;
        }

        let ImageSource::Pixmap(mut pixmap) = ImageSource::from_peniko_image_data(image) else {
            unreachable!();
        };

        // Downscale if it does not fit in the atlas
        let size = (u32::from(pixmap.width()), u32::from(pixmap.height()));
        let fit = fit_size(size, self.resources.max_image_size())?;
        if fit != size {
            let (width, height) = (u16::try_from(fit.0).ok()?, u16::try_from(fit.1).ok()?);
            pixmap = Arc::new(downscale(&pixmap, width, height)?);
        }
        let scale = (
            f64::from(fit.0) / f64::from(size.0),
            f64::from(fit.1) / f64::from(size.1),
        );

        let id = self.renderer.upload_image(self.resources, &pixmap).ok()?;
        Some(UploadedImage { id, scale })
    }
}

enum LayerKind {
    /// A layer with a clip path, made of this many vello layers (one per filter of a filter
    /// list, at least one).
    Layer(usize),
    Clip,
}

pub struct WebGlScenePainter<'s> {
    scene: &'s mut vello_hybrid::Scene,
    layer_stack: Vec<LayerKind>,
    image_manager: WebGlImageManager<'s>,
}

impl<'s> WebGlScenePainter<'s> {
    pub fn new(scene: &'s mut vello_hybrid::Scene, image_manager: WebGlImageManager<'s>) -> Self {
        Self {
            scene,
            layer_stack: Vec::with_capacity(16),
            image_manager,
        }
    }
}

impl WebGlScenePainter<'_> {
    /// Converts a paint, and returns the correction to multiply onto the right of the paint
    /// transform (identity unless an image was downscaled on upload).
    ///
    /// Returns `None` for an image that could not be uploaded: nothing should be drawn with it.
    fn convert_paint(&mut self, paint: PaintRef<'_>) -> Option<(PaintType, Affine)> {
        let paint = match paint {
            Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
            Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
            Paint::Image(image_brush) => return self.convert_image_paint(image_brush),

            // TODO: custom paint
            Paint::Resource(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
            Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        };
        Some((paint, Affine::IDENTITY))
    }

    fn convert_image_paint(
        &mut self,
        image_brush: peniko::ImageBrushRef<'_>,
    ) -> Option<(PaintType, Affine)> {
        let uploaded = self.image_manager.upload_image(image_brush.image)?;
        let image = PaintType::Image(ImageBrush {
            image: ImageSource::OpaqueId {
                id: uploaded.id,
                // TODO: optimize opaque case
                may_have_transparency: true,
            },
            sampler: image_brush.sampler,
        });
        Some((image, uploaded.paint_correction()))
    }
}

impl RenderContext for WebGlScenePainter<'_> {}
impl PaintScene for WebGlScenePainter<'_> {
    fn reset(&mut self) {
        self.scene.reset();
    }

    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
        filter: Option<Arc<Filter>>,
        _backdrop_filter: Option<Arc<Filter>>,
    ) {
        let mut filters = filter
            .map(|filter| crate::filters::convert_filters(&filter))
            .unwrap_or_default()
            .into_iter();
        self.scene.set_transform(transform);
        self.scene
            .push_clip_path(&clip.into_path(DEFAULT_TOLERANCE));
        // The blend, alpha and clip apply to the filtered result, so they go on the outermost
        // layer; the rest of a filter list nests inside it.
        self.scene
            .push_layer(None, Some(blend.into()), Some(alpha), None, filters.next());
        let inner = filters.len();
        for filter in filters {
            self.scene.push_layer(None, None, None, None, Some(filter));
        }
        self.layer_stack.push(LayerKind::Layer(1 + inner));
    }

    fn push_clip_layer(&mut self, transform: Affine, clip: &impl Shape) {
        self.scene.set_transform(transform);
        self.layer_stack.push(LayerKind::Clip);
        self.scene
            .push_clip_path(&clip.into_path(DEFAULT_TOLERANCE));
    }

    fn pop_layer(&mut self) {
        if let Some(kind) = self.layer_stack.pop() {
            match kind {
                LayerKind::Layer(layers) => {
                    for _ in 0..layers {
                        self.scene.pop_layer();
                    }
                    self.scene.pop_clip_path();
                }
                LayerKind::Clip => self.scene.pop_clip_path(),
            }
        }
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.scene.set_transform(transform);
        self.scene.set_stroke(style.clone());
        let Some((paint, correction)) = self.convert_paint(paint.into()) else {
            return;
        };
        self.scene.set_paint(paint);
        self.scene
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY) * correction);
        self.scene.stroke_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.scene.set_transform(transform);
        self.scene.set_fill_rule(style);
        let Some((paint, correction)) = self.convert_paint(paint.into()) else {
            return;
        };
        self.scene.set_paint(paint);
        self.scene
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY) * correction);
        self.scene.fill_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn draw_glyphs<'a, 's2: 'a>(
        &'a mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: impl Into<StyleRef<'a>>,
        paint: impl Into<PaintRef<'a>>,
        _brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph> + Clone,
    ) {
        let Some((paint, correction)) = self.convert_paint(paint.into()) else {
            return;
        };
        self.scene.set_paint(paint);
        if correction != Affine::IDENTITY {
            // Glyph runs do not take a brush transform; only a downscaled image needs one.
            self.scene.set_paint_transform(correction);
        }
        self.scene.set_transform(transform);

        let style: StyleRef<'a> = style.into();
        match style {
            StyleRef::Fill(fill) => {
                self.scene.set_fill_rule(fill);
                self.scene
                    .glyph_run(self.image_manager.resources, font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .font_embolden(FontEmbolden::new(Diagonal2::new(embolden.x, embolden.y)))
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .fill_glyphs(glyphs.map(|g| glifo::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
            StyleRef::Stroke(stroke) => {
                self.scene.set_stroke(stroke.clone());
                self.scene
                    .glyph_run(self.image_manager.resources, font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .stroke_glyphs(glyphs.map(|g| glifo::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
        }
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        color: Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.scene.set_transform(transform);
        self.scene.set_paint(PaintType::Solid(color));
        self.scene
            .fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32, false);
    }
}
