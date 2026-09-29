use anyrender::{
    Filter,
    filters::{EdgeMode, FilterEffect},
};
use vello_common::filter_effects::FilterPrimitive;

/// The vello filters to push, outermost first, for an anyrender filter.
///
/// Vello filters are single primitives, so a filter list (`contrast(1.2) blur(4px)`) becomes one
/// filter layer per effect: the first effect in the list applies to the content first, so it is
/// the innermost layer. Effects that vello cannot draw are skipped.
pub(crate) fn convert_filters(filter: &Filter) -> Vec<vello_common::filter_effects::Filter> {
    filter
        .nodes()
        .iter()
        .filter_map(|node| convert_filter_effect(&node.effect))
        .map(vello_common::filter_effects::Filter::from_primitive)
        .rev()
        .collect()
}

pub(crate) fn convert_filter_effect(effect: &FilterEffect) -> Option<FilterPrimitive> {
    Some(match effect {
        FilterEffect::Flood(color) => FilterPrimitive::Flood { color: *color },
        FilterEffect::GaussianBlur(blur) => FilterPrimitive::GaussianBlur {
            std_deviation: blur.std_deviation,
            edge_mode: convert_edge_mode(blur.edge_mode),
        },
        FilterEffect::DropShadow(shadow) => FilterPrimitive::DropShadow {
            dx: shadow.dx,
            dy: shadow.dy,
            std_deviation: shadow.std_deviation,
            color: shadow.color,
            edge_mode: convert_edge_mode(shadow.edge_mode),
        },
        FilterEffect::Offset(offset) => FilterPrimitive::Offset {
            dx: offset.x as f32,
            dy: offset.y as f32,
        },
        FilterEffect::ColorMatrix(matrix) => FilterPrimitive::ColorMatrix { matrix: matrix.0 },
        // The CSS opacity(), invert(), brightness() and contrast() functions are affine
        // transfers, which are colour matrices.
        FilterEffect::ComponentTransfer(_) => FilterPrimitive::ColorMatrix {
            matrix: effect.as_color_matrix()?,
        },
        // Vello cannot draw a blend filter yet (it would panic when the filter is prepared).
        FilterEffect::Blend(_mode) => return None,
        FilterEffect::Composite(_composite_operator) => return None,
        FilterEffect::Morphology(_morphology_filter) => return None,
        FilterEffect::ConvolveMatrix(_convolution_kernel) => return None,
        FilterEffect::Turbulence(_turbulence_filter) => return None,
        FilterEffect::DisplacementMap(_displacement_map_filter) => return None,
        FilterEffect::Image(_external_image_source) => return None,
        FilterEffect::Tile => return None,
        FilterEffect::DiffuseLighting(_diffuse_lighting_filter) => return None,
        FilterEffect::SpecularLighting(_specular_lighting_filter) => return None,
    })
}

fn convert_edge_mode(edge_mode: EdgeMode) -> vello_common::filter_effects::EdgeMode {
    match edge_mode {
        EdgeMode::Duplicate => vello_common::filter_effects::EdgeMode::Duplicate,
        EdgeMode::Wrap => vello_common::filter_effects::EdgeMode::Wrap,
        EdgeMode::Mirror => vello_common::filter_effects::EdgeMode::Mirror,
        EdgeMode::None => vello_common::filter_effects::EdgeMode::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_css_colour_function_is_a_colour_matrix() {
        for effect in [
            FilterEffect::brightness(1.2),
            FilterEffect::contrast(1.2),
            FilterEffect::grayscale(0.5),
            FilterEffect::hue_rotate(1.0),
            FilterEffect::invert(0.5),
            FilterEffect::opacity(0.5),
            FilterEffect::saturate(2.0),
            FilterEffect::sepia(0.5),
        ] {
            assert!(
                matches!(
                    convert_filter_effect(&effect),
                    Some(FilterPrimitive::ColorMatrix { .. })
                ),
                "{effect:?}"
            );
        }
    }

    #[test]
    fn effects_vello_cannot_draw_are_dropped_from_a_list() {
        let list = Filter::linear_list(
            [
                FilterEffect::contrast(1.2),
                FilterEffect::Tile,
                FilterEffect::blur(4.0),
            ]
            .into_iter(),
        );
        assert_eq!(convert_filters(&list).len(), 2);
        assert!(convert_filters(&Filter::empty()).is_empty());
    }
}
