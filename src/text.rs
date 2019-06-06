//! Tex
use super::{utils::*, Color};
use crate::{
    data::{DrawType, DynamicTexture, VxDraw},
    strtex,
};
use ::image as load_image;
use cgmath::Matrix4;
use cgmath::Rad;
use core::ptr::read;
#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
use gfx_backend_gl as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use gfx_hal::{
    adapter::PhysicalDevice,
    command,
    device::Device,
    format, image, memory,
    memory::Properties,
    pass,
    pso::{self, DescriptorPool},
    Backend, Primitive,
};
use glyph_brush::{BrushAction, BrushError, GlyphBrushBuilder, Section};
use std::mem::ManuallyDrop;

// ---

/// Options for this text layer
pub struct LayerOptions {}

impl LayerOptions {
    /// Create a new options structure
    pub fn new() -> Self {
        Self {}
    }
}

/// Options when adding a text
pub struct TextOptions<'a> {
    text: &'a str,
    font_size_x: f32,
    font_size_y: f32,
    translation: (f32, f32),
}

impl<'a> TextOptions<'a> {
    /// Create a new text option
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            font_size_x: 16.0,
            font_size_y: 16.0,
            translation: (0.0, 0.0),
        }
    }

    /// Set the font width and height
    pub fn font_size(self, font_size: f32) -> Self {
        Self {
            font_size_x: font_size,
            font_size_y: font_size,
            ..self
        }
    }

    /// Set the font width
    pub fn font_size_x(self, font_size: f32) -> Self {
        Self {
            font_size_x: font_size,
            ..self
        }
    }

    /// Set the font height
    pub fn font_size_y(self, font_size: f32) -> Self {
        Self {
            font_size_y: font_size,
            ..self
        }
    }

    /// Set the translation
    pub fn translation(self, trn: (f32, f32)) -> Self {
        Self {
            translation: trn,
            ..self
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SData {
    uv_begin: (f32, f32),
    uv_end: (f32, f32),
    topleft: (i32, i32),
    bottomright: (i32, i32),
}

/// Handle to a piece of text
pub struct Handle(strtex::Layer, Vec<usize>);

/// Handle to a layer (a single glyph store/font)
pub struct Layer {
    strtex_layer: strtex::Layer,
    glyph_brush: glyph_brush::GlyphBrush<'static, SData>,
}

/// Accessor object to all text
pub struct Text<'a> {
    vx: &'a mut VxDraw,
}

impl<'a> Text<'a> {
    /// Prepare to edit text
    ///
    /// You're not supposed to use this function directly (although you can).
    /// The recommended way of spawning a text is via [VxDraw::text()].
    pub fn new(vx: &'a mut VxDraw) -> Self {
        Self { vx }
    }

    /// Add a text layer to the system
    pub fn add_layer(&mut self, font: &'static [u8], opts: LayerOptions) -> Layer {
        let mut glyph_brush = GlyphBrushBuilder::using_font_bytes(font).build();
        let (width, height) = glyph_brush.texture_dimensions();

        let layer = self.vx.strtex().add_layer(
            &strtex::LayerOptions::new()
                .width(width as usize)
                .height(height as usize)
                .filter(strtex::Filter::Linear),
        );

        // self.vx.strtex().add(&layer, strtex::Sprite::new());

        Layer {
            strtex_layer: layer,
            glyph_brush,
        }
    }

    /// Add text to this layer
    pub fn add(&mut self, layer: &mut Layer, opts: TextOptions) {
        let strtex = &layer.strtex_layer;
        layer.glyph_brush.queue(Section {
            text: opts.text,
            scale: glyph_brush::rusttype::Scale {
                x: opts.font_size_x,
                y: opts.font_size_y,
            },
            ..Section::default()
        });
        match layer.glyph_brush.process_queued(
            |rect, tex_data| {
                let width = rect.max.x - rect.min.x;
                for (idx, alpha) in tex_data.iter().enumerate() {
                    let idx = idx as u32;
                    self.vx.strtex().set_pixel(
                        &strtex,
                        rect.min.x + idx % width,
                        rect.min.y + idx / width,
                        Color::Rgba(255, 255, 255, *alpha),
                    );
                }
            },
            |vtx| SData {
                uv_begin: (vtx.tex_coords.min.x, vtx.tex_coords.min.y),
                uv_end: (vtx.tex_coords.max.x, vtx.tex_coords.max.y),
                topleft: (vtx.pixel_coords.min.x, vtx.pixel_coords.min.y),
                bottomright: (vtx.pixel_coords.max.x, vtx.pixel_coords.max.y),
            },
        ) {
            Ok(BrushAction::Draw(vertices)) => {
                for (idx, vtx) in vertices.iter().enumerate() {
                    let muscale = 500.0;
                    let uv_b = vtx.uv_begin;
                    let uv_e = vtx.uv_end;
                    let beg = vtx.topleft;
                    let end = vtx.bottomright;
                    let begf = (
                        beg.0 as f32 / muscale + opts.translation.0,
                        beg.1 as f32 / muscale + opts.translation.1,
                    );
                    let width = (end.0 - beg.0) as f32 / muscale;
                    let height = (end.1 - beg.1) as f32 / muscale;
                    let orig = (-width / 2.0, -height / 2.0);
                    self.vx.strtex().add(
                        strtex,
                        strtex::Sprite::new()
                            .width(width)
                            .height(height)
                            .translation(begf)
                            .origin(orig)
                            .uv_begin(uv_b)
                            .uv_end(uv_e),
                    );
                }
                // Draw new vertices.
            }
            Ok(BrushAction::ReDraw) => {}
            Err(BrushError::TextureTooSmall { suggested }) => {
                println!["{:?}", suggested];
            }
        }
    }
}
