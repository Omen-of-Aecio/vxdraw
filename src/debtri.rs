//! Methods and types to control debug triangles
//!
//! A debug triangle is a triangle that ignores all global transformations and is always shown on the
//! screen (except when a triangle's coordinates are outside the screen). Debug triangles are meant
//! to be used to quickly find out if a state has been reached (for instance, change the color of a
//! debug triangle if collision is detected, or rotate a debug triangle while loading).
//!
//! Debug triangles always ignore all layers, and are always shown on top of the entire scene.
//!
//! See [debtri::Debtri] for all operations supported on debug triangles.
//! ```
//! use cgmath::{prelude::*, Deg, Matrix4};
//! use logger::{Generic, GenericLogger, Logger};
//! use vxdraw::{ShowWindow, VxDraw};
//! fn main() {
//!     let mut vx = VxDraw::new(Logger::<Generic>::spawn_test().to_logpass(),
//!         ShowWindow::Headless1k); // Change this to ShowWindow::Enable to show the window
//!
//!     let tri = vx.debtri().push(vxdraw::debtri::DebugTriangle::default());
//!
//!     // Turn the triangle white
//!     vx.debtri().set_color(&tri, [255, 255, 255, 255]);
//!
//!     // Rotate the triangle 90 degrees (counter clockwise)
//!     vx.debtri().set_rotation(&tri, Deg(90.0));
//!
//!     // Draw the frame with the identity matrix transformation (meaning no transformations)
//!     vx.draw_frame(&Matrix4::identity());
//!
//!     // Sleep here so the window does not instantly disappear
//!     std::thread::sleep(std::time::Duration::new(3, 0));
//! }
use super::utils::*;
use crate::data::{DebugTriangleData, VxDraw};
use cgmath::Rad;
#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
use gfx_backend_gl as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use gfx_hal::{device::Device, format, image, pass, pso, Adapter, Backend, Primitive};
use std::mem::ManuallyDrop;

// ---

/// Debug triangles accessor object returned by [VxDraw::debtri]
///
/// Merely used for grouping together all operations on debug triangles. This is a very cheap
/// object to create/destroy (it really does nothing).
pub struct Debtri<'a> {
    vx: &'a mut VxDraw,
}

impl<'a> Debtri<'a> {
    /// Spawn the accessor object from [VxDraw].
    ///
    /// This is a very cheap operation.
    pub fn new(vx: &'a mut VxDraw) -> Self {
        Self { vx }
    }

    /// Enable drawing of the debug triangles
    pub fn show(&mut self) {
        self.vx.debtris.hidden = false;
    }

    /// Disable drawing of the debug triangles
    pub fn hide(&mut self) {
        self.vx.debtris.hidden = true;
    }

    /// Compare triangle draw order
    ///
    /// All triangles are drawn in a specific order. This method figures out which order is used
    /// between two triangles. The order can be manipulated by [Debtri::swap].
    pub fn compare_triangle_order(&self, left: &Handle, right: &Handle) -> std::cmp::Ordering {
        left.0.cmp(&right.0)
    }

    /// Swap two triangles with each other
    ///
    /// Swaps the internal data of each triangle (all vertices and their data, translation,
    /// and so on). The side-effect of this is that the draw order is swapped too, meaning that the
    /// triangles may reverse order (one drawn on top of the other).
    pub fn swap(&mut self, left: &mut Handle, right: &mut Handle) {
        let debtris = &mut self.vx.debtris;

        for i in 0..6 {
            debtris.posbuffer.swap(left.0 * 6 + i, right.0 * 6 + i);
        }
        for i in 0..12 {
            debtris.colbuffer.swap(left.0 * 12 + i, right.0 * 12 + i);
        }
        for i in 0..6 {
            debtris.tranbuffer.swap(left.0 * 6 + i, right.0 * 6 + i);
        }
        for i in 0..3 {
            debtris.rotbuffer.swap(left.0 * 3 + i, right.0 * 3 + i);
        }
        for i in 0..3 {
            debtris.scalebuffer.swap(left.0 * 3 + i, right.0 * 3 + i);
        }

        debtris.posbuf_touch = self.vx.swapconfig.image_count;
        debtris.colbuf_touch = self.vx.swapconfig.image_count;
        debtris.tranbuf_touch = self.vx.swapconfig.image_count;
        debtris.rotbuf_touch = self.vx.swapconfig.image_count;
        debtris.scalebuf_touch = self.vx.swapconfig.image_count;

        std::mem::swap(&mut left.0, &mut right.0);
    }

    // ---

    /// Add a new debug triangle to the renderer
    ///
    /// The new triangle will be drawn upon the next draw invocation. The triangle generated is NOT
    /// guaranteed to be the triangle to be rendered on top of all other triangles if
    /// [Debtri::remove] has been called earlier. In general, use [Debtri::compare_triangle_order]
    /// and [Debtri::swap] to enforce drawing order if that's needed.
    pub fn push(&mut self, triangle: DebugTriangle) -> Handle {
        let debtris = &mut self.vx.debtris;

        let handle = if let Some(hole) = debtris.holes.pop() {
            for (idx, origin) in triangle.origin.iter().enumerate() {
                debtris.posbuffer[hole * 6 + idx * 2] = origin.0;
                debtris.posbuffer[hole * 6 + idx * 2 + 1] = origin.1;
            }

            for (idx, col) in triangle.colors_rgba.iter().enumerate() {
                debtris.colbuffer[hole * 12 + idx * 4] = col.0;
                debtris.colbuffer[hole * 12 + idx * 4 + 1] = col.1;
                debtris.colbuffer[hole * 12 + idx * 4 + 2] = col.2;
                debtris.colbuffer[hole * 12 + idx * 4 + 3] = col.3;
            }

            for idx in 0..3 {
                debtris.tranbuffer[hole * 6 + idx * 2] = triangle.translation.0;
                debtris.tranbuffer[hole * 6 + idx * 2 + 1] = triangle.translation.1;
            }

            for idx in 0..3 {
                debtris.rotbuffer[hole * 3 + idx] = triangle.rotation;
            }

            for idx in 0..3 {
                debtris.scalebuffer[hole * 3 + idx] = triangle.scale;
            }
            Handle(hole)
        } else {
            for origin in &triangle.origin {
                debtris.posbuffer.push(origin.0);
                debtris.posbuffer.push(origin.1);
            }

            for col in &triangle.colors_rgba {
                debtris.colbuffer.push(col.0);
                debtris.colbuffer.push(col.1);
                debtris.colbuffer.push(col.2);
                debtris.colbuffer.push(col.3);
            }

            for _ in 0..3 {
                debtris.tranbuffer.push(triangle.translation.0);
                debtris.tranbuffer.push(triangle.translation.1);
            }

            for _ in 0..3 {
                debtris.rotbuffer.push(triangle.rotation);
            }

            for _ in 0..3 {
                debtris.scalebuffer.push(triangle.scale);
            }
            debtris.triangles_count += 1;
            Handle(debtris.triangles_count - 1)
        };

        debtris.posbuf_touch = self.vx.swapconfig.image_count;
        debtris.colbuf_touch = self.vx.swapconfig.image_count;
        debtris.tranbuf_touch = self.vx.swapconfig.image_count;
        debtris.rotbuf_touch = self.vx.swapconfig.image_count;
        debtris.scalebuf_touch = self.vx.swapconfig.image_count;

        handle
    }

    /// Remove the topmost debug triangle from rendering
    ///
    /// Has no effect if there are no debug triangles.
    /// Beware that this function is intended for simple use cases, when using [Debtri::remove],
    /// holes may be created which cause [Debtri::pop] to not work as expected. Please use
    /// [Debtri::remove] instead of [Debtri::pop] in complex applications.
    /// See [Debtri::push] for more information.
    pub fn pop(&mut self) {
        self.vx.debtris.triangles_count =
            self.vx.debtris.triangles_count.checked_sub(1).unwrap_or(0);
    }

    /// Remove the last N added debug triangle from rendering
    ///
    /// If the amount to pop is bigger than the amount of debug triangles, then all debug triangles
    /// wil be removed.
    pub fn pop_many(&mut self, n: usize) {
        let end = self.vx.debtris.triangles_count;
        let begin = end.checked_sub(n).unwrap_or(0);

        let debtris = &mut self.vx.debtris;
        debtris.posbuffer.drain(begin * 6..end * 6);
        debtris.colbuffer.drain(begin * 12..end * 12);
        debtris.tranbuffer.drain(begin * 6..end * 6);
        debtris.rotbuffer.drain(begin * 3..end * 3);
        debtris.scalebuffer.drain(begin * 3..end * 3);

        debtris.triangles_count = begin;
    }

    /// Remove a debug triangle
    ///
    /// The triangle is set to a scale of 0 and its handle is stored internally in a list of
    /// `holes`. Calling [Debtri::push] with available holes will fill the first available hole
    /// with the new triangle.
    pub fn remove(&mut self, handle: Handle) {
        self.vx.debtris.holes.push(handle.0);
        self.set_scale(&handle, 0.0);
    }

    // ---

    /// Change the vertices of the model-space
    ///
    /// The name `set_deform` is used to keep consistent with the verb `deform` and `deform_all`.
    /// What this function does is just setting absolute vertex positions for each vertex in the
    /// triangle.
    pub fn set_deform(&mut self, handle: &Handle, points: [(f32, f32); 3]) {
        self.vx.debtris.posbuf_touch = self.vx.swapconfig.image_count;
        for (idx, vertex) in self.vx.debtris.posbuffer[handle.0 * 6..(handle.0 + 1) * 6]
            .chunks_exact_mut(2)
            .enumerate()
        {
            vertex[0] = points[idx].0;
            vertex[1] = points[idx].1;
        }
    }

    /// Set a solid color of a debug triangle
    pub fn set_color(&mut self, handle: &Handle, rgba: [u8; 4]) {
        self.vx.debtris.colbuf_touch = self.vx.swapconfig.image_count;
        for vtx in 0..3 {
            for (coli, cmpnt) in rgba.iter().enumerate() {
                self.vx.debtris.colbuffer[handle.0 * 12 + vtx * 4 + coli] = *cmpnt;
            }
        }
    }

    /// Set the position (translation) of a debug triangle
    ///
    /// The name `set_translation` is chosen to keep the counterparts `translate` and
    /// `translate_all` consistent. This function can purely be thought of as setting the position
    /// of the triangle with respect to the model-space's origin.
    pub fn set_translation(&mut self, handle: &Handle, pos: (f32, f32)) {
        self.vx.debtris.tranbuf_touch = self.vx.swapconfig.image_count;
        let idx = handle.0 * 3 * 2;
        for vtx in 0..3 {
            self.vx.debtris.tranbuffer[idx + vtx * 2] = pos.0;
            self.vx.debtris.tranbuffer[idx + vtx * 2 + 1] = pos.1;
        }
    }

    /// Set the rotation of a debug triangle
    ///
    /// The rotation is about the model space origin.
    pub fn set_rotation<T: Copy + Into<Rad<f32>>>(&mut self, handle: &Handle, deg: T) {
        let angle = deg.into().0;
        self.vx.debtris.rotbuf_touch = self.vx.swapconfig.image_count;
        self.vx.debtris.rotbuffer[handle.0 * 3..(handle.0 + 1) * 3]
            .copy_from_slice(&[angle, angle, angle]);
    }

    /// Set the scale of a debug triangle
    pub fn set_scale(&mut self, handle: &Handle, scale: f32) {
        self.vx.debtris.scalebuf_touch = self.vx.swapconfig.image_count;
        for sc in &mut self.vx.debtris.scalebuffer[handle.0 * 3..(handle.0 + 1) * 3] {
            *sc = scale;
        }
    }

    // ---

    /// Deform a triangle by adding delta vertices
    ///
    /// Adds the delta vertices to the debug triangle. Beware: This changes model space form.
    pub fn deform(&mut self, handle: &Handle, delta: [(f32, f32); 3]) {
        self.vx.debtris.posbuf_touch = self.vx.swapconfig.image_count;
        for (idx, trn) in self.vx.debtris.posbuffer[handle.0 * 6..(handle.0 + 1) * 6]
            .chunks_exact_mut(2)
            .enumerate()
        {
            trn[0] += delta[idx].0;
            trn[1] += delta[idx].1;
        }
    }

    /// Color a debug triangle by adding a color
    ///
    /// Color mutates the model-space of a triangle. The color value this takes is [i16] because it
    /// needs to be able to add and subtract the color components. Internally the RGBA u8 color
    /// values are converted to [i16] and then cast back to [u8] using clamping.
    pub fn color(&mut self, handle: &Handle, color: [i16; 4]) {
        self.vx.debtris.tranbuf_touch = self.vx.swapconfig.image_count;
        for cols in
            self.vx.debtris.colbuffer[handle.0 * 12..(handle.0 + 1) * 12].chunks_exact_mut(4)
        {
            for (idx, color) in color.iter().enumerate() {
                let excol = i16::from(cols[idx]);
                cols[idx] = (excol + *color).min(255).max(0) as u8;
            }
        }
    }

    /// Translate a debug triangle by a vector
    ///
    /// Translation does not mutate the model-space of a triangle.
    pub fn translate(&mut self, handle: &Handle, delta: (f32, f32)) {
        self.vx.debtris.tranbuf_touch = self.vx.swapconfig.image_count;
        for stride in 0..3 {
            self.vx.debtris.tranbuffer[handle.0 * 6 + stride * 2] += delta.0;
            self.vx.debtris.tranbuffer[handle.0 * 6 + stride * 2 + 1] += delta.1;
        }
    }

    /// Rotate a debug triangle
    ///
    /// Rotation does not mutate the model-space of a triangle.
    pub fn rotate<T: Copy + Into<Rad<f32>>>(&mut self, handle: &Handle, deg: T) {
        self.vx.debtris.rotbuf_touch = self.vx.swapconfig.image_count;
        for rot in &mut self.vx.debtris.rotbuffer[handle.0 * 3..(handle.0 + 1) * 3] {
            *rot += deg.into().0;
        }
    }

    /// Scale a debug triangle
    ///
    /// Scale does not mutate the model-space of a triangle.
    pub fn scale(&mut self, handle: &Handle, scale: f32) {
        self.vx.debtris.scalebuf_touch = self.vx.swapconfig.image_count;
        for sc in &mut self.vx.debtris.scalebuffer[handle.0 * 3..(handle.0 + 1) * 3] {
            *sc *= scale;
        }
    }

    // ---

    /// Deform all triangles by adding delta vertices
    ///
    /// Adds the delta vertices to each debug triangle.
    /// See [Debtri::deform] for more information.
    pub fn deform_all(&mut self, delta: [(f32, f32); 3]) {
        self.vx.debtris.posbuf_touch = self.vx.swapconfig.image_count;
        for (idx, trn) in self.vx.debtris.posbuffer.chunks_exact_mut(2).enumerate() {
            trn[0] += delta[idx % 3].0;
            trn[1] += delta[idx % 3].1;
        }
    }

    /// Color all debug triangles by adding a color
    ///
    /// Adds the color in the argument to the existing color of each triangle.
    /// See [Debtri::color] for more information.
    pub fn color_all(&mut self, color: [i16; 4]) {
        self.vx.debtris.colbuf_touch = self.vx.swapconfig.image_count;
        for cols in self.vx.debtris.colbuffer.chunks_exact_mut(4) {
            for (idx, color) in color.iter().enumerate() {
                let excol = i16::from(cols[idx]);
                cols[idx] = (excol + *color).min(255).max(0) as u8;
            }
        }
    }

    /// Translate all debug triangles by a vector
    ///
    /// Adds the translation in the argument to the existing translation of each triangle.
    /// See [Debtri::translate] for more information.
    pub fn translate_all(&mut self, delta: (f32, f32)) {
        self.vx.debtris.tranbuf_touch = self.vx.swapconfig.image_count;
        for trn in self.vx.debtris.tranbuffer.chunks_exact_mut(2) {
            trn[0] += delta.0;
            trn[1] += delta.1;
        }
    }

    /// Rotate all debug triangles
    ///
    /// Adds the rotation in the argument to the existing rotation of each triangle.
    /// See [Debtri::rotate] for more information.
    pub fn rotate_all<T: Copy + Into<Rad<f32>>>(&mut self, rotation: T) {
        self.vx.debtris.rotbuf_touch = self.vx.swapconfig.image_count;
        for rot in &mut self.vx.debtris.rotbuffer {
            *rot += rotation.into().0;
        }
    }

    /// Scale all debug triangles (multiplicative)
    ///
    /// Multiplies the scale in the argument with the existing scale of each triangle.
    /// See [Debtri::scale] for more information.
    pub fn scale_all(&mut self, scale: f32) {
        self.vx.debtris.scalebuf_touch = self.vx.swapconfig.image_count;
        for sc in &mut self.vx.debtris.scalebuffer {
            *sc *= scale;
        }
    }
}

/// Handle to a debug triangle
///
/// Used to update/remove a debug triangle.
pub struct Handle(usize);

/// Information used when creating/updating a debug triangle
#[derive(Clone, Copy)]
pub struct DebugTriangle {
    pub origin: [(f32, f32); 3],
    pub colors_rgba: [(u8, u8, u8, u8); 3],
    pub translation: (f32, f32),
    pub rotation: f32,
    pub scale: f32,
}

impl From<[f32; 6]> for DebugTriangle {
    fn from(array: [f32; 6]) -> Self {
        let mut tri = Self::default();
        tri.origin[0].0 = array[0];
        tri.origin[0].1 = array[1];
        tri.origin[1].0 = array[2];
        tri.origin[1].1 = array[3];
        tri.origin[2].0 = array[4];
        tri.origin[2].1 = array[5];
        tri
    }
}

impl Default for DebugTriangle {
    /// Creates a default equilateral RGB triangle without opacity or rotation
    fn default() -> Self {
        let origin = make_centered_equilateral_triangle();
        DebugTriangle {
            origin: [
                (origin[0], origin[1]),
                (origin[2], origin[3]),
                (origin[4], origin[5]),
            ],
            colors_rgba: [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)],
            rotation: 0f32,
            translation: (0f32, 0f32),
            scale: 1f32,
        }
    }
}

impl DebugTriangle {
    /// Compute the circle that contains the entire triangle regardless of rotation
    ///
    /// Useful when making sure triangles do not touch by adding both their radii together and
    /// using that to space triangles.
    pub fn radius(&self) -> f32 {
        (self.origin[0].0.powi(2) + self.origin[0].1.powi(2))
            .sqrt()
            .max(
                (self.origin[1].0.powi(2) + self.origin[1].1.powi(2))
                    .sqrt()
                    .max((self.origin[2].0.powi(2) + self.origin[2].1.powi(2)).sqrt()),
            )
            * self.scale
    }
}

// ---

pub fn create_debug_triangle(
    device: &back::Device,
    adapter: &Adapter<back::Backend>,
    format: format::Format,
    image_count: usize,
) -> DebugTriangleData {
    pub const VERTEX_SOURCE: &[u8] = include_bytes!["../_build/spirv/debtri.vert.spirv"];
    pub const FRAGMENT_SOURCE: &[u8] = include_bytes!["../_build/spirv/debtri.frag.spirv"];

    let vs_module = { unsafe { device.create_shader_module(&VERTEX_SOURCE) }.unwrap() };
    let fs_module = { unsafe { device.create_shader_module(&FRAGMENT_SOURCE) }.unwrap() };

    const ENTRY_NAME: &str = "main";
    let vs_module: <back::Backend as Backend>::ShaderModule = vs_module;
    let (vs_entry, fs_entry) = (
        pso::EntryPoint {
            entry: ENTRY_NAME,
            module: &vs_module,
            specialization: pso::Specialization::default(),
        },
        pso::EntryPoint {
            entry: ENTRY_NAME,
            module: &fs_module,
            specialization: pso::Specialization::default(),
        },
    );

    let shader_entries = pso::GraphicsShaderSet {
        vertex: vs_entry,
        hull: None,
        domain: None,
        geometry: None,
        fragment: Some(fs_entry),
    };

    let input_assembler = pso::InputAssemblerDesc::new(Primitive::TriangleList);

    let vertex_buffers = vec![
        pso::VertexBufferDesc {
            binding: 0,
            stride: 8,
            rate: pso::VertexInputRate::Vertex,
        },
        pso::VertexBufferDesc {
            binding: 1,
            stride: 4,
            rate: pso::VertexInputRate::Vertex,
        },
        pso::VertexBufferDesc {
            binding: 2,
            stride: 8,
            rate: pso::VertexInputRate::Vertex,
        },
        pso::VertexBufferDesc {
            binding: 3,
            stride: 4,
            rate: pso::VertexInputRate::Vertex,
        },
        pso::VertexBufferDesc {
            binding: 4,
            stride: 4,
            rate: pso::VertexInputRate::Vertex,
        },
    ];

    let attributes: Vec<pso::AttributeDesc> = vec![
        pso::AttributeDesc {
            location: 0,
            binding: 0,
            element: pso::Element {
                format: format::Format::Rg32Sfloat,
                offset: 0,
            },
        },
        pso::AttributeDesc {
            location: 1,
            binding: 1,
            element: pso::Element {
                format: format::Format::Rgba8Unorm,
                offset: 0,
            },
        },
        pso::AttributeDesc {
            location: 2,
            binding: 2,
            element: pso::Element {
                format: format::Format::Rg32Sfloat,
                offset: 0,
            },
        },
        pso::AttributeDesc {
            location: 3,
            binding: 3,
            element: pso::Element {
                format: format::Format::R32Sfloat,
                offset: 0,
            },
        },
        pso::AttributeDesc {
            location: 4,
            binding: 4,
            element: pso::Element {
                format: format::Format::R32Sfloat,
                offset: 0,
            },
        },
    ];

    let rasterizer = pso::Rasterizer {
        depth_clamping: false,
        polygon_mode: pso::PolygonMode::Fill,
        cull_face: pso::Face::NONE,
        front_face: pso::FrontFace::CounterClockwise,
        depth_bias: None,
        conservative: false,
    };

    let depth_stencil = pso::DepthStencilDesc {
        depth: pso::DepthTest::Off,
        depth_bounds: false,
        stencil: pso::StencilTest::Off,
    };

    let blender = {
        let blend_state = pso::BlendState::On {
            color: pso::BlendOp::Add {
                src: pso::Factor::One,
                dst: pso::Factor::Zero,
            },
            alpha: pso::BlendOp::Add {
                src: pso::Factor::One,
                dst: pso::Factor::Zero,
            },
        };
        pso::BlendDesc {
            logic_op: Some(pso::LogicOp::Copy),
            targets: vec![pso::ColorBlendDesc(pso::ColorMask::ALL, blend_state)],
        }
    };

    let triangle_render_pass = {
        let attachment = pass::Attachment {
            format: Some(format),
            samples: 1,
            ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::Clear,
                pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: pass::AttachmentOps::DONT_CARE,
            layouts: image::Layout::Undefined..image::Layout::Present,
        };
        let depth = pass::Attachment {
            format: Some(format::Format::D32Sfloat),
            samples: 1,
            ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::Clear,
                pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: pass::AttachmentOps::DONT_CARE,
            layouts: image::Layout::Undefined..image::Layout::DepthStencilAttachmentOptimal,
        };

        let subpass = pass::SubpassDesc {
            colors: &[(0, image::Layout::ColorAttachmentOptimal)],
            depth_stencil: Some(&(1, image::Layout::DepthStencilAttachmentOptimal)),
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        unsafe { device.create_render_pass(&[attachment, depth], &[subpass], &[]) }
            .expect("Can't create render pass")
    };

    let baked_states = pso::BakedStates {
        viewport: None,
        scissor: None,
        blend_color: None,
        depth_bounds: None,
    };

    let bindings = Vec::<pso::DescriptorSetLayoutBinding>::new();
    let immutable_samplers = Vec::<<back::Backend as Backend>::Sampler>::new();
    let triangle_descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> =
        vec![unsafe {
            device
                .create_descriptor_set_layout(bindings, immutable_samplers)
                .expect("Couldn't make a DescriptorSetLayout")
        }];
    let mut push_constants = Vec::<(pso::ShaderStageFlags, core::ops::Range<u32>)>::new();
    push_constants.push((pso::ShaderStageFlags::VERTEX, 0..1));

    let triangle_pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&triangle_descriptor_set_layouts, push_constants)
            .expect("Couldn't create a pipeline layout")
    };

    // Describe the pipeline (rasterization, triangle interpretation)
    let pipeline_desc = pso::GraphicsPipelineDesc {
        shaders: shader_entries,
        rasterizer,
        vertex_buffers,
        attributes,
        input_assembler,
        blender,
        depth_stencil,
        multisampling: None,
        baked_states,
        layout: &triangle_pipeline_layout,
        subpass: pass::Subpass {
            index: 0,
            main_pass: &triangle_render_pass,
        },
        flags: pso::PipelineCreationFlags::empty(),
        parent: pso::BasePipeline::None,
    };

    let triangle_pipeline = unsafe {
        device
            .create_graphics_pipeline(&pipeline_desc, None)
            .expect("Couldn't create a graphics pipeline!")
    };

    unsafe {
        device.destroy_shader_module(vs_module);
        device.destroy_shader_module(fs_module);
    }

    let posbuf = (0..image_count)
        .map(|_| super::utils::ResizBuf::new(&device, &adapter))
        .collect::<Vec<_>>();
    let colbuf = (0..image_count)
        .map(|_| super::utils::ResizBuf::new(&device, &adapter))
        .collect::<Vec<_>>();
    let tranbuf = (0..image_count)
        .map(|_| super::utils::ResizBuf::new(&device, &adapter))
        .collect::<Vec<_>>();
    let rotbuf = (0..image_count)
        .map(|_| super::utils::ResizBuf::new(&device, &adapter))
        .collect::<Vec<_>>();
    let scalebuf = (0..image_count)
        .map(|_| super::utils::ResizBuf::new(&device, &adapter))
        .collect::<Vec<_>>();

    DebugTriangleData {
        hidden: false,
        triangles_count: 0,

        holes: vec![],

        posbuf_touch: 0,
        colbuf_touch: 0,
        tranbuf_touch: 0,
        rotbuf_touch: 0,
        scalebuf_touch: 0,

        posbuffer: vec![],   // 6 per triangle
        colbuffer: vec![],   // 12 per triangle
        tranbuffer: vec![],  // 6 per triangle
        rotbuffer: vec![],   // 3 per triangle
        scalebuffer: vec![], // 3 per triangle

        posbuf,
        colbuf,
        tranbuf,
        rotbuf,
        scalebuf,

        descriptor_set: triangle_descriptor_set_layouts,
        pipeline: ManuallyDrop::new(triangle_pipeline),
        pipeline_layout: ManuallyDrop::new(triangle_pipeline_layout),
        render_pass: ManuallyDrop::new(triangle_render_pass),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use cgmath::Deg;
    use logger::{Generic, GenericLogger, Logger};
    use test::{black_box, Bencher};

    // ---

    #[test]
    fn simple_triangle() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);
        let tri = DebugTriangle::default();

        vx.debtri().push(tri);
        utils::add_4_screencorners(&mut vx);

        let img = vx.draw_frame_copy_framebuffer(&prspect);

        utils::assert_swapchain_eq(&mut vx, "simple_triangle", img);
    }

    #[test]
    fn simple_triangle_pop() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);
        let tri = DebugTriangle::default();

        vx.debtri().push(tri);
        utils::add_4_screencorners(&mut vx);
        for _ in 0..4 {
            vx.debtri().pop();
        }

        let img = vx.draw_frame_copy_framebuffer(&prspect);

        utils::assert_swapchain_eq(&mut vx, "simple_triangle_middle", img);
    }

    #[test]
    fn simple_triangle_color() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);
        let tri = DebugTriangle::default();

        let triangle = vx.debtri().push(tri);
        vx.debtri().color(&triangle, [-255, 0, 0, 0]);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "simple_triangle_color", img);

        vx.debtri().color_all([0, 0, -128, 0]);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "simple_triangle_color_opacity", img);
    }

    #[test]
    fn test_single_triangle_api() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);
        let tri = DebugTriangle::default();

        let mut debtri = vx.debtri();
        let handle = debtri.push(tri);
        debtri.set_scale(&handle, 0.1);
        debtri.scale(&handle, 1.0);
        debtri.set_rotation(&handle, Deg(25.0));
        debtri.set_translation(&handle, (0.05, 0.4));
        debtri.translate(&handle, (0.2, 0.1));
        debtri.rotate(&handle, Deg(5.0));

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "test_single_triangle_api", img);
    }

    #[test]
    fn remove_middle_triangle() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);
        let tri = DebugTriangle::default();

        let mut debtri = vx.debtri();

        let left = debtri.push(tri);
        debtri.set_translation(&left, (-0.25, 0.0));
        debtri.set_color(&left, [255, 0, 0, 255]);

        let middle = debtri.push(tri);
        debtri.set_color(&middle, [0, 255, 0, 255]);

        let right = debtri.push(tri);
        debtri.set_translation(&right, (0.25, 0.0));
        debtri.set_color(&right, [0, 0, 255, 255]);

        debtri.remove(middle);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "remove_middle_triangle", img);
    }

    #[test]
    fn fill_remove_hole() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);
        let tri = DebugTriangle::default();

        let mut debtri = vx.debtri();

        let left = debtri.push(tri);
        debtri.set_translation(&left, (-0.25, 0.0));
        debtri.set_color(&left, [255, 0, 0, 255]);

        let middle = debtri.push(tri);
        debtri.set_color(&middle, [0, 255, 0, 255]);

        let right = debtri.push(tri);
        debtri.set_translation(&right, (0.25, 0.0));
        debtri.set_color(&right, [0, 0, 255, 255]);

        debtri.remove(middle);
        let middle = debtri.push(tri);
        debtri.set_rotation(&middle, Deg(60.0));

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "fill_remove_hole", img);
    }

    #[test]
    fn swap_triangles() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);
        let tri = DebugTriangle::default();

        let mut debtri = vx.debtri();

        let mut left = debtri.push(tri);
        debtri.set_translation(&left, (-0.25, 0.0));
        debtri.set_color(&left, [255, 0, 0, 255]);

        let mut right = debtri.push(tri);
        debtri.set_translation(&right, (0.25, 0.0));
        debtri.set_color(&right, [0, 0, 255, 255]);

        assert_eq![
            std::cmp::Ordering::Less,
            debtri.compare_triangle_order(&left, &right)
        ];
        debtri.swap(&mut left, &mut right);
        assert_eq![
            std::cmp::Ordering::Greater,
            debtri.compare_triangle_order(&left, &right)
        ];

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "swap_triangles", img);
    }

    #[test]
    fn deform_triangles() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);
        let tri = DebugTriangle::default();

        let mut debtri = vx.debtri();

        let left = debtri.push(tri);
        debtri.set_translation(&left, (-0.25, 0.0));
        debtri.set_color(&left, [255, 0, 0, 255]);

        let right = debtri.push(tri);
        debtri.set_translation(&right, (0.25, 0.0));
        debtri.set_color(&right, [0, 0, 255, 255]);

        debtri.set_deform(&right, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]);
        debtri.deform(&left, [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]);
        debtri.deform_all([(-1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "deform_triangles", img);
    }

    // ---

    #[test]
    fn simple_triangle_change_color() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);
        let tri = DebugTriangle::default();

        let mut debtri = vx.debtri();
        let idx = debtri.push(tri);
        debtri.set_color(&idx, [255, 0, 255, 255]);

        let img = vx.draw_frame_copy_framebuffer(&prspect);

        utils::assert_swapchain_eq(&mut vx, "simple_triangle_change_color", img);
    }

    #[test]
    fn debug_triangle_corners_widescreen() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless2x1k);
        let prspect = gen_perspective(&vx);

        for i in [-1f32, 1f32].iter() {
            for j in [-1f32, 1f32].iter() {
                let mut tri = DebugTriangle::default();
                tri.translation = (*i, *j);
                let _idx = vx.debtri().push(tri);
            }
        }

        let img = vx.draw_frame_copy_framebuffer(&prspect);

        utils::assert_swapchain_eq(&mut vx, "debug_triangle_corners_widescreen", img);
    }

    #[test]
    fn debug_triangle_corners_tallscreen() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1x2k);
        let prspect = gen_perspective(&vx);

        for i in [-1f32, 1f32].iter() {
            for j in [-1f32, 1f32].iter() {
                let mut tri = DebugTriangle::default();
                tri.translation = (*i, *j);
                let _idx = vx.debtri().push(tri);
            }
        }

        let img = vx.draw_frame_copy_framebuffer(&prspect);

        utils::assert_swapchain_eq(&mut vx, "debug_triangle_corners_tallscreen", img);
    }

    #[test]
    fn circle_of_triangles() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless2x1k);
        let prspect = gen_perspective(&vx);

        for i in 0..360 {
            let mut tri = DebugTriangle::default();
            tri.translation = ((i as f32).cos(), (i as f32).sin());
            tri.scale = 0.1f32;
            let _idx = vx.debtri().push(tri);
        }

        let img = vx.draw_frame_copy_framebuffer(&prspect);

        utils::assert_swapchain_eq(&mut vx, "circle_of_triangles", img);
    }

    #[test]
    fn triangle_in_corner() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut tri = DebugTriangle::default();
        tri.scale = 0.1f32;
        let radi = tri.radius();

        let trans = -1f32 + radi;
        for j in 0..31 {
            for i in 0..31 {
                tri.translation = (trans + i as f32 * 2.0 * radi, trans + j as f32 * 2.0 * radi);
                vx.debtri().push(tri);
            }
        }

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "triangle_in_corner", img);
    }

    #[test]
    fn windmills() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        utils::add_windmills(&mut vx, false);
        let img = vx.draw_frame_copy_framebuffer(&prspect);

        utils::assert_swapchain_eq(&mut vx, "windmills", img);
    }

    #[test]
    fn windmills_mass_edits() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        utils::add_windmills(&mut vx, false);
        let mut debtri = vx.debtri();

        debtri.translate_all((1.0, 0.5));
        debtri.rotate_all(Deg(90.0));
        debtri.scale_all(2.0);

        let img = vx.draw_frame_copy_framebuffer(&prspect);

        utils::assert_swapchain_eq(&mut vx, "windmills_mass_edits", img);
    }

    #[test]
    fn windmills_hidden() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        utils::add_windmills(&mut vx, false);

        vx.debtri().hide();

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "windmills_hidden", img);

        vx.debtri().show();

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "windmills_hidden_now_shown", img);
    }

    #[test]
    fn windmills_ignore_perspective() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless2x1k);
        let prspect = gen_perspective(&vx);

        utils::add_windmills(&mut vx, false);
        let img = vx.draw_frame_copy_framebuffer(&prspect);

        utils::assert_swapchain_eq(&mut vx, "windmills_ignore_perspective", img);
    }

    #[test]
    fn windmills_change_color() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let handles = utils::add_windmills(&mut vx, false);
        let mut debtri = vx.debtri();
        debtri.set_color(&handles[0], [255, 0, 0, 255]);
        debtri.set_color(&handles[249], [0, 255, 0, 255]);
        debtri.set_color(&handles[499], [0, 0, 255, 255]);
        debtri.set_color(&handles[999], [0, 0, 0, 255]);

        let img = vx.draw_frame_copy_framebuffer(&prspect);

        utils::assert_swapchain_eq(&mut vx, "windmills_change_color", img);
    }

    #[test]
    fn rotating_windmills_drawing_invariant() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        utils::add_windmills(&mut vx, false);
        for _ in 0..30 {
            vx.debtri().rotate_all(Deg(-1.0f32));
        }
        let img = vx.draw_frame_copy_framebuffer(&prspect);

        utils::assert_swapchain_eq(&mut vx, "rotating_windmills_drawing_invariant", img);
        utils::remove_windmills(&mut vx);

        utils::add_windmills(&mut vx, false);
        for _ in 0..30 {
            vx.debtri().rotate_all(Deg(-1.0f32));
            vx.draw_frame(&prspect);
        }
        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "rotating_windmills_drawing_invariant", img);
    }

    #[test]
    fn windmills_given_initial_rotation() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        utils::add_windmills(&mut vx, true);
        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "windmills_given_initial_rotation", img);
    }

    // ---

    #[bench]
    fn bench_simple_triangle(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        vx.debtri().push(DebugTriangle::default());
        utils::add_4_screencorners(&mut vx);

        b.iter(|| {
            vx.draw_frame(&prspect);
        });
    }

    #[bench]
    fn bench_still_windmills(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        utils::add_windmills(&mut vx, false);

        b.iter(|| {
            vx.draw_frame(&prspect);
        });
    }

    #[bench]
    fn bench_windmills_set_color(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let handles = utils::add_windmills(&mut vx, false);

        b.iter(|| {
            vx.debtri()
                .set_color(&handles[0], black_box([0, 0, 0, 255]));
        });
    }

    #[bench]
    fn bench_rotating_windmills(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        utils::add_windmills(&mut vx, false);

        b.iter(|| {
            vx.debtri().rotate_all(Deg(1.0f32));
            vx.draw_frame(&prspect);
        });
    }

    #[bench]
    fn bench_rotating_windmills_set_color(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let last = utils::add_windmills(&mut vx, false).pop().unwrap();

        b.iter(|| {
            vx.debtri().rotate_all(Deg(1.0f32));
            vx.debtri().set_color(&last, [255, 0, 255, 255]);
            vx.draw_frame(&prspect);
        });
    }

    #[bench]
    fn bench_rotating_windmills_no_render(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        utils::add_windmills(&mut vx, false);

        b.iter(|| {
            vx.debtri().rotate_all(Deg(1.0f32));
        });
    }
}
