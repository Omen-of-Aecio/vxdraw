//! Methods and types to control quads
//!
//! A quad is a renderable consisting of 4 points. Each point has a color and position associated
//! with it. By using different colors in the different points, the colors will "blend" into each
//! other. Opacity is also supported on quads.
//!
//! See [quads::Quads] for all operations supported on quads.
//!
//! # Example - Simple quad and some operations #
//! A showcase of basic operations on a quad.
//! ```
//! use vxdraw::{prelude::*, void_logger, Deg, Color, ShowWindow, VxDraw};
//! fn main() {
//!     #[cfg(feature = "doctest-headless")]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Headless1k);
//!     #[cfg(not(feature = "doctest-headless"))]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Enable);
//!
//!     // Create a new layer of quads
//!     let quad = vx.quads().add_layer(&vxdraw::quads::LayerOptions::new());
//!
//!     // Create a new quad
//!     let handle = vx.quads().add(&quad, vxdraw::quads::Quad::new());
//!
//!     // Turn the quad white
//!     vx.quads().set_solid_color(&handle, Color::Rgba(255, 255, 255, 255));
//!
//!     // Rotate the quad 45 degrees (counter clockwise)
//!     vx.quads().set_rotation(&handle, Deg(45.0));
//!
//!     // Scale the quad to half its current size
//!     vx.quads().scale(&handle, 0.5);
//!
//!     // Draw the frame
//!     vx.draw_frame();
//!
//!     // Sleep here so the window does not instantly disappear
//!     #[cfg(not(feature = "doctest-headless"))]
//!     std::thread::sleep(std::time::Duration::new(3, 0));
//! }
//! ```
//!
//! # Example - Curtain-like fade based on 4 quads #
//! This example moves 4 quads from the sides to "close" the scene as curtains would do.
//! ```
//! use vxdraw::{prelude::*, quads::*, void_logger, Deg, Matrix4, ShowWindow, VxDraw};
//!
//! fn main() {
//!     #[cfg(feature = "doctest-headless")]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Headless1k);
//!     #[cfg(not(feature = "doctest-headless"))]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Enable);
//!
//!     // Create a new layer of quads
//!     let layer = vx.quads().add_layer(&LayerOptions::new());
//!
//!     // The width of the faded quad, try changing this to 2.0, or 1.0 and observe
//!     let fade_width = 0.5;
//!
//!     // The left quad data, has the right vertices completely transparent
//!     let quad_data = Quad::new()
//!         .width(fade_width)
//!         .colors([
//!             (0, 0, 0, 255),
//!             (0, 0, 0, 255),
//!             (0, 0, 0, 0),
//!             (0, 0, 0, 0),
//!         ])
//!         .translation((- 1.0 - fade_width / 2.0, 0.0));
//!
//!     // Create a new quad
//!     let left_quad_fade = vx.quads().add(&layer, quad_data);
//!
//!     // The right quad data, has the left vertices completely transparent
//!     let quad_data = Quad::new()
//!         .width(fade_width)
//!         .colors([
//!             (0, 0, 0, 0),
//!             (0, 0, 0, 0),
//!             (0, 0, 0, 255),
//!             (0, 0, 0, 255),
//!         ])
//!         .translation((1.0 + fade_width / 2.0, 0.0));
//!
//!     // Create a new quad
//!     let right_quad_fade = vx.quads().add(&layer, quad_data);
//!
//!     // Now create the completely black quads
//!     let quad_data = Quad::new();
//!     let left_quad = vx.quads().add(&layer, quad_data);
//!     let right_quad = vx.quads().add(&layer, quad_data);
//!
//!     // Some math to ensure the faded quad and the solid quads move at the same rate, and that
//!     // both solid quads cover half the screen on the last frame.
//!     let fade_width_offscreen = 1.0 + fade_width / 2.0;
//!     let fade_pos_solid = 2.0 + fade_width;
//!     let nlscale = (1.0 + fade_width) / (1.0 + fade_width / 2.0);
//!
//!     // How many frames the entire animation takes, try making it shorter or longer
//!     let frames = 50;
//!
//!     for idx in 0..frames {
//!
//!         let perc = idx as f32 * nlscale;
//!         // Move the quads
//!         vx.quads().set_translation(&left_quad_fade, (-fade_width_offscreen + (fade_width_offscreen / frames as f32) * perc, 0.0));
//!         vx.quads().set_translation(&right_quad_fade, (fade_width_offscreen - (fade_width_offscreen / frames as f32) * perc, 0.0));
//!         vx.quads().set_translation(&left_quad, (-fade_pos_solid + (fade_width_offscreen / frames as f32) * perc, 0.0));
//!         vx.quads().set_translation(&right_quad, (fade_pos_solid - (fade_width_offscreen / frames as f32) * perc, 0.0));
//!
//!         vx.draw_frame();
//!
//!         // Sleep so we can see some animation
//!         #[cfg(not(feature = "doctest-headless"))]
//!         std::thread::sleep(std::time::Duration::new(0, 16_000_000));
//!     }
//! }
//! ```
//!
//! Note how the above has two overlapping, faded quads. This can be an undesired animation
//! artifact. The intent of the example is to show how to work with the library.
use super::{blender, utils::*, Color};
use crate::data::{DrawType, QuadsData, VxDraw};
use cgmath::{Matrix4, Rad};
use core::ptr::read;
#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
use gfx_backend_gl as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use gfx_hal::{device::Device, format, image, pass, pso, Backend, Primitive};
use std::{io::Cursor, mem::ManuallyDrop};

// ---

/// Handle referring to a single quad
#[derive(Debug)]
pub struct Handle(usize, usize);

/// Handle referring to a quad layer
#[derive(Debug)]
pub struct Layer(usize);

impl Layerable for Layer {
    fn get_layer(&self, vx: &VxDraw) -> usize {
        for (idx, ord) in vx.draw_order.iter().enumerate() {
            match ord {
                DrawType::Quad { id } if *id == self.0 => {
                    return idx;
                }
                _ => {}
            }
        }
        panic!["Unable to get layer"]
    }
}

/// Enum describing which vertex shader to use
#[derive(Clone, Debug)]
pub enum VertexShader {
    /// Use the given SPIRV code
    Spirv(Vec<u8>),
    /// Use the shader provided by `vxdraw`
    Standard,
}

/// Enum describing which fragment shader to use
#[derive(Clone, Debug)]
pub enum FragmentShader {
    /// Use the given SPIRV code
    Spirv(Vec<u8>),
    /// Use the shader provided by `vxdraw`
    Standard,
}

/// Options for creating a layer of quads
#[derive(Debug)]
pub struct LayerOptions {
    depth_test: bool,
    hide: bool,
    blend: blender::Blender,
    fixed_perspective: Option<Matrix4<f32>>,
    vertex_shader: VertexShader,
    fragment_shader: FragmentShader,
}

impl Default for LayerOptions {
    fn default() -> Self {
        Self {
            depth_test: false,
            hide: false,
            blend: blender::Blender::default(),
            fixed_perspective: None,
            vertex_shader: VertexShader::Standard,
            fragment_shader: FragmentShader::Standard,
        }
    }
}

impl LayerOptions {
    /// Same as default
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vertex shader
    pub fn vertex_shader(mut self, shader: VertexShader) -> Self {
        self.vertex_shader = shader;
        self
    }

    /// Set the fragment shader
    pub fn fragment_shader(mut self, shader: FragmentShader) -> Self {
        self.fragment_shader = shader;
        self
    }

    /// Hide this layer
    pub fn hide(mut self) -> Self {
        self.hide = true;
        self
    }

    /// Show this layer (default)
    pub fn show(mut self) -> Self {
        self.hide = false;
        self
    }

    /// Set a fixed perspective for this layer
    pub fn fixed_perspective(mut self, mat: Matrix4<f32>) -> Self {
        self.fixed_perspective = Some(mat);
        self
    }

    /// Set the blender of this layer (see [blender])
    pub fn blend(mut self, blend_setter: impl Fn(blender::Blender) -> blender::Blender) -> Self {
        self.blend = blend_setter(self.blend);
        self
    }
}

// ---

/// Quad information used for creating and getting
#[derive(Clone, Copy, Debug)]
pub struct Quad {
    width: f32,
    height: f32,
    depth: f32,
    colors: [(u8, u8, u8, u8); 4],
    translation: (f32, f32),
    rotation: f32,
    scale: f32,
    /// Moves the origin of the quad to some point, for instance, you may want a corner of the quad
    /// to be the origin. This affects rotation and translation of the quad.
    origin: (f32, f32),
}

impl Quad {
    /// Same as default
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the width of the quad
    pub fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// Set the height of the quad
    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// Set the colors of the quad
    ///
    /// The colors are added on top of whatever the quad's texture data is
    pub fn colors(mut self, colors: [(u8, u8, u8, u8); 4]) -> Self {
        self.colors = colors;
        self
    }

    /// Set the translation
    pub fn translation(mut self, trn: (f32, f32)) -> Self {
        self.translation = trn;
        self
    }

    /// Set the rotation. Rotation is counter-clockwise
    pub fn rotation<T: Copy + Into<Rad<f32>>>(mut self, angle: T) -> Self {
        self.rotation = angle.into().0;
        self
    }

    /// Set the scaling factor of this quad
    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set the origin of this quad
    pub fn origin(mut self, origin: (f32, f32)) -> Self {
        self.origin = origin;
        self
    }
}

impl Default for Quad {
    fn default() -> Self {
        Quad {
            width: 2.0,
            height: 2.0,
            depth: 0.0,
            colors: [(0, 0, 0, 255); 4],
            translation: (0.0, 0.0),
            rotation: 0.0,
            scale: 1.0,
            origin: (0.0, 0.0),
        }
    }
}

// ---

/// Accessor object to all quads
///
/// A quad is a colored object with 4 points.
/// See [crate::quads] for examples.
pub struct Quads<'a> {
    vx: &'a mut VxDraw,
}

impl<'a> Quads<'a> {
    /// Spawn the accessor object from [VxDraw].
    ///
    /// This is a very cheap operation.
    pub(crate) fn new(vx: &'a mut VxDraw) -> Self {
        Self { vx }
    }

    /// Compare quad draw order
    ///
    /// All quads are drawn in a specific order. This method figures out which order is used
    /// between two quads. The order can be manipulated by [Quads::swap_draw_order].
    pub fn compare_draw_order(&self, left: &Handle, right: &Handle) -> std::cmp::Ordering {
        let layer_ordering = left.0.cmp(&right.0);
        if layer_ordering == std::cmp::Ordering::Equal {
            left.1.cmp(&right.1)
        } else {
            layer_ordering
        }
    }

    /// Swap two quads with each other
    ///
    /// Swaps the internal data of each quad (all vertices and their data, translation,
    /// and so on). The effect of this is that the draw order is swapped too, meaning that the
    /// quads reverse order (one drawn on top of the other).
    ///
    /// This function can swap quads from two different layers, but also quads in the same
    /// layer.
    pub fn swap_draw_order(&mut self, left: &mut Handle, right: &mut Handle) {
        let q1d = self.vx.quads[left.0].posbuffer[left.1];
        let q2d = self.vx.quads[right.0].posbuffer[right.1];
        self.vx.quads[left.0].posbuffer[left.1] = q2d;
        self.vx.quads[right.0].posbuffer[right.1] = q1d;

        let q1d = self.vx.quads[left.0].colbuffer[left.1];
        let q2d = self.vx.quads[right.0].colbuffer[right.1];
        self.vx.quads[left.0].colbuffer[left.1] = q2d;
        self.vx.quads[right.0].colbuffer[right.1] = q1d;

        let q1d = self.vx.quads[left.0].tranbuffer[left.1];
        let q2d = self.vx.quads[right.0].tranbuffer[right.1];
        self.vx.quads[left.0].tranbuffer[left.1] = q2d;
        self.vx.quads[right.0].tranbuffer[right.1] = q1d;

        let q1d = self.vx.quads[left.0].rotbuffer[left.1];
        let q2d = self.vx.quads[right.0].rotbuffer[right.1];
        self.vx.quads[left.0].rotbuffer[left.1] = q2d;
        self.vx.quads[right.0].rotbuffer[right.1] = q1d;

        let q1d = self.vx.quads[left.0].scalebuffer[left.1];
        let q2d = self.vx.quads[right.0].scalebuffer[right.1];
        self.vx.quads[left.0].scalebuffer[left.1] = q2d;
        self.vx.quads[right.0].scalebuffer[right.1] = q1d;

        self.vx.quads[left.0].posbuf_touch = self.vx.swapconfig.image_count;
        self.vx.quads[left.0].colbuf_touch = self.vx.swapconfig.image_count;
        self.vx.quads[left.0].tranbuf_touch = self.vx.swapconfig.image_count;
        self.vx.quads[left.0].rotbuf_touch = self.vx.swapconfig.image_count;
        self.vx.quads[left.0].scalebuf_touch = self.vx.swapconfig.image_count;

        self.vx.quads[right.0].posbuf_touch = self.vx.swapconfig.image_count;
        self.vx.quads[right.0].colbuf_touch = self.vx.swapconfig.image_count;
        self.vx.quads[right.0].tranbuf_touch = self.vx.swapconfig.image_count;
        self.vx.quads[right.0].rotbuf_touch = self.vx.swapconfig.image_count;
        self.vx.quads[right.0].scalebuf_touch = self.vx.swapconfig.image_count;

        std::mem::swap(&mut left.0, &mut right.0);
        std::mem::swap(&mut left.1, &mut right.1);
    }

    /// Create a new layer for quads
    ///
    /// This new layer will be ordered on top of all previous layers, meaning that its quads will
    /// be drawn on top of all other drawn items. If another layer is created, that layer will be
    /// drawn on top of this layer, and so on.
    pub fn add_layer(&mut self, options: &LayerOptions) -> Layer {
        let s = &mut *self.vx;
        pub const VERTEX_SOURCE: &[u8] = include_bytes!["../target/spirv/quads.vert.spirv"];

        pub const FRAGMENT_SOURCE: &[u8] = include_bytes!["../target/spirv/quads.frag.spirv"];

        let vertex_source = match options.vertex_shader {
            VertexShader::Standard => pso::read_spirv(Cursor::new(VERTEX_SOURCE)).unwrap(),
            VertexShader::Spirv(ref data) => pso::read_spirv(Cursor::new(data)).unwrap(),
        };
        let fragment_source = match options.fragment_shader {
            FragmentShader::Standard => pso::read_spirv(Cursor::new(FRAGMENT_SOURCE)).unwrap(),
            FragmentShader::Spirv(ref data) => pso::read_spirv(Cursor::new(data)).unwrap(),
        };

        let vs_module = { unsafe { s.device.create_shader_module(&vertex_source) }.unwrap() };
        let fs_module = { unsafe { s.device.create_shader_module(&fragment_source) }.unwrap() };

        // Describe the shaders
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

        let vertex_buffers: Vec<pso::VertexBufferDesc> = vec![
            pso::VertexBufferDesc {
                binding: 0,
                stride: 2 * 4,
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
            front_face: pso::FrontFace::Clockwise,
            depth_bias: None,
            conservative: false,
        };

        let depth_stencil = pso::DepthStencilDesc {
            depth: if options.depth_test {
                Some(pso::DepthTest {
                    fun: pso::Comparison::LessEqual,
                    write: true,
                })
            } else {
                None
            },
            depth_bounds: false,
            stencil: None,
        };
        let blender = options.blend.clone().into_gfx_blender();
        let quad_render_pass = {
            let attachment = pass::Attachment {
                format: Some(s.format),
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

            unsafe {
                s.device
                    .create_render_pass(&[attachment, depth], &[subpass], &[])
            }
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
        let quad_descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> =
            vec![unsafe {
                s.device
                    .create_descriptor_set_layout(bindings, immutable_samplers)
                    .expect("Couldn't make a DescriptorSetLayout")
            }];
        let mut push_constants = Vec::<(pso::ShaderStageFlags, std::ops::Range<u32>)>::new();
        push_constants.push((pso::ShaderStageFlags::VERTEX, 0..16));

        let quad_pipeline_layout = unsafe {
            s.device
                .create_pipeline_layout(&quad_descriptor_set_layouts, push_constants)
                .expect("Couldn't create a pipeline layout")
        };

        // Describe the pipeline (rasterization, quad interpretation)
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
            layout: &quad_pipeline_layout,
            subpass: pass::Subpass {
                index: 0,
                main_pass: &quad_render_pass,
            },
            flags: pso::PipelineCreationFlags::empty(),
            parent: pso::BasePipeline::None,
        };

        let quad_pipeline = unsafe {
            s.device
                .create_graphics_pipeline(&pipeline_desc, None)
                .expect("Couldn't create a graphics pipeline!")
        };

        unsafe {
            s.device.destroy_shader_module(vs_module);
            s.device.destroy_shader_module(fs_module);
        }

        let image_count = s.swapconfig.image_count;
        let posbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&s.device, &s.adapter))
            .collect::<Vec<_>>();
        let colbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&s.device, &s.adapter))
            .collect::<Vec<_>>();
        let tranbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&s.device, &s.adapter))
            .collect::<Vec<_>>();
        let rotbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&s.device, &s.adapter))
            .collect::<Vec<_>>();
        let scalebuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&s.device, &s.adapter))
            .collect::<Vec<_>>();

        let indices = (0..image_count)
            .map(|_| super::utils::ResizBufIdx4::new(&s.device, &s.adapter))
            .collect::<Vec<_>>();

        let quads = QuadsData {
            hidden: options.hide,

            fixed_perspective: options.fixed_perspective,
            holes: vec![],

            posbuf_touch: 0,
            colbuf_touch: 0,
            tranbuf_touch: 0,
            rotbuf_touch: 0,
            scalebuf_touch: 0,

            posbuffer: vec![],
            colbuffer: vec![],
            tranbuffer: vec![],
            rotbuffer: vec![],
            scalebuffer: vec![],

            posbuf,
            colbuf,
            tranbuf,
            rotbuf,
            scalebuf,

            indices,

            descriptor_set: quad_descriptor_set_layouts,
            pipeline: ManuallyDrop::new(quad_pipeline),
            pipeline_layout: ManuallyDrop::new(quad_pipeline_layout),
            render_pass: ManuallyDrop::new(quad_render_pass),
        };

        let prev_layer = s.layer_holes.find_available(|x| match x {
            DrawType::Quad { .. } => true,
            _ => false,
        });

        if let Some(prev_layer) = prev_layer {
            match prev_layer {
                DrawType::Quad { id } => {
                    let old_quad = std::mem::replace(&mut s.quads[id], quads);
                    destroy_layer(s, old_quad);
                    s.draw_order.push(DrawType::Quad { id });
                    Layer(id)
                }
                _ => panic!["Got a non-quads drawtype, should be impossible!"],
            }
        } else {
            s.quads.push(quads);
            s.draw_order.push(DrawType::Quad {
                id: s.quads.len() - 1,
            });
            Layer(s.quads.len() - 1)
        }
    }

    /// Query the amount of layers of this type there are
    pub fn layer_count(&self) -> usize {
        self.vx.quads.len()
    }

    /// Disable drawing of the quads at this layer
    pub fn hide(&mut self, layer: &Layer) {
        self.vx.quads[layer.0].hidden = true;
    }

    /// Enable drawing of the quads at this layer
    pub fn show(&mut self, layer: &Layer) {
        self.vx.quads[layer.0].hidden = false;
    }

    /// Add a new quad to the given layer
    ///
    /// The new quad will be based on the data in [Quad], and inserted into the given [Layer].
    pub fn add(&mut self, layer: &Layer, quad: Quad) -> Handle {
        let width = quad.width;
        let height = quad.height;

        let topleft = (
            -width / 2f32 - quad.origin.0,
            -height / 2f32 - quad.origin.1,
        );
        let topright = (width / 2f32 - quad.origin.0, -height / 2f32 - quad.origin.1);
        let bottomleft = (-width / 2f32 - quad.origin.0, height / 2f32 - quad.origin.1);
        let bottomright = (width / 2f32 - quad.origin.0, height / 2f32 - quad.origin.1);
        let replace = self.vx.quads.get(layer.0).map(|x| !x.holes.is_empty());
        if replace.is_none() {
            panic!["Layer does not exist"];
        }
        let handle = if replace.unwrap() {
            let hole = self.vx.quads.get_mut(layer.0).unwrap().holes.pop().unwrap();
            let handle = Handle(layer.0, hole);
            self.set_deform(
                &handle,
                [
                    (topleft.0, topleft.1),
                    (bottomleft.0, bottomleft.1),
                    (bottomright.0, bottomright.1),
                    (topright.0, topright.1),
                ],
            );
            self.set_color(
                &handle,
                [
                    Color::Rgba(
                        quad.colors[0].0,
                        quad.colors[0].1,
                        quad.colors[0].2,
                        quad.colors[0].3,
                    ),
                    Color::Rgba(
                        quad.colors[1].0,
                        quad.colors[1].1,
                        quad.colors[1].2,
                        quad.colors[1].3,
                    ),
                    Color::Rgba(
                        quad.colors[2].0,
                        quad.colors[2].1,
                        quad.colors[2].2,
                        quad.colors[2].3,
                    ),
                    Color::Rgba(
                        quad.colors[3].0,
                        quad.colors[3].1,
                        quad.colors[3].2,
                        quad.colors[3].3,
                    ),
                ],
            );
            self.set_translation(&handle, (quad.translation.0, quad.translation.1));
            self.set_rotation(&handle, Rad(quad.rotation));
            self.set_scale(&handle, quad.scale);
            handle
        } else {
            let quads = self.vx.quads.get_mut(layer.0).unwrap();
            quads.posbuffer.push([
                topleft.0,
                topleft.1,
                bottomleft.0,
                bottomleft.1,
                bottomright.0,
                bottomright.1,
                topright.0,
                topright.1,
            ]);
            quads.colbuffer.push([
                quad.colors[0].0,
                quad.colors[0].1,
                quad.colors[0].2,
                quad.colors[0].3,
                quad.colors[1].0,
                quad.colors[1].1,
                quad.colors[1].2,
                quad.colors[1].3,
                quad.colors[2].0,
                quad.colors[2].1,
                quad.colors[2].2,
                quad.colors[2].3,
                quad.colors[3].0,
                quad.colors[3].1,
                quad.colors[3].2,
                quad.colors[3].3,
            ]);
            quads.tranbuffer.push([
                quad.translation.0,
                quad.translation.1,
                quad.translation.0,
                quad.translation.1,
                quad.translation.0,
                quad.translation.1,
                quad.translation.0,
                quad.translation.1,
            ]);
            quads
                .rotbuffer
                .push([quad.rotation, quad.rotation, quad.rotation, quad.rotation]);
            quads
                .scalebuffer
                .push([quad.scale, quad.scale, quad.scale, quad.scale]);

            Handle(layer.0, quads.posbuffer.len() - 1)
        };

        let quads = self.vx.quads.get_mut(layer.0).unwrap();
        quads.posbuf_touch = self.vx.swapconfig.image_count;
        quads.colbuf_touch = self.vx.swapconfig.image_count;
        quads.tranbuf_touch = self.vx.swapconfig.image_count;
        quads.rotbuf_touch = self.vx.swapconfig.image_count;
        quads.scalebuf_touch = self.vx.swapconfig.image_count;

        handle
    }

    /// Remove a layer of quads
    ///
    /// Removes the quad layer from memory and destroys all quads associated with it.
    /// All lingering quad handles that were spawned using this layer will be invalidated.
    pub fn remove_layer(&mut self, layer: Layer) {
        let s = &mut *self.vx;
        let mut index = None;
        for (idx, x) in s.draw_order.iter().enumerate() {
            match x {
                DrawType::Quad { id } if *id == layer.0 => {
                    index = Some(idx);
                    break;
                }
                _ => {}
            }
        }
        if let Some(idx) = index {
            let draw_type = s.draw_order.remove(idx);
            s.layer_holes.push(draw_type);
        }
    }

    /// Remove a quad
    ///
    /// The quad is set to a scale of 0 and its handle is stored internally in a list of
    /// `holes`. Calling [Quads::add] with available holes will fill the first available hole
    /// with the new quad.
    pub fn remove(&mut self, handle: Handle) {
        self.vx.quads[handle.0].holes.push(handle.1);
        self.set_scale(&handle, 0.0);
    }

    // ---

    /// Change the vertices of the model-space
    ///
    /// The name `set_deform` is used to keep consistent [Quads::deform].
    /// What this function does is just setting absolute vertex positions for each vertex in the
    /// quad.
    pub fn set_deform(&mut self, handle: &Handle, points: [(f32, f32); 4]) {
        self.vx.quads[handle.0].posbuf_touch = self.vx.swapconfig.image_count;
        let vertex = &mut self.vx.quads[handle.0].posbuffer[handle.1];
        for (idx, point) in points.iter().enumerate() {
            vertex[idx * 2] = point.0;
            vertex[idx * 2 + 1] = point.1;
        }
    }

    /// Set a solid color of a quad
    pub fn set_solid_color(&mut self, handle: &Handle, rgba: Color) {
        self.vx.quads[handle.0].colbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            let Color::Rgba(r, g, b, a) = rgba;
            self.vx.quads[handle.0].colbuffer[handle.1][idx * 4..(idx + 1) * 4]
                .copy_from_slice(&[r, g, b, a]);
        }
    }

    /// Set a solid color each vertex of a quad
    pub fn set_color(&mut self, handle: &Handle, rgba: [Color; 4]) {
        self.vx.quads[handle.0].colbuf_touch = self.vx.swapconfig.image_count;
        for (idx, dt) in rgba.iter().enumerate() {
            let Color::Rgba(r, g, b, a) = dt;
            self.vx.quads[handle.0].colbuffer[handle.1][idx * 4..(idx + 1) * 4]
                .copy_from_slice(&[*r, *g, *b, *a]);
        }
    }

    /// Set the position (translation) of a quad
    ///
    /// The name `set_translation` is chosen to keep the counterparts [Quads::translate] and
    /// `translate_all` consistent. This function can purely be thought of as setting the position
    /// of the quad with respect to the model-space's origin.
    pub fn set_translation(&mut self, handle: &Handle, position: (f32, f32)) {
        self.vx.quads[handle.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            self.vx.quads[handle.0].tranbuffer[handle.1][idx * 2] = position.0;
            self.vx.quads[handle.0].tranbuffer[handle.1][idx * 2 + 1] = position.1;
        }
    }

    /// Set the rotation of a quad
    ///
    /// The rotation is about the model space origin.
    pub fn set_rotation<T: Copy + Into<Rad<f32>>>(&mut self, handle: &Handle, angle: T) {
        let angle = angle.into().0;
        self.vx.quads[handle.0].rotbuf_touch = self.vx.swapconfig.image_count;
        self.vx.quads[handle.0].rotbuffer[handle.1].copy_from_slice(&[angle, angle, angle, angle]);
    }

    /// Set the scale of a quad
    pub fn set_scale(&mut self, handle: &Handle, scale: f32) {
        self.vx.quads[handle.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for sc in &mut self.vx.quads[handle.0].scalebuffer[handle.1] {
            *sc = scale;
        }
    }

    // ---

    /// Deform a quad by adding delta vertices
    ///
    /// Adds the delta vertices to the quad. Beware: This changes model space form.
    pub fn deform(&mut self, handle: &Handle, delta: [(f32, f32); 4]) {
        self.vx.quads[handle.0].posbuf_touch = self.vx.swapconfig.image_count;
        let points = &mut self.vx.quads[handle.0].posbuffer[handle.1];
        points[0] += delta[0].0;
        points[1] += delta[0].1;
        points[2] += delta[1].0;
        points[3] += delta[1].1;
        points[4] += delta[2].0;
        points[5] += delta[2].1;
        points[6] += delta[3].0;
        points[7] += delta[3].1;
    }

    /// Translate a quad by a vector
    ///
    /// Translation does not mutate the model-space of a quad.
    pub fn translate(&mut self, handle: &Handle, movement: (f32, f32)) {
        self.vx.quads[handle.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            self.vx.quads[handle.0].tranbuffer[handle.1][idx * 2] += movement.0;
            self.vx.quads[handle.0].tranbuffer[handle.1][idx * 2 + 1] += movement.1;
        }
    }

    /// Rotate a quad
    ///
    /// Rotation does not mutate the model-space of a quad.
    pub fn rotate<T: Copy + Into<Rad<f32>>>(&mut self, handle: &Handle, deg: T) {
        self.vx.quads[handle.0].rotbuf_touch = self.vx.swapconfig.image_count;
        for rot in &mut self.vx.quads[handle.0].rotbuffer[handle.1] {
            *rot += deg.into().0;
        }
    }

    /// Scale a quad
    ///
    /// Scale does not mutate the model-space of a quad.
    pub fn scale(&mut self, handle: &Handle, scale: f32) {
        self.vx.quads[handle.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for sc in &mut self.vx.quads[handle.0].scalebuffer[handle.1] {
            *sc *= scale;
        }
    }

    // ---

    /// Deform all quads by adding delta vertices
    ///
    /// Applies [Quads::deform] to each quad.
    pub fn deform_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> [(f32, f32); 4]) {
        self.vx.quads[layer.0].posbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.quads[layer.0].posbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            quad[0] += delta[0].0;
            quad[1] += delta[0].1;
            quad[2] += delta[1].0;
            quad[3] += delta[1].1;
            quad[4] += delta[2].0;
            quad[5] += delta[2].1;
            quad[6] += delta[3].0;
            quad[7] += delta[3].1;
        }
    }

    /// Translate all quads by adding delta vertices
    ///
    /// Applies [Quads::translate] to each quad.
    pub fn translate_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> (f32, f32)) {
        self.vx.quads[layer.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.quads[layer.0].tranbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for idx in 0..4 {
                quad[idx * 2] += delta.0;
                quad[idx * 2 + 1] += delta.1;
            }
        }
    }

    /// Rotate all quads by adding delta rotations
    ///
    /// Applies [Quads::rotate] to each quad.
    pub fn rotate_all<T: Copy + Into<Rad<f32>>>(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> T,
    ) {
        self.vx.quads[layer.0].rotbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.quads[layer.0].rotbuffer.iter_mut().enumerate() {
            let delta = delta(idx).into().0;
            for rotation in quad.iter_mut() {
                *rotation += delta;
            }
        }
    }

    /// Scale all quads by multiplying a delta scale
    ///
    /// Applies [Quads::scale] to each quad.
    pub fn scale_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> f32) {
        self.vx.quads[layer.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.quads[layer.0].scalebuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for scale in quad.iter_mut() {
                *scale *= delta;
            }
        }
    }

    // ---

    /// Deform all quads by setting delta vertices
    ///
    /// Applies [Quads::set_deform] to each quad.
    pub fn set_deform_all(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> [(f32, f32); 4],
    ) {
        self.vx.quads[layer.0].posbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.quads[layer.0].posbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            quad[0] = delta[0].0;
            quad[1] = delta[0].1;
            quad[2] = delta[1].0;
            quad[3] = delta[1].1;
            quad[4] = delta[2].0;
            quad[5] = delta[2].1;
            quad[6] = delta[3].0;
            quad[7] = delta[3].1;
        }
    }

    /// Set the color on all quads
    ///
    /// Applies [Quads::set_solid_color] to each quad.
    pub fn set_solid_color_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> Color) {
        self.vx.quads[layer.0].colbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.quads[layer.0].colbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for idx in 0..4 {
                let Color::Rgba(r, g, b, a) = delta;
                quad[idx * 4..(idx + 1) * 4].copy_from_slice(&[r, g, b, a]);
            }
        }
    }

    /// Set the color on all quads (for each vertex)
    ///
    /// Applies [Quads::set_color] to each quad.
    pub fn set_color_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> [Color; 4]) {
        self.vx.quads[layer.0].colbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.quads[layer.0].colbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for idx in 0..4 {
                let Color::Rgba(r, g, b, a) = delta[idx];
                quad[idx * 4..(idx + 1) * 4].copy_from_slice(&[r, g, b, a]);
            }
        }
    }

    /// Set the translation on all quads
    ///
    /// Applies [Quads::set_translation] to each quad.
    pub fn set_translation_all(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> (f32, f32),
    ) {
        self.vx.quads[layer.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.quads[layer.0].tranbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for idx in 0..4 {
                quad[idx * 2] = delta.0;
                quad[idx * 2 + 1] = delta.1;
            }
        }
    }

    /// Set the rotation on all quads
    ///
    /// Applies [Quads::set_rotation] to each quad.
    pub fn set_rotation_all<T: Copy + Into<Rad<f32>>>(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> T,
    ) {
        self.vx.quads[layer.0].rotbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.quads[layer.0].rotbuffer.iter_mut().enumerate() {
            let delta = delta(idx).into().0;
            quad.copy_from_slice(&[delta; 4]);
        }
    }

    /// Set the scale on all quads
    ///
    /// Applies [Quads::set_scale] to each quad.
    /// Note: This may re-enable removed quads, see [Quads::remove].
    pub fn set_scale_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> f32) {
        self.vx.quads[layer.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.quads[layer.0].scalebuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            quad.copy_from_slice(&[delta; 4]);
        }
    }
}

// ---

fn destroy_layer(s: &mut VxDraw, mut quad: QuadsData) {
    unsafe {
        for mut indices in quad.indices.drain(..) {
            indices.destroy(&s.device);
        }
        for mut posbuf in quad.posbuf.drain(..) {
            posbuf.destroy(&s.device);
        }
        for mut colbuf in quad.colbuf.drain(..) {
            colbuf.destroy(&s.device);
        }
        for mut tranbuf in quad.tranbuf.drain(..) {
            tranbuf.destroy(&s.device);
        }
        for mut rotbuf in quad.rotbuf.drain(..) {
            rotbuf.destroy(&s.device);
        }
        for mut scalebuf in quad.scalebuf.drain(..) {
            scalebuf.destroy(&s.device);
        }
        for dsl in quad.descriptor_set.drain(..) {
            s.device.destroy_descriptor_set_layout(dsl);
        }
        s.device
            .destroy_graphics_pipeline(ManuallyDrop::into_inner(read(&quad.pipeline)));
        s.device
            .destroy_pipeline_layout(ManuallyDrop::into_inner(read(&quad.pipeline_layout)));
        s.device
            .destroy_render_pass(ManuallyDrop::into_inner(read(&quad.render_pass)));
    }
}

// ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use cgmath::Deg;
    use fast_logger::{Generic, GenericLogger, Logger};

    #[test]
    fn simple_quad() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut quad = quads::Quad::new();
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        let layer = vx.quads().add_layer(&LayerOptions::new());
        vx.quads().add(&layer, quad);

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "simple_quad", img);
    }

    #[test]
    fn simple_quad_hide() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let quad = quads::Quad::new();

        let layer = vx.quads().add_layer(&LayerOptions::new());
        vx.quads().add(&layer, quad);
        vx.quads().hide(&layer);

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "simple_quad_hide", img);
    }

    #[test]
    fn simple_quad_translated() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut quad = quads::Quad::new();
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        let mut quads = vx.quads();
        let layer = quads.add_layer(&LayerOptions::new());
        let handle = quads.add(&layer, quad);
        quads.translate(&handle, (0.25, 0.4));

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "simple_quad_translated", img);
    }

    #[test]
    fn swapping_quad_draw_order() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let quad = quads::Quad::new();
        let layer = vx.quads().add_layer(&LayerOptions::new());
        let mut q1 = vx.quads().add(&layer, quad);
        let mut q2 = vx.quads().add(&layer, quad);

        let mut quads = vx.quads();
        quads.translate(&q1, (-0.5, -0.5));
        quads.set_solid_color(&q1, Color::Rgba(255, 0, 255, 255));
        quads.translate(&q2, (0.5, 0.5));
        quads.set_solid_color(&q2, Color::Rgba(0, 255, 255, 128));

        assert_eq![std::cmp::Ordering::Less, quads.compare_draw_order(&q1, &q2)];
        quads.swap_draw_order(&mut q1, &mut q2);
        assert_eq![
            std::cmp::Ordering::Greater,
            quads.compare_draw_order(&q1, &q2)
        ];

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "swapping_quad_draw_order", img);
    }

    #[test]
    fn swapping_quad_draw_order_different_layers() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let quad = quads::Quad::new();
        let layer1 = vx.quads().add_layer(&LayerOptions::new());
        let layer2 = vx.quads().add_layer(&LayerOptions::new());
        let mut q1 = vx.quads().add(&layer1, quad);
        let mut q2 = vx.quads().add(&layer2, quad);

        let mut quads = vx.quads();
        quads.translate(&q1, (-0.5, -0.5));
        quads.set_solid_color(&q1, Color::Rgba(255, 0, 255, 255));
        quads.translate(&q2, (0.5, 0.5));
        quads.set_solid_color(&q2, Color::Rgba(0, 255, 255, 128));

        assert_eq![std::cmp::Ordering::Less, quads.compare_draw_order(&q1, &q2)];
        quads.swap_draw_order(&mut q1, &mut q2);
        assert_eq![
            std::cmp::Ordering::Greater,
            quads.compare_draw_order(&q1, &q2)
        ];

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "swapping_quad_draw_order_different_layers", img);
    }

    #[test]
    fn three_quads_add_remove() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut quad = quads::Quad::new();
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        let mut quads = vx.quads();
        let layer = quads.add_layer(&LayerOptions::new());
        let _q1 = quads.add(&layer, quad);
        let q2 = quads.add(&layer, quad);
        let q3 = quads.add(&layer, quad);

        quads.translate(&q2, (0.25, 0.4));
        quads.set_solid_color(&q2, Color::Rgba(0, 0, 255, 128));

        quads.translate(&q3, (0.35, 0.8));
        quads.remove(q2);

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "three_quads_add_remove", img);
    }

    #[test]
    fn three_quads_add_remove_layer() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut quad = quads::Quad::new();
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        let mut quads = vx.quads();
        let layer1 = quads.add_layer(&LayerOptions::new());
        let _q1 = quads.add(&layer1, quad);
        let layer2 = quads.add_layer(&LayerOptions::new());
        let q2 = quads.add(&layer2, quad);
        let layer3 = quads.add_layer(&LayerOptions::new());
        let q3 = quads.add(&layer3, quad);

        quads.translate(&q2, (0.25, 0.4));
        quads.set_solid_color(&q2, Color::Rgba(0, 0, 255, 128));

        quads.translate(&q3, (0.35, 0.8));
        quads.remove_layer(layer2);

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "three_quads_add_remove_layer", img);
    }

    #[test]
    fn simple_quad_set_position() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut quad = quads::Quad::new();
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        let mut quads = vx.quads();
        let layer = quads.add_layer(&LayerOptions::new());
        let handle = quads.add(&layer, quad);
        quads.set_translation(&handle, (0.25, 0.4));

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "simple_quad_set_position", img);
    }

    #[test]
    fn simple_quad_scale() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut quad = quads::Quad::new();
        quad.colors[0].1 = 255;
        quad.colors[2].2 = 255;

        let mut quads = vx.quads();
        let layer = quads.add_layer(&LayerOptions::new());
        let handle = quads.add(&layer, quad);
        quads.set_scale(&handle, 0.5);

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "simple_quad_scale", img);
    }

    #[test]
    fn simple_quad_deform() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut quad = quads::Quad::new();
        quad.colors[0].1 = 255;
        quad.colors[2].2 = 255;

        let mut quads = vx.quads();
        let layer = quads.add_layer(&LayerOptions::new());
        let handle = quads.add(&layer, quad);
        quads.scale(&handle, 0.5);
        quads.deform(&handle, [(-0.5, 0.0), (0.0, 0.0), (0.0, 0.0), (0.5, 0.1)]);

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "simple_quad_deform", img);
    }

    #[test]
    fn set_color_all() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let quad = quads::Quad::new();

        let mut quads = vx.quads();
        let layer = quads.add_layer(&LayerOptions::new());
        quads.add(&layer, quad.scale(0.5).translation((-0.5, 0.0)));
        quads.add(&layer, quad.scale(0.5).translation((0.5, 0.0)));

        quads.set_color_all(&layer, |idx| match idx {
            0 => [
                Color::Rgba(0, 0, 0, 255),
                Color::Rgba(255, 0, 0, 255),
                Color::Rgba(0, 255, 0, 255),
                Color::Rgba(0, 0, 255, 255),
            ],
            1 => [
                Color::Rgba(0, 0, 0, 0),
                Color::Rgba(255, 255, 0, 255),
                Color::Rgba(255, 0, 255, 255),
                Color::Rgba(0, 255, 255, 255),
            ],
            _ => panic!["There should only be 2 quads"],
        });

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "set_color_all", img);
    }

    #[test]
    fn simple_quad_set_position_after_initial() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut quad = quads::Quad::new();
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        let mut quads = vx.quads();
        let layer = quads.add_layer(&LayerOptions::new());
        let handle = quads.add(&layer, quad);

        for _ in 0..3 {
            vx.draw_frame();
        }

        vx.quads().set_translation(&handle, (0.25, 0.4));

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "simple_quad_set_position_after_initial", img);
    }

    #[test]
    fn simple_quad_rotated_with_exotic_origin() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut quad = quads::Quad::new();
        quad.scale = 0.2;
        quad.colors[0].0 = 255;
        quad.colors[3].0 = 255;

        let layer = vx.quads().add_layer(&LayerOptions::new());
        vx.quads().add(&layer, quad);

        let mut quad = quads::Quad::new();
        quad.scale = 0.2;
        quad.origin = (-1.0, -1.0);
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        let mut quads = vx.quads();
        quads.add(&layer, quad);

        // when
        quads.rotate_all(&layer, |_| Deg(30.0));

        // then
        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "simple_quad_rotated_with_exotic_origin", img);
    }

    #[test]
    fn quad_layering() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let mut quad = quads::Quad {
            scale: 0.5,
            ..quads::Quad::new()
        };

        for i in 0..4 {
            quad.colors[i] = (0, 255, 0, 255);
        }
        quad.depth = 0.0;
        quad.translation = (0.25, 0.25);

        let layer1 = vx.quads().add_layer(&LayerOptions::new());
        let layer2 = vx.quads().add_layer(&LayerOptions::new());

        vx.quads().add(&layer2, quad);

        quad.scale = 0.6;
        for i in 0..4 {
            quad.colors[i] = (0, 0, 255, 255);
        }
        vx.quads().add(&layer1, quad);

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "quad_layering", img);
    }

    #[test]
    fn quad_mass_manip() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let layer = vx.quads().add_layer(&LayerOptions::new());

        use rand::Rng;
        use rand_pcg::Pcg64Mcg as random;
        let mut rng = random::new(0);

        let quad = quads::Quad::new();

        for _ in 0..1000 {
            vx.quads().add(&layer, quad);
        }

        for _ in 0..vx.buffer_count() {
            vx.draw_frame();
        }

        vx.quads().set_translation_all(&layer, |idx| {
            if idx < 500 {
                (
                    rng.gen_range(-1.0f32, 0.0f32),
                    rng.gen_range(-1.0f32, 1.0f32),
                )
            } else {
                (
                    rng.gen_range(0.0f32, 1.0f32),
                    rng.gen_range(-1.0f32, 1.0f32),
                )
            }
        });

        vx.quads()
            .set_scale_all(&layer, |idx| if idx < 500 { 0.01 } else { 0.02 });

        vx.quads().set_solid_color_all(&layer, |idx| {
            if idx < 250 {
                Color::Rgba(0, 255, 255, 128)
            } else if idx < 500 {
                Color::Rgba(0, 255, 0, 128)
            } else if idx < 750 {
                Color::Rgba(0, 0, 255, 128)
            } else {
                Color::Rgba(255, 255, 255, 128)
            }
        });

        vx.quads()
            .set_rotation_all(&layer, |idx| if idx < 500 { Deg(0.0) } else { Deg(30.0) });

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "quad_mass_manip", img);
    }

    #[test]
    fn rapidly_add_remove_layer() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let options = &LayerOptions::new();

        for _ in 0..10 {
            let mut quads = vx.quads();
            let layer = quads.add_layer(options);

            quads.add(&layer, Quad::new());

            vx.draw_frame();

            vx.quads().remove_layer(layer);
            assert![vx.swapconfig.image_count + 1 >= vx.quads().layer_count() as u32];
            assert![0 < vx.quads().layer_count()];
        }
    }
}
