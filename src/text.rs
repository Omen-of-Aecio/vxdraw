//! Methods and types to control text rendering
//!
//! # Example - Drawing centered text #
//! ```
//! use vxdraw::{prelude::*, text, void_logger, Deg, Matrix4, ShowWindow, VxDraw};
//! const DEJAVU: &[u8] = include_bytes!["../fonts/DejaVuSans.ttf"];
//! #[cfg(feature = "doctest-headless")]
//! let mut vx = VxDraw::new(void_logger(), ShowWindow::Headless1k);
//! #[cfg(not(feature = "doctest-headless"))]
//! let mut vx = VxDraw::new(void_logger(), ShowWindow::Enable);
//!
//! // Create a new layer. A layer consists of a font file and some options
//! let mut layer = vx.text().add_layer(DEJAVU, text::LayerOptions::new());
//!
//! // Create a new piece of text. We use the origin to center the text. The origin spans
//! // between 0 and 1, where 0 is the left-most side of the entire text section, and 1 the
//! // right-most. The same goes for the top and bottom respectively. Note that values outside
//! // of 0 to 1 are allowed, and define the origin of transformations on that text.
//! vx.text().add(
//!     &mut layer,
//!     "This text shall be\ncentered, as a whole,\nbut each line is not centered individually",
//!     text::TextOptions::new().font_size(40.0).origin((0.5, 0.5)),
//! );
//!
//! vx.draw_frame();
//!
//! #[cfg(not(feature = "doctest-headless"))]
//! std::thread::sleep(std::time::Duration::new(3, 0));
//! ```
//!
//! # Example - Textured text #
//!
//! Text itself does not directly support textures, but by using blending modes we can overlay a texture
//! onto the text.
//! ```
//! use vxdraw::{prelude::*, blender, dyntex::{Filter, ImgData, LayerOptions, Sprite}, quads, text, void_logger, Deg, Matrix4, ShowWindow, VxDraw};
//! static FOREST: &ImgData = &ImgData::PNGBytes(include_bytes!["../images/testure.png"]);
//! const DEJAVU: &[u8] = include_bytes!["../fonts/DejaVuSans.ttf"];
//!
//! #[cfg(feature = "doctest-headless")]
//! let mut vx = VxDraw::new(void_logger(), ShowWindow::Headless1k);
//! #[cfg(not(feature = "doctest-headless"))]
//! let mut vx = VxDraw::new(void_logger(), ShowWindow::Enable);
//!
//! let clear_alpha = vx.quads().add_layer(&quads::LayerOptions::new().blend(|x| {
//!     x.alpha(blender::BlendOp::Add {
//!         src: blender::BlendFactor::Zero,
//!         dst: blender::BlendFactor::Zero,
//!     })
//! })
//! .fixed_perspective(Matrix4::identity()));
//!
//! let text =  vx.text().add_layer(DEJAVU, text::LayerOptions::new().blend(|x| {
//!     x.alpha(blender::BlendOp::Add {
//!         src: blender::BlendFactor::One,
//!         dst: blender::BlendFactor::Zero,
//!     })
//! }));
//!
//! let texture = vx.dyntex().add_layer(
//!     FOREST,
//!     &LayerOptions::new().blend(|x| {
//!         x.colors(blender::BlendOp::Add {
//!             src: blender::BlendFactor::DstAlpha,
//!             dst: blender::BlendFactor::OneMinusDstAlpha,
//!         })
//!     })
//!     .filter(Filter::Linear),
//! );
//!
//! vx.quads().add(&clear_alpha, quads::Quad::new());
//! vx.text().add(&text, "This is\ntextured text", text::TextOptions::new()
//!     .origin((0.5, 0.5))
//!     .font_size(120.0));
//! vx.dyntex().add(&texture, Sprite::new().scale(1.0));
//!
//! vx.draw_frame();
//! #[cfg(not(feature = "doctest-headless"))]
//! std::thread::sleep(std::time::Duration::new(3, 0));
//! ```
////!     #[cfg(not(test))]
use super::utils::*;
use crate::{
    blender,
    data::{DrawType, SData, Text, VxDraw},
};
use cgmath::{Matrix4, Rad, Vector4};
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
    command::{CommandBuffer, CommandBufferFlags},
    device::Device,
    format, image, memory, pass,
    pso::{self, DescriptorPool, Primitive},
    queue::CommandQueue,
    Backend,
};
use glyph_brush::{BrushAction, BrushError, GlyphBrushBuilder};
use std::{io::Cursor, mem::ManuallyDrop};

// ---

const PIX_WIDTH_DIVISOR: f32 = 500f32;

// ---

/// Options for this text layer
pub struct LayerOptions {
    // Wrap mode does not make sense here...
    /// Specify filtering mode for sampling the grid texture (default is [Filter::Nearest])
    filtering: Filter,
    /// Blending mode for this layer
    blend: blender::Blender,
    vertex_shader: VertexShader,
    fragment_shader: FragmentShader,
    fixed_perspective: Option<Matrix4<f32>>,
}

impl Default for LayerOptions {
    fn default() -> Self {
        Self {
            filtering: Filter::Linear,
            blend: blender::Blender::default(),
            vertex_shader: VertexShader::Standard,
            fragment_shader: FragmentShader::Standard,
            fixed_perspective: None,
        }
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

impl LayerOptions {
    /// Create a new options structure
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

    /// Set the sampling filter mode for the texture
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filtering = filter;
        self
    }

    /// Set the blender of this layer (see [blender])
    pub fn blend(mut self, blend_setter: impl Fn(blender::Blender) -> blender::Blender) -> Self {
        self.blend = blend_setter(self.blend);
        self
    }

    /// Set a fixed perspective for this layer
    pub fn fixed_perspective(mut self, mat: Matrix4<f32>) -> Self {
        self.fixed_perspective = Some(mat);
        self
    }
}

/// Specify filter options
#[derive(Clone, Copy)]
pub enum Filter {
    /// Sample single a single texel and use its value
    Nearest,
    /// Compose the color of by sampling the surrounding pixels bilinearly
    Linear,
}

/// Options when adding a text
pub struct TextOptions {
    font_size_x: f32,
    font_size_y: f32,
    translation: (f32, f32),
    origin: (f32, f32),
    rotation: f32,
    scale: f32,
}

impl Default for TextOptions {
    fn default() -> Self {
        Self {
            font_size_x: 16.0,
            font_size_y: 16.0,
            translation: (0.0, 0.0),
            origin: (0.0, 0.0),
            rotation: 0.0,
            scale: 1.0,
        }
    }
}

impl TextOptions {
    /// Create a new text option
    pub fn new() -> Self {
        Self::default()
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

    /// Set the origin of the text
    ///
    /// Each glyph will have its model space translated by the negation of the origin multiplied by
    /// the width or height depending on the coordinate. This means an origin of (0.5, 0.5) will
    /// put the origin in the center of the entire text block. An origin of (1.0, 1.0) puts the
    /// origin in the bottom right of the text block. And (1.0, 0.0) puts the origin in the
    /// top-right corner of the text block. You can use any value, even (3.0, -5.0) if you want to,
    /// which just means that the text when rotated rotates around that relative point to the text
    /// block width and height.
    pub fn origin(self, origin: (f32, f32)) -> Self {
        Self { origin, ..self }
    }

    /// Set the rotation of the text
    pub fn rotation(self, rotation: f32) -> Self {
        Self { rotation, ..self }
    }

    /// Set the rotation of the text
    pub fn scale(self, scale: f32) -> Self {
        Self { scale, ..self }
    }
}

/// Handle to a piece of text
pub struct Handle {
    layer: usize,
    vertices: std::ops::Range<usize>,
    id: usize,
}

/// Handle to a layer (a single glyph store/font)
pub struct Layer(usize);

/// Accessor object to all text
pub struct Texts<'a> {
    vx: &'a mut VxDraw,
}

impl<'a> Texts<'a> {
    /// Prepare to edit text
    ///
    /// You're not supposed to use this function directly (although you can).
    /// The recommended way of spawning a text is via [VxDraw::text()].
    pub(crate) fn new(vx: &'a mut VxDraw) -> Self {
        Self { vx }
    }

    #[cfg(test)]
    fn get_texture_dimensions(&self, layer: &Layer) -> (u32, u32) {
        self.vx.texts[layer.0].glyph_brush.texture_dimensions()
    }

    /// Set the fixed perspective of a layer. `None` uses the vxdraw perspective.
    pub fn set_perspective(&mut self, layer: &Layer, perspective: Option<Matrix4<f32>>) {
        self.vx.texts[layer.0].fixed_perspective = perspective;
    }

    /// Query the amount of layers of this type there are
    pub fn layer_count(&self) -> usize {
        self.vx.texts.len()
    }

    /// Add a text layer to the system
    pub fn add_layer(&mut self, font: &'static [u8], options: LayerOptions) -> Layer {
        let mut glyph_brush = GlyphBrushBuilder::using_font_bytes(font);
        glyph_brush.cache_glyph_positioning = false;
        glyph_brush.cache_glyph_drawing = false;
        let glyph_brush = glyph_brush.build();
        let (width, height) = glyph_brush.texture_dimensions();

        // Create an image, image view, and memory
        self.vx
            .adapter
            .physical_device
            .image_format_properties(
                format::Format::Rgba8Srgb,
                2,
                image::Tiling::Linear,
                image::Usage::SAMPLED | image::Usage::TRANSFER_SRC,
                image::ViewCapabilities::empty(),
            )
            .expect("Device does not support linear sampled textures");
        let mut image = unsafe {
            self.vx
                .device
                .create_image(
                    image::Kind::D2(width as u32, height as u32, 1, 1),
                    1,
                    format::Format::Rgba8Srgb,
                    image::Tiling::Linear,
                    image::Usage::SAMPLED | image::Usage::TRANSFER_DST,
                    image::ViewCapabilities::empty(),
                )
                .expect("Couldn't create the image!")
        };

        let image_requirements = unsafe { self.vx.device.get_image_requirements(&image) };
        let image_memory = unsafe {
            let memory_type_id = find_memory_type_id(
                &self.vx.adapter,
                image_requirements,
                memory::Properties::DEVICE_LOCAL
                    | memory::Properties::CPU_VISIBLE
                    | memory::Properties::COHERENT,
            );
            self.vx
                .device
                .allocate_memory(memory_type_id, image_requirements.size)
                .expect("Unable to allocate")
        };

        let image_view = unsafe {
            self.vx
                .device
                .bind_image_memory(&image_memory, 0, &mut image)
                .expect("Unable to bind memory");

            self.vx
                .device
                .create_image_view(
                    &image,
                    image::ViewKind::D2,
                    format::Format::Rgba8Srgb,
                    format::Swizzle::NO,
                    image::SubresourceRange {
                        aspects: format::Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                )
                .expect("Couldn't create the image view!")
        };

        // Create a sampler
        let sampler = unsafe {
            self.vx
                .device
                .create_sampler(&image::SamplerDesc::new(
                    match options.filtering {
                        Filter::Nearest => image::Filter::Nearest,
                        Filter::Linear => image::Filter::Linear,
                    },
                    image::WrapMode::Tile,
                ))
                .expect("Couldn't create the sampler!")
        };

        /// Add shader
        const VERTEX_SOURCE_TEXTURE: &[u8] = include_bytes!["../target/spirv/text.vert.spirv"];
        const FRAGMENT_SOURCE_TEXTURE: &[u8] = include_bytes!["../target/spirv/text.frag.spirv"];

        let vertex_source_texture = match options.vertex_shader {
            VertexShader::Standard => pso::read_spirv(Cursor::new(VERTEX_SOURCE_TEXTURE)).unwrap(),
            VertexShader::Spirv(ref data) => pso::read_spirv(Cursor::new(data)).unwrap(),
        };
        let fragment_source_texture = match options.fragment_shader {
            FragmentShader::Standard => {
                pso::read_spirv(Cursor::new(FRAGMENT_SOURCE_TEXTURE)).unwrap()
            }
            FragmentShader::Spirv(ref data) => pso::read_spirv(Cursor::new(data)).unwrap(),
        };

        let vs_module =
            { unsafe { self.vx.device.create_shader_module(&vertex_source_texture) }.unwrap() };
        let fs_module = {
            unsafe {
                self.vx
                    .device
                    .create_shader_module(&fragment_source_texture)
            }
            .unwrap()
        };

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

        // Describe input data
        let vertex_buffers: Vec<pso::VertexBufferDesc> = vec![
            pso::VertexBufferDesc {
                binding: 0,
                stride: 8,
                rate: pso::VertexInputRate::Vertex,
            },
            pso::VertexBufferDesc {
                binding: 1,
                stride: 8,
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
            pso::VertexBufferDesc {
                binding: 5,
                stride: 1,
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
                    format: format::Format::Rg32Sfloat,
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
            pso::AttributeDesc {
                location: 5,
                binding: 5,
                element: pso::Element {
                    format: format::Format::R8Unorm,
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
            depth: None,
            depth_bounds: false,
            stencil: None,
        };

        let blender = options.blend.clone().into_gfx_blender();

        let render_pass = {
            let attachment = pass::Attachment {
                format: Some(self.vx.format),
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
                self.vx
                    .device
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

        let mut bindings = Vec::<pso::DescriptorSetLayoutBinding>::new();
        bindings.push(pso::DescriptorSetLayoutBinding {
            binding: 0,
            ty: pso::DescriptorType::SampledImage,
            count: 1,
            stage_flags: pso::ShaderStageFlags::FRAGMENT,
            immutable_samplers: false,
        });
        bindings.push(pso::DescriptorSetLayoutBinding {
            binding: 1,
            ty: pso::DescriptorType::Sampler,
            count: 1,
            stage_flags: pso::ShaderStageFlags::FRAGMENT,
            immutable_samplers: false,
        });
        let immutable_samplers = Vec::<<back::Backend as Backend>::Sampler>::new();
        let descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> =
            vec![unsafe {
                self.vx
                    .device
                    .create_descriptor_set_layout(bindings, immutable_samplers)
                    .expect("Couldn't make a DescriptorSetLayout")
            }];

        let mut descriptor_pool = unsafe {
            self.vx
                .device
                .create_descriptor_pool(
                    1, // sets
                    &[
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::SampledImage,
                            count: 1,
                        },
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::Sampler,
                            count: 1,
                        },
                    ],
                    pso::DescriptorPoolCreateFlags::empty(),
                )
                .expect("Couldn't create a descriptor pool!")
        };

        let descriptor_set = unsafe {
            descriptor_pool
                .allocate_set(&descriptor_set_layouts[0])
                .expect("Couldn't make a Descriptor Set!")
        };

        // Write descriptor sets
        unsafe {
            self.vx.device.write_descriptor_sets(vec![
                pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Image(&image_view, image::Layout::General)),
                },
                pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 1,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Sampler(&sampler)),
                },
            ]);
        }

        // Push constants
        let mut push_constants = Vec::<(pso::ShaderStageFlags, core::ops::Range<u32>)>::new();
        push_constants.push((pso::ShaderStageFlags::VERTEX, 0..64));

        let pipeline_layout = unsafe {
            self.vx
                .device
                .create_pipeline_layout(&descriptor_set_layouts, push_constants)
                .expect("Couldn't create a pipeline layout")
        };

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
            layout: &pipeline_layout,
            subpass: pass::Subpass {
                index: 0,
                main_pass: &render_pass,
            },
            flags: pso::PipelineCreationFlags::empty(),
            parent: pso::BasePipeline::None,
        };

        let pipeline = unsafe {
            self.vx
                .device
                .create_graphics_pipeline(&pipeline_desc, None)
                .expect("Couldn't create a graphics pipeline!")
        };

        // Clean up
        unsafe {
            self.vx.device.destroy_shader_module(vs_module);
            self.vx.device.destroy_shader_module(fs_module);
        }

        // Create vertex buffers
        let image_count = self.vx.swapconfig.image_count;
        let posbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&self.vx.device, &self.vx.adapter))
            .collect::<Vec<_>>();
        let opacbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&self.vx.device, &self.vx.adapter))
            .collect::<Vec<_>>();
        let uvbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&self.vx.device, &self.vx.adapter))
            .collect::<Vec<_>>();
        let tranbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&self.vx.device, &self.vx.adapter))
            .collect::<Vec<_>>();
        let rotbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&self.vx.device, &self.vx.adapter))
            .collect::<Vec<_>>();
        let scalebuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&self.vx.device, &self.vx.adapter))
            .collect::<Vec<_>>();

        let indices = (0..image_count)
            .map(|_| super::utils::ResizBufIdx4::new(&self.vx.device, &self.vx.adapter))
            .collect::<Vec<_>>();

        // Change the image to layout general
        unsafe {
            let barrier_fence = self
                .vx
                .device
                .create_fence(false)
                .expect("unable to make fence");
            // TODO Use a proper command buffer here
            self.vx.device.wait_idle().unwrap();
            let buffer = &mut self.vx.command_buffers[self.vx.current_frame];
            buffer.begin_primary(CommandBufferFlags::EMPTY);
            let image_barrier = memory::Barrier::Image {
                states: (image::Access::empty(), image::Layout::Undefined)
                    ..(
                        // image::Access::HOST_READ | image::Access::HOST_WRITE,
                        image::Access::empty(),
                        image::Layout::General,
                    ),
                target: &image,
                families: None,
                range: image::SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };
            buffer.pipeline_barrier(
                pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::HOST,
                memory::Dependencies::empty(),
                &[image_barrier],
            );
            buffer.finish();
            self.vx.queue_group.queues[0]
                .submit_without_semaphores(Some(&*buffer), Some(&barrier_fence));
            self.vx
                .device
                .wait_for_fence(&barrier_fence, u64::max_value())
                .unwrap();
            self.vx.device.destroy_fence(barrier_fence);
        }

        let text = Text {
            hidden: false,
            removed: vec![],
            glyph_brush,

            texts: vec![],
            font_sizes: vec![],
            origin: vec![],

            width: vec![],
            height: vec![],

            fixed_perspective: options.fixed_perspective,

            posbuf_touch: 0,
            opacbuf_touch: 0,
            uvbuf_touch: 0,
            tranbuf_touch: 0,
            rotbuf_touch: 0,
            scalebuf_touch: 0,

            posbuffer: vec![],
            opacbuffer: vec![],
            uvbuffer: vec![],
            tranbuffer: vec![],
            rotbuffer: vec![],
            scalebuffer: vec![],

            posbuf,
            opacbuf,
            uvbuf,
            tranbuf,
            rotbuf,
            scalebuf,

            indices,

            image_buffer: ManuallyDrop::new(image),
            image_memory: ManuallyDrop::new(image_memory),
            image_view: ManuallyDrop::new(image_view),
            image_requirements,

            sampler: ManuallyDrop::new(sampler),
            descriptor_pool: ManuallyDrop::new(descriptor_pool),
            descriptor_set_layouts,
            descriptor_set: ManuallyDrop::new(descriptor_set),
            pipeline: ManuallyDrop::new(pipeline),
            pipeline_layout: ManuallyDrop::new(pipeline_layout),
            render_pass: ManuallyDrop::new(render_pass),
        };
        let prev_layer = self.vx.layer_holes.find_available(|x| match x {
            DrawType::Text { .. } => true,
            _ => false,
        });
        if let Some(prev_layer) = prev_layer {
            match prev_layer {
                DrawType::Text { id } => {
                    let old_text = std::mem::replace(&mut self.vx.texts[id], text);
                    old_text.destroy(&self.vx.device);
                    self.vx.draw_order.push(DrawType::Text { id });
                    Layer(id)
                }
                _ => panic!["Got a non-text drawtype, should be impossible!"],
            }
        } else {
            self.vx.texts.push(text);
            self.vx.draw_order.push(DrawType::Text {
                id: self.vx.texts.len() - 1,
            });
            Layer(self.vx.texts.len() - 1)
        }
    }

    /// Remove a layer
    pub fn remove_layer(&mut self, layer: Layer) {
        let s = &mut *self.vx;
        let mut index = None;
        for (idx, x) in s.draw_order.iter().enumerate() {
            match x {
                DrawType::Text { id } if *id == layer.0 => {
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

    /// Add text to this layer
    pub fn add(&mut self, layer: &Layer, string: &str, opts: TextOptions) -> Handle {
        let section = glyph_brush::Section {
            text: string,
            scale: glyph_brush::rusttype::Scale {
                x: opts.font_size_x,
                y: opts.font_size_y,
            },
            ..glyph_brush::Section::default()
        };
        self.vx.texts[layer.0].glyph_brush.queue(section);
        self.vx.texts[layer.0].posbuf_touch = self.vx.swapconfig.image_count;
        self.vx.texts[layer.0].opacbuf_touch = self.vx.swapconfig.image_count;
        self.vx.texts[layer.0].uvbuf_touch = self.vx.swapconfig.image_count;
        self.vx.texts[layer.0].tranbuf_touch = self.vx.swapconfig.image_count;
        self.vx.texts[layer.0].rotbuf_touch = self.vx.swapconfig.image_count;
        self.vx.texts[layer.0].scalebuf_touch = self.vx.swapconfig.image_count;
        self.vx.wait_for_fences();
        let prev_begin = self.vx.texts[layer.0].posbuffer.len();
        let mut count = 0usize;
        let mut tex_values = vec![];

        let mut top = 0;
        let mut bottom = 0;
        let mut left = 0;
        let mut right = 0;

        let mut resized = false;

        match self.vx.texts[layer.0].glyph_brush.process_queued(
            |rect, tex_data| {
                tex_values.push((rect, tex_data.to_owned()));
            },
            |vtx| SData {
                uv_begin: (vtx.tex_coords.min.x, vtx.tex_coords.min.y),
                uv_end: (vtx.tex_coords.max.x, vtx.tex_coords.max.y),
                topleft: (vtx.pixel_coords.min.x, vtx.pixel_coords.min.y),
                bottomright: (vtx.pixel_coords.max.x, vtx.pixel_coords.max.y),
            },
        ) {
            Ok(BrushAction::Draw(vertices)) => {
                assert_eq![0, count];
                count = vertices.len();
                self.vx.texts[layer.0].texts.push(string.to_string());
                self.vx.texts[layer.0]
                    .font_sizes
                    .push((opts.font_size_x, opts.font_size_y));
                self.vx.texts[layer.0]
                    .origin
                    .push((opts.origin.0, opts.origin.1));
                for vtx in vertices.iter() {
                    top = top.min(vtx.topleft.0);
                    left = left.min(vtx.topleft.1);
                    bottom = bottom.max(vtx.bottomright.1);
                    right = right.max(vtx.bottomright.0);
                }
                for vtx in vertices.iter() {
                    let muscale = PIX_WIDTH_DIVISOR;
                    let uv_b = vtx.uv_begin;
                    let uv_e = vtx.uv_end;
                    let beg = vtx.topleft;
                    let end = vtx.bottomright;
                    let begf = (beg.0 as f32 / muscale, beg.1 as f32 / muscale);
                    let width = (end.0 - beg.0) as f32 / muscale;
                    let height = (end.1 - beg.1) as f32 / muscale;
                    let uv_a = uv_b;
                    let uv_b = uv_e;

                    let topleft = (begf.0, begf.1);
                    let topleft_uv = uv_a;

                    let topright = (begf.0 + width, begf.1);
                    let topright_uv = (uv_b.0, uv_a.1);

                    let bottomleft = (begf.0, begf.1 + height);
                    let bottomleft_uv = (uv_a.0, uv_b.1);

                    let bottomright = (begf.0 + width, begf.1 + height);
                    let bottomright_uv = (uv_b.0, uv_b.1);
                    let tex = &mut self.vx.texts[layer.0];
                    tex.posbuffer.push([
                        topleft.0,
                        topleft.1,
                        bottomleft.0,
                        bottomleft.1,
                        bottomright.0,
                        bottomright.1,
                        topright.0,
                        topright.1,
                    ]);
                    tex.opacbuffer.push([255, 255, 255, 255]);
                    tex.tranbuffer.push([
                        opts.translation.0,
                        opts.translation.1,
                        opts.translation.0,
                        opts.translation.1,
                        opts.translation.0,
                        opts.translation.1,
                        opts.translation.0,
                        opts.translation.1,
                    ]);
                    tex.rotbuffer.push([opts.rotation; 4]);
                    tex.scalebuffer.push([opts.scale; 4]);
                    tex.uvbuffer.push([
                        topleft_uv.0,
                        topleft_uv.1,
                        bottomleft_uv.0,
                        bottomleft_uv.1,
                        bottomright_uv.0,
                        bottomright_uv.1,
                        topright_uv.0,
                        topright_uv.1,
                    ]);
                }
            }
            Ok(BrushAction::ReDraw) => {}
            Err(BrushError::TextureTooSmall { suggested }) => {
                self.resize_internal_texture(&layer, suggested);
                resized = true;
            }
        }

        if resized {
            self.recompute_text(layer);
            return self.add(layer, string, opts);
        }

        let width = right - left;
        let height = bottom - top;
        self.vx.texts[layer.0].width.push(width);
        self.vx.texts[layer.0].height.push(height);
        for idx in prev_begin..prev_begin + count {
            let pos = &mut self.vx.texts[layer.0].posbuffer[idx];
            pos[0] -= opts.origin.0 * width as f32 / PIX_WIDTH_DIVISOR;
            pos[1] -= opts.origin.1 * height as f32 / PIX_WIDTH_DIVISOR;
            pos[2] -= opts.origin.0 * width as f32 / PIX_WIDTH_DIVISOR;
            pos[3] -= opts.origin.1 * height as f32 / PIX_WIDTH_DIVISOR;
            pos[4] -= opts.origin.0 * width as f32 / PIX_WIDTH_DIVISOR;
            pos[5] -= opts.origin.1 * height as f32 / PIX_WIDTH_DIVISOR;
            pos[6] -= opts.origin.0 * width as f32 / PIX_WIDTH_DIVISOR;
            pos[7] -= opts.origin.1 * height as f32 / PIX_WIDTH_DIVISOR;
        }

        for (rect, tex_data) in tex_values {
            unsafe {
                let foot = self.vx.device.get_image_subresource_footprint(
                    &self.vx.texts[layer.0].image_buffer,
                    image::Subresource {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layer: 0,
                    },
                );
                let target = self
                    .vx
                    .device
                    .map_memory(
                        &self.vx.texts[layer.0].image_memory,
                        0..self.vx.texts[layer.0].image_requirements.size,
                    )
                    .expect("unable to acquire mapping writer");

                let width = rect.max.x - rect.min.x;
                for (idx, alpha) in tex_data.iter().enumerate() {
                    let idx = idx as u32;
                    let x = rect.min.x + idx % width;
                    let y = rect.min.y + idx / width;
                    let access = foot.row_pitch * u64::from(y) + u64::from(x * 4);
                    std::slice::from_raw_parts_mut(
                        target,
                        self.vx.texts[layer.0].image_requirements.size as usize,
                    )[access as usize..(access + 4) as usize]
                        .copy_from_slice(&[255, 255, 255, *alpha]);
                }
                self.vx
                    .device
                    .unmap_memory(&self.vx.texts[layer.0].image_memory);
            }
        }
        Handle {
            layer: layer.0,
            vertices: prev_begin..prev_begin + count,
            id: self.vx.texts[layer.0].width.len() - 1,
        }
    }

    fn recompute_text(&mut self, layer: &Layer) {
        let this_layer = &mut self.vx.texts[layer.0];
        let mut count = 0;
        for (idx, text) in this_layer.texts.iter().enumerate() {
            let font_size = this_layer.font_sizes[idx];
            let width = this_layer.width[idx];
            let height = this_layer.height[idx];
            let origin = this_layer.origin[idx];
            let section = glyph_brush::Section {
                text: &text,
                scale: glyph_brush::rusttype::Scale {
                    x: font_size.0,
                    y: font_size.1,
                },
                ..glyph_brush::Section::default()
            };
            let mut tex_values = vec![];

            let _just_clear_the_cache = this_layer.glyph_brush.process_queued(
                |rect, tex_data| {
                    tex_values.push((rect, tex_data.to_owned()));
                },
                |_| SData::default(),
            );

            this_layer.glyph_brush.queue(section);
            let mut resized = None;
            match this_layer.glyph_brush.process_queued(
                |rect, tex_data| {
                    tex_values.push((rect, tex_data.to_owned()));
                },
                |vtx| SData {
                    uv_begin: (vtx.tex_coords.min.x, vtx.tex_coords.min.y),
                    uv_end: (vtx.tex_coords.max.x, vtx.tex_coords.max.y),
                    topleft: (vtx.pixel_coords.min.x, vtx.pixel_coords.min.y),
                    bottomright: (vtx.pixel_coords.max.x, vtx.pixel_coords.max.y),
                },
            ) {
                Ok(BrushAction::Draw(vertices)) => {
                    for vtx in vertices.iter() {
                        let muscale = PIX_WIDTH_DIVISOR;
                        let uv_b = vtx.uv_begin;
                        let uv_e = vtx.uv_end;
                        let beg = vtx.topleft;
                        let end = vtx.bottomright;
                        let begf = (beg.0 as f32 / muscale, beg.1 as f32 / muscale);
                        let width2 = (end.0 - beg.0) as f32 / muscale;
                        let height2 = (end.1 - beg.1) as f32 / muscale;
                        let uv_a = uv_b;
                        let uv_b = uv_e;

                        let topleft = (begf.0, begf.1);
                        let topleft_uv = uv_a;

                        let topright = (begf.0 + width2, begf.1);
                        let topright_uv = (uv_b.0, uv_a.1);

                        let bottomleft = (begf.0, begf.1 + height2);
                        let bottomleft_uv = (uv_a.0, uv_b.1);

                        let bottomright = (begf.0 + width2, begf.1 + height2);
                        let bottomright_uv = (uv_b.0, uv_b.1);

                        this_layer.posbuffer[count].copy_from_slice(&[
                            topleft.0 - origin.0 * width as f32 / PIX_WIDTH_DIVISOR,
                            topleft.1 - origin.1 * height as f32 / PIX_WIDTH_DIVISOR,
                            bottomleft.0 - origin.0 * width as f32 / PIX_WIDTH_DIVISOR,
                            bottomleft.1 - origin.1 * height as f32 / PIX_WIDTH_DIVISOR,
                            bottomright.0 - origin.0 * width as f32 / PIX_WIDTH_DIVISOR,
                            bottomright.1 - origin.1 * height as f32 / PIX_WIDTH_DIVISOR,
                            topright.0 - origin.0 * width as f32 / PIX_WIDTH_DIVISOR,
                            topright.1 - origin.1 * height as f32 / PIX_WIDTH_DIVISOR,
                        ]);
                        this_layer.uvbuffer[count].copy_from_slice(&[
                            topleft_uv.0,
                            topleft_uv.1,
                            bottomleft_uv.0,
                            bottomleft_uv.1,
                            bottomright_uv.0,
                            bottomright_uv.1,
                            topright_uv.0,
                            topright_uv.1,
                        ]);
                        count += 1;
                    }
                }
                Ok(BrushAction::ReDraw) => {}
                Err(BrushError::TextureTooSmall { suggested }) => {
                    resized = Some(suggested);
                }
            }

            for (rect, tex_data) in tex_values {
                unsafe {
                    let foot = self.vx.device.get_image_subresource_footprint(
                        &this_layer.image_buffer,
                        image::Subresource {
                            aspects: format::Aspects::COLOR,
                            level: 0,
                            layer: 0,
                        },
                    );
                    let target = self
                        .vx
                        .device
                        .map_memory(
                            &this_layer.image_memory,
                            0..this_layer.image_requirements.size,
                        )
                        .expect("unable to acquire mapping writer");

                    let width = rect.max.x - rect.min.x;
                    for (idx, alpha) in tex_data.iter().enumerate() {
                        let idx = idx as u32;
                        let x = rect.min.x + idx % width;
                        let y = rect.min.y + idx / width;
                        let access = foot.row_pitch * u64::from(y) + u64::from(x * 4);
                        std::slice::from_raw_parts_mut(
                            target,
                            this_layer.image_requirements.size as usize,
                        )[access as usize..(access + 4) as usize]
                            .copy_from_slice(&[255, 255, 255, *alpha]);
                    }
                    self.vx.device.unmap_memory(&this_layer.image_memory);
                }
            }

            if let Some(suggested) = resized {
                self.resize_internal_texture(layer, suggested);
                self.recompute_text(layer);
                return;
            }
        }
    }

    fn resize_internal_texture(&mut self, layer: &Layer, suggested: (u32, u32)) {
        // Assume the existing image is not in use. Wait for fences before using this
        // function!
        // Create an image, image view, and memory
        self.vx
            .adapter
            .physical_device
            .image_format_properties(
                format::Format::Rgba8Srgb,
                2,
                image::Tiling::Linear,
                image::Usage::SAMPLED | image::Usage::TRANSFER_SRC,
                image::ViewCapabilities::empty(),
            )
            .expect("Device does not support linear sampled textures");
        let mut image = unsafe {
            self.vx
                .device
                .create_image(
                    image::Kind::D2(suggested.0 as u32, suggested.1 as u32, 1, 1),
                    1,
                    format::Format::Rgba8Srgb,
                    image::Tiling::Linear,
                    image::Usage::SAMPLED | image::Usage::TRANSFER_DST,
                    image::ViewCapabilities::empty(),
                )
                .expect("Couldn't create the image!")
        };

        let image_requirements = unsafe { self.vx.device.get_image_requirements(&image) };
        let image_memory = unsafe {
            let memory_type_id = find_memory_type_id(
                &self.vx.adapter,
                image_requirements,
                memory::Properties::DEVICE_LOCAL
                    | memory::Properties::CPU_VISIBLE
                    | memory::Properties::COHERENT,
            );
            self.vx
                .device
                .allocate_memory(memory_type_id, image_requirements.size)
                .expect("Unable to allocate")
        };

        let image_view = unsafe {
            self.vx
                .device
                .bind_image_memory(&image_memory, 0, &mut image)
                .expect("Unable to bind memory");

            self.vx
                .device
                .create_image_view(
                    &image,
                    image::ViewKind::D2,
                    format::Format::Rgba8Srgb,
                    format::Swizzle::NO,
                    image::SubresourceRange {
                        aspects: format::Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                )
                .expect("Couldn't create the image view!")
        };
        unsafe {
            self.vx
                .device
                .write_descriptor_sets(vec![pso::DescriptorSetWrite {
                    set: &*self.vx.texts[layer.0].descriptor_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Image(&image_view, image::Layout::General)),
                }]);
        }
        unsafe {
            self.vx.device.destroy_image(ManuallyDrop::into_inner(read(
                &self.vx.texts[layer.0].image_buffer,
            )));
            self.vx.device.free_memory(ManuallyDrop::into_inner(read(
                &self.vx.texts[layer.0].image_memory,
            )));
            self.vx
                .device
                .destroy_image_view(ManuallyDrop::into_inner(read(
                    &self.vx.texts[layer.0].image_view,
                )));
        }
        // Change the image to layout general
        unsafe {
            let barrier_fence = self
                .vx
                .device
                .create_fence(false)
                .expect("unable to make fence");
            // TODO Use a proper command buffer here
            self.vx.device.wait_idle().unwrap();
            let buffer = &mut self.vx.command_buffers[self.vx.current_frame];
            buffer.begin_primary(CommandBufferFlags::EMPTY);
            let image_barrier = memory::Barrier::Image {
                states: (image::Access::empty(), image::Layout::Undefined)
                    ..(
                        // image::Access::HOST_READ | image::Access::HOST_WRITE,
                        image::Access::empty(),
                        image::Layout::General,
                    ),
                target: &image,
                families: None,
                range: image::SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };
            buffer.pipeline_barrier(
                pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::HOST,
                memory::Dependencies::empty(),
                &[image_barrier],
            );
            buffer.finish();
            self.vx.queue_group.queues[0]
                .submit_without_semaphores(Some(&*buffer), Some(&barrier_fence));
            self.vx
                .device
                .wait_for_fence(&barrier_fence, u64::max_value())
                .unwrap();
            self.vx.device.destroy_fence(barrier_fence);
        }
        self.vx.texts[layer.0].image_buffer = ManuallyDrop::new(image);
        self.vx.texts[layer.0].image_view = ManuallyDrop::new(image_view);
        self.vx.texts[layer.0].image_memory = ManuallyDrop::new(image_memory);
        self.vx.texts[layer.0].image_requirements = image_requirements;
        self.vx.texts[layer.0]
            .glyph_brush
            .resize_texture(suggested.0, suggested.1);
    }

    /// Get the width of the text in native -1..1 coordinates
    ///
    /// The width is in screen coordinates without considering scaling or translation effects. The
    /// width is just the modelspace width.
    pub fn get_width(&self, handle: &Handle) -> f32 {
        self.vx.texts[handle.layer].width[handle.id] as f32 / PIX_WIDTH_DIVISOR
    }

    /// Get the height of the text in native -1..1 coordinates
    ///
    /// The height is in screen coordinates without considering scaling or translation effects. The
    /// height is just the modelspace height.
    pub fn get_height(&self, handle: &Handle) -> f32 {
        self.vx.texts[handle.layer].height[handle.id] as f32 / PIX_WIDTH_DIVISOR
    }

    /// Get the size of the model in object coordinates.
    ///
    /// Returns the bounding box of the text that takes into account scale and rotation
    /// The perspective used during rendering does not factor into the model size.
    pub fn get_model_size(&self, handle: &Handle) -> (f32, f32) {
        let (w, h) = (self.get_width(handle), self.get_height(handle));
        let scale = self.vx.texts[handle.layer].scalebuffer[handle.vertices.start][0];
        let rotation = self.vx.texts[handle.layer].rotbuffer[handle.vertices.start][0];

        let size = Vector4::new(w, h, 0.0, 0.0);
        let angle = Matrix4::from_angle_z(Rad(rotation));

        let model_space = angle * scale * size;
        (model_space.x.abs(), model_space.y.abs())
    }

    /// Get the size of the model in pixels
    pub fn get_model_size_in_pixels(&self, handle: &Handle) -> (f32, f32) {
        let (w, h) = self.get_model_size(handle);
        let (ww, wh) = self.vx.get_window_size_in_pixels_float();
        (w * ww / 2.0, h * wh / 2.0)
    }

    /// Get the width and height after the shader would've performed its transformations to
    /// the text.
    ///
    /// The result is given in -1..1 coordinates that are mapped fully to the window.
    /// Note that this depends on the perspective set on either [VxDraw] or the specific text
    /// layer.
    pub fn get_world_size(&self, handle: &Handle) -> (f32, f32) {
        let (w, h) = (self.get_width(handle), self.get_height(handle));
        let scale = self.vx.texts[handle.layer].scalebuffer[handle.vertices.start][0];
        let rotation = self.vx.texts[handle.layer].rotbuffer[handle.vertices.start][0];

        let size = Vector4::new(w, h, 0.0, 0.0);
        let angle = Matrix4::from_angle_z(Rad(rotation));

        let world = if let Some(prspect) = self.vx.texts[handle.layer].fixed_perspective {
            prspect * angle * scale * size
        } else {
            self.vx.perspective * angle * scale * size
        };
        (world.x.abs(), world.y.abs())
    }

    /// Get the width and height after the shader would've performed its transformations to
    /// the text.
    ///
    /// The result is given in pixel values
    pub fn get_world_size_in_pixels(&self, handle: &Handle) -> (f32, f32) {
        let (w, h) = self.get_world_size(handle);
        let (ww, wh) = self.vx.get_window_size_in_pixels_float();
        (w * ww / 2.0, h * wh / 2.0)
    }

    /// Get the amount of glyphs for this handle
    pub fn get_glyph_count(&self, handle: &Handle) -> usize {
        handle.vertices.end - handle.vertices.start
    }

    // ---

    /// Set the scale of the text segment
    pub fn set_translation(&mut self, handle: &Handle, translation: (f32, f32)) {
        self.vx.texts[handle.layer].tranbuf_touch = self.vx.swapconfig.image_count;
        for idx in handle.vertices.start..handle.vertices.end {
            self.vx.texts[handle.layer].tranbuffer[idx].copy_from_slice(&[
                translation.0,
                translation.1,
                translation.0,
                translation.1,
                translation.0,
                translation.1,
                translation.0,
                translation.1,
            ]);
        }
    }

    /// Set the scale of the text segment
    pub fn set_scale(&mut self, handle: &Handle, scale: f32) {
        self.vx.texts[handle.layer].scalebuf_touch = self.vx.swapconfig.image_count;
        for idx in handle.vertices.start..handle.vertices.end {
            self.vx.texts[handle.layer].scalebuffer[idx].copy_from_slice(&[scale; 4]);
        }
    }

    /// Set the opacity of a text segment
    pub fn set_opacity(&mut self, handle: &Handle, opacity: u8) {
        self.vx.texts[handle.layer].opacbuf_touch = self.vx.swapconfig.image_count;
        for idx in handle.vertices.start..handle.vertices.end {
            self.vx.texts[handle.layer].opacbuffer[idx].copy_from_slice(&[opacity; 4]);
        }
    }

    /// Set the rotation of the text segment as a whole
    pub fn set_rotation<T: Copy + Into<Rad<f32>>>(&mut self, handle: &Handle, angle: T) {
        self.vx.texts[handle.layer].rotbuf_touch = self.vx.swapconfig.image_count;
        for idx in handle.vertices.start..handle.vertices.end {
            self.vx.texts[handle.layer].rotbuffer[idx].copy_from_slice(&[angle.into().0; 4]);
        }
    }

    // ---

    /// Set the opacity on a per-glyph basis. Glyphs are enumerated as they would in a string
    pub fn set_opacity_glyphs(&mut self, handle: &Handle, mut delta: impl FnMut(usize) -> u8) {
        self.vx.texts[handle.layer].opacbuf_touch = self.vx.swapconfig.image_count;
        for idx in handle.vertices.start..handle.vertices.end {
            let delta = delta(idx - handle.vertices.start);
            self.vx.texts[handle.layer].opacbuffer[idx]
                .copy_from_slice(&[delta, delta, delta, delta]);
        }
    }

    /// Set the rotation of the text segment as a whole
    pub fn set_translation_glyphs(
        &mut self,
        handle: &Handle,
        mut delta: impl FnMut(usize) -> (f32, f32),
    ) {
        self.vx.texts[handle.layer].tranbuf_touch = self.vx.swapconfig.image_count;
        for idx in handle.vertices.start..handle.vertices.end {
            let delta = delta(idx - handle.vertices.start);
            self.vx.texts[handle.layer].tranbuffer[idx].copy_from_slice(&[
                delta.0, delta.1, delta.0, delta.1, delta.0, delta.1, delta.0, delta.1,
            ]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use fast_logger::{Generic, GenericLogger, Logger};
    use test::Bencher;

    const DEJAVU: &[u8] = include_bytes!["../fonts/DejaVuSans.ttf"];

    #[test]
    fn texting() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut layer = vx.text().add_layer(DEJAVU, text::LayerOptions::new());

        vx.text()
            .add(&mut layer, "font", text::TextOptions::new().font_size(60.0));

        vx.text().add(
            &mut layer,
            "my text",
            text::TextOptions::new()
                .font_size(32.0)
                .translation((0.0, -0.5)),
        );

        let img = vx.draw_frame_copy_framebuffer();
        assert_swapchain_eq(&mut vx, "some_text", img);
    }

    #[test]
    fn text_world_size() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut layer = vx.text().add_layer(DEJAVU, text::LayerOptions::new());

        let handle = vx.text().add(
            &mut layer,
            "text that spans to end of the screen",
            text::TextOptions::new().font_size(32.0),
        );

        assert_eq![
            (454.116, 29.536001),
            vx.text().get_world_size_in_pixels(&handle)
        ];

        vx.text().set_rotation(&handle, Deg(90.0));

        assert_eq![
            (29.536022, 454.116),
            vx.text().get_world_size_in_pixels(&handle)
        ];

        vx.set_perspective(Matrix4::from_angle_z(Deg(-90.0)));

        assert_eq![
            (454.116, 29.536001),
            vx.text().get_world_size_in_pixels(&handle)
        ];

        vx.text().set_scale(&handle, 2.0);

        assert_eq![
            (908.232, 59.072002),
            vx.text().get_world_size_in_pixels(&handle)
        ];
        assert_eq![(1.968, 0.128), vx.text().get_world_size(&handle)];

        vx.set_perspective(Matrix4::from_angle_z(Deg(45.0)));

        assert_eq![
            (683.9873, 600.4468),
            vx.text().get_world_size_in_pixels(&handle)
        ];

        assert_eq![(0.1280001, 1.968), vx.text().get_model_size(&handle)];

        vx.text().set_scale(&handle, 0.5);

        assert_eq![(0.032000024, 0.492), vx.text().get_model_size(&handle)];

        vx.set_perspective(Matrix4::identity());

        assert_eq![(0.032000024, 0.492), vx.text().get_model_size(&handle)];
    }

    #[test]
    fn centered_text_rotates_around_origin() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut layer = vx.text().add_layer(DEJAVU, text::LayerOptions::new());

        vx.text().add(
            &mut layer,
            "This text shall be\ncentered, as a whole,\nbut each line is not centered individually",
            text::TextOptions::new()
                .font_size(40.0)
                .origin((0.5, 0.5))
                .rotation(0.3),
        );

        let img = vx.draw_frame_copy_framebuffer();
        assert_swapchain_eq(&mut vx, "centered_text_rotates_around_origin", img);
    }

    #[test]
    fn fixed_perspective_text() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut layer = vx.text().add_layer(
            DEJAVU,
            text::LayerOptions::new().fixed_perspective(Matrix4::from_angle_z(Deg(-17.188))),
        );

        vx.text().add(
            &mut layer,
            "This text shall be\nrotated, as a whole,\nbecause of a fixed perspective",
            text::TextOptions::new().font_size(40.0).origin((0.5, 0.5)),
        );

        let img = vx.draw_frame_copy_framebuffer();
        assert_swapchain_eq(&mut vx, "fixed_perspective_text", img);
    }

    #[test]
    fn centered_text_translated_up() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut layer = vx.text().add_layer(DEJAVU, text::LayerOptions::new());

        let handle = vx.text().add(
            &mut layer,
            "This text shall be\ncentered, as a whole,\nbut each line is not centered individually",
            text::TextOptions::new().font_size(40.0).origin((0.5, 0.5)),
        );

        vx.text().set_translation(&handle, (0.0, -0.5));

        let img = vx.draw_frame_copy_framebuffer();
        assert_swapchain_eq(&mut vx, "centered_text_translated_up", img);
    }

    #[test]
    fn one_opaque_and_another_transparent() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut layer = vx.text().add_layer(DEJAVU, text::LayerOptions::new());

        vx.text().add(
            &mut layer,
            "This text shall be\ncentered, as a whole,\nbut each line is not centered individually",
            text::TextOptions::new().font_size(40.0).origin((0.5, 1.0)),
        );

        let transparent = vx.text().add(
            &mut layer,
            "This text shall be\ncentered, as a whole,\nbut each line is not centered individually",
            text::TextOptions::new().font_size(40.0).origin((0.5, 0.0)),
        );

        vx.text().set_opacity(&transparent, 128);

        let img = vx.draw_frame_copy_framebuffer();
        assert_swapchain_eq(&mut vx, "one_opaque_and_another_transparent", img);
    }

    #[test]
    fn resizing_back_texture() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut layer = vx.text().add_layer(DEJAVU, text::LayerOptions::new());

        vx.text().add(
            &mut layer,
            "This is some angled text",
            text::TextOptions::new()
                .font_size(40.0)
                .origin((0.5, 0.5))
                .translation((0.0, -0.5))
                .rotation(0.3),
        );

        // vx.draw_frame();

        vx.text().add(
            &mut layer,
            "Big text",
            text::TextOptions::new()
                .font_size(300.0)
                .scale(0.5)
                .origin((0.5, 0.0)),
        );

        vx.text().add(
            &mut layer,
            "Bottom Text",
            text::TextOptions::new()
                .font_size(40.0)
                .origin((0.5, 1.0))
                .translation((0.0, 1.0)),
        );

        // vx.text().add(
        //     &mut layer,
        //     "Even Bigger",
        //     text::TextOptions::new()
        //         .font_size(800.0)
        //         .origin((0.5, 0.0))
        //         .translation((0.0, -2.0)),
        // );

        let img = vx.draw_frame_copy_framebuffer();
        assert_swapchain_eq(&mut vx, "resizing_back_texture", img);
    }

    #[test]
    fn resizing_twice() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut layer = vx.text().add_layer(DEJAVU, text::LayerOptions::new());

        vx.text().add(
            &mut layer,
            "Bottom Text",
            text::TextOptions::new()
                .font_size(40.0)
                .origin((0.5, 1.0))
                .translation((0.0, 0.8)),
        );

        vx.text().add(
            &mut layer,
            "Even Bigger",
            text::TextOptions::new()
                .font_size(300.0)
                .origin((0.5, 0.0))
                .translation((0.0, -1.0)),
        );

        let img = vx.draw_frame_copy_framebuffer();
        assert_swapchain_eq(&mut vx, "resizing_twice", img);
    }

    #[test]
    fn set_glyph_opacity() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut layer = vx.text().add_layer(DEJAVU, text::LayerOptions::new());

        let text = vx.text().add(
            &mut layer,
            "Here is some text that fades with every letter",
            text::TextOptions::new().font_size(40.0).origin((0.5, 0.5)),
        );

        let cnt = vx.text().get_glyph_count(&text);
        vx.text().set_opacity_glyphs(&text, |idx| {
            ((1.0 - (idx as f32 / (cnt - 1) as f32)) * 255.0) as u8
        });

        let img = vx.draw_frame_copy_framebuffer();
        assert_swapchain_eq(&mut vx, "set_glyph_opacity", img);
    }

    #[test]
    fn do_not_resize_texture_when_making_the_same_text() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut layer = vx.text().add_layer(DEJAVU, text::LayerOptions::new());

        for idx in -10..=10 {
            vx.text().add(
                &mut layer,
                "Some text",
                text::TextOptions::new()
                    .font_size(153.0)
                    .origin(((idx as f32 + 10.0) / 20.0, 0.0))
                    .scale((idx as f32).abs() / 10.0)
                    .translation((idx as f32 / 10.0, idx as f32 / 20.0)),
            );
        }

        assert_eq![(256, 256), vx.text().get_texture_dimensions(&layer)];

        let img = vx.draw_frame_copy_framebuffer();
        assert_swapchain_eq(
            &mut vx,
            "do_not_resize_texture_when_making_the_same_text",
            img,
        );
    }

    #[test]
    fn rapidly_add_remove_layer() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        for _ in 0..10 {
            let mut text = vx.text();
            let layer = text.add_layer(DEJAVU, text::LayerOptions::new());

            text.add(&layer, "Abc", text::TextOptions::new());

            vx.draw_frame();

            vx.text().remove_layer(layer);
            assert![vx.swapconfig.image_count + 1 >= vx.text().layer_count() as u32];
            assert![0 < vx.text().layer_count()];
        }
    }

    #[bench]
    fn text_flag(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut layer = vx.text().add_layer(DEJAVU, text::LayerOptions::new());

        let handle = vx.text().add(
            &mut layer,
            "All your base are belong to us!",
            text::TextOptions::new().font_size(40.0).origin((0.5, 0.5)),
        );

        let mut degree = 0;

        b.iter(|| {
            degree = if degree == 360 { 0 } else { degree + 1 };
            vx.text().set_translation_glyphs(&handle, |idx| {
                let value = (((degree + idx * 4) as f32) / 180.0 * std::f32::consts::PI).sin();
                (0.0, value / 3.0)
            });
            vx.draw_frame();
        });
    }
}
