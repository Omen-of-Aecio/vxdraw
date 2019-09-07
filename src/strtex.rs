//! Methods and types to control streaming textures
//!
//! A streaming texture is a texture from which you can spawn sprites. The `streaming` part of
//! the name refers to the texture. The individual pixels of the texture can be read and written
//! during runtime.
//!
//! To display this texture we create sprites, which are rectangular views of the texture.
//!
//! See [strtex::Strtex] for all operations supported on streaming textures.
//!
//! # Example - Binary counter using a streaming texture #
//! Here is a binary counter using a streaming texture. The counter increments from left to right.
//! ```
//! use cgmath::{prelude::*, Deg};
//! use vxdraw::{strtex::{LayerOptions, Sprite}, void_logger, Color, ShowWindow, VxDraw};
//! fn main() {
//!     #[cfg(feature = "doctest-headless")]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Headless1k);
//!     #[cfg(not(feature = "doctest-headless"))]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Enable);
//!
//!     // Create a new layer/streaming texture, each streaming texture is on its own layer
//!     let clock = vx.strtex().add_layer(&LayerOptions::new().width(8));
//!
//!     // Create a new sprite view into this streaming texture
//!     let handle = vx.strtex().add(&clock, Sprite::new());
//!
//!     for cnt in 0..=255 {
//!
//!         // Set all pixels accoring to the current count (cnt)
//!         for idx in 0..8 {
//!             let bit_set = cnt >> idx & 1 == 1;
//!             vx.strtex().set_pixel(&clock,
//!                 idx,
//!                 0,
//!                 if bit_set {
//!                     Color::Rgba(0, (256 / 8 * idx) as u8, 0, 255)
//!                 } else {
//!                     Color::Rgba(0, 0, 0, 128)
//!                 }
//!             );
//!         }
//!
//!         // Draw the frame
//!         vx.draw_frame();
//!
//!         // Sleep here so we can see some animation
//!         #[cfg(not(feature = "doctest-headless"))]
//!         std::thread::sleep(std::time::Duration::new(0, 16_000_000));
//!     }
//! }
//! ```
use super::{blender, utils::*, Color};
use crate::data::{DrawType, StreamingTexture, StreamingTextureWrite, VxDraw};
use arrayvec::ArrayVec;
use cgmath::Matrix4;
use cgmath::Rad;
use core::ptr;
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
    format, image, memory, pass,
    pso::{self, DescriptorPool},
    Backend, Primitive,
};
use std::iter::once;
use std::{io::Cursor, mem::ManuallyDrop};

// ---

/// A view into a sprite
pub struct Handle(usize, usize);

/// Handle to a texture (layer)
pub struct Layer(usize);

impl Layerable for Layer {
    fn get_layer(&self, vx: &VxDraw) -> usize {
        for (idx, ord) in vx.draw_order.iter().enumerate() {
            match ord {
                DrawType::StreamingTexture { id } if *id == self.0 => {
                    return idx;
                }
                _ => {}
            }
        }
        panic!["Unable to get layer"]
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

/// Specify texture wrapping mode
#[derive(Clone, Copy)]
pub enum WrapMode {
    /// UV coordinates are modulo 1.0
    Tile,
    /// UV coordinates are abs modulo 1.0
    Mirror,
    /// Use the edge's value
    Clamp,
    // Border, // Not supported, need borders
}

/// Options for creating a layer of a single streaming texture with sprites
#[derive(Clone)]
pub struct LayerOptions {
    /// Perform depth testing (and fragment culling) when drawing sprites from this texture
    depth_test: bool,
    /// Fix the perspective, this ignores the perspective sent into draw for this texture and
    /// all its associated sprites
    fixed_perspective: Option<Matrix4<f32>>,
    /// Width of this texture in pixels
    width: usize,
    /// Height of this texture in pixels
    height: usize,
    /// Specify filtering mode for sampling the grid texture (default is [Filter::Nearest])
    filtering: Filter,
    /// Specify wrap mode for texture sampling
    wrap_mode: WrapMode,
    /// Blending mode for this layer
    blend: blender::Blender,
}

impl LayerOptions {
    /// Create a default layer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the sampling filter mode for the texture
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filtering = filter;
        self
    }

    /// Set the width of this layer in addressable pixels
    pub fn width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Set the height of this layer in addressable pixels
    pub fn height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    /// Enable/disable depth testing
    pub fn depth(mut self, test: bool) -> Self {
        self.depth_test = test;
        self
    }

    /// Enable a fixed perspective
    pub fn fixed_perspective(mut self, mat: Matrix4<f32>) -> Self {
        self.fixed_perspective = Some(mat);
        self
    }

    /// Set the wrap mode of the texture sampler
    pub fn wrap_mode(mut self, wrap_mode: WrapMode) -> Self {
        self.wrap_mode = wrap_mode;
        self
    }

    /// Set the blender of this layer (see [blender])
    pub fn blend(mut self, blend_setter: impl Fn(blender::Blender) -> blender::Blender) -> Self {
        self.blend = blend_setter(self.blend);
        self
    }
}

impl Default for LayerOptions {
    fn default() -> Self {
        Self {
            depth_test: false,
            filtering: Filter::Nearest,
            fixed_perspective: None,
            width: 1,
            height: 1,
            wrap_mode: WrapMode::Tile,
            blend: blender::Blender::default(),
        }
    }
}

/// Sprite creation builder
///
/// A sprite is a rectangular view into a texture. This structure sets up the necessary data to
/// call [Strtex::add] with.
#[derive(Clone, Copy)]
pub struct Sprite {
    opacity: [u8; 4],
    height: f32,
    origin: (f32, f32),
    rotation: f32,
    scale: f32,
    translation: (f32, f32),
    uv_begin: (f32, f32),
    uv_end: (f32, f32),
    width: f32,
}

impl Sprite {
    /// Same as default
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the width of the sprite
    pub fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// Set the height of the sprite
    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// Set the colors of the sprite
    ///
    /// The colors are added on top of whatever the sprite's texture data is
    pub fn opacity(mut self, opacity: u8) -> Self {
        self.opacity = [opacity; 4];
        self
    }

    /// Set the topleft corner's UV coordinates
    pub fn uv_begin(mut self, uv: (f32, f32)) -> Self {
        self.uv_begin = uv;
        self
    }

    /// Set the bottom right corner's UV coordinates
    pub fn uv_end(mut self, uv: (f32, f32)) -> Self {
        self.uv_end = uv;
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

    /// Set the scaling factor of this sprite
    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set the origin of this sprite
    pub fn origin(mut self, origin: (f32, f32)) -> Self {
        self.origin = origin;
        self
    }
}

impl Default for Sprite {
    fn default() -> Self {
        Sprite {
            width: 2.0,
            height: 2.0,
            opacity: [255; 4],
            uv_begin: (0.0, 0.0),
            uv_end: (1.0, 1.0),
            translation: (0.0, 0.0),
            rotation: 0.0,
            scale: 1.0,
            origin: (0.0, 0.0),
        }
    }
}

// ---

/// Accessor object to all streaming textures
///
/// A streaming texture is a texture which can be edited at run-time. Sprites are made from this
/// texture and drawn to the screen.
/// See [crate::strtex] for examples.
pub struct Strtex<'a> {
    vx: &'a mut VxDraw,
}

impl<'a> Strtex<'a> {
    /// Prepare to edit streaming textures
    ///
    /// You're not supposed to use this function directly (although you can).
    /// The recommended way of spawning a strtex is via [VxDraw::strtex()].
    pub fn new(vx: &'a mut VxDraw) -> Self {
        Self { vx }
    }

    /// Add a streaming texture layer to the system
    ///
    /// You use a texture to create sprites. Sprites are rectangular views into a texture. Sprites
    /// based on different texures are drawn in the order in which the textures were allocated, that
    /// means that the first texture's sprites are drawn first, then, the second texture's sprites,and
    /// so on.
    ///
    /// Each texture has options (See [LayerOptions]). This decides how the derivative sprites are
    /// drawn.
    ///
    /// Note: Alpha blending with depth testing will make foreground transparency not be transparent.
    /// To make sure transparency works correctly you can turn off the depth test for foreground
    /// objects and ensure that the foreground texture is allocated last.
    pub fn add_layer(&mut self, options: &LayerOptions) -> Layer {
        let s = &mut *self.vx;

        let device = &s.device;

        let (the_images, image_memories, image_views, image_requirements) = {
            let mut the_images = vec![];
            let mut image_memories = vec![];
            let mut image_views = vec![];
            let mut image_requirements = vec![];
            s.adapter
                .physical_device
                .image_format_properties(
                    format::Format::Rgba8Srgb,
                    2,
                    image::Tiling::Linear,
                    image::Usage::SAMPLED | image::Usage::TRANSFER_DST,
                    image::ViewCapabilities::empty(),
                )
                .expect("Device does not support linear sampled textures");
            for _ in 0..s.swapconfig.image_count {
                let mut the_image = unsafe {
                    device
                        .create_image(
                            image::Kind::D2(options.width as u32, options.height as u32, 1, 1),
                            1,
                            format::Format::Rgba8Srgb,
                            image::Tiling::Linear,
                            image::Usage::SAMPLED | image::Usage::TRANSFER_DST,
                            image::ViewCapabilities::empty(),
                        )
                        .expect("Couldn't create the image!")
                };

                let requirements = unsafe { device.get_image_requirements(&the_image) };
                let image_memory = unsafe {
                    let memory_type_id = find_memory_type_id(
                        &s.adapter,
                        requirements,
                        memory::Properties::COHERENT
                            | memory::Properties::CPU_VISIBLE
                            | memory::Properties::DEVICE_LOCAL,
                    );
                    device
                        .allocate_memory(memory_type_id, requirements.size)
                        .expect("Unable to allocate")
                };

                let image_view = unsafe {
                    device
                        .bind_image_memory(&image_memory, 0, &mut the_image)
                        .expect("Unable to bind memory");

                    device
                        .create_image_view(
                            &the_image,
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
                the_images.push(the_image);
                image_memories.push(image_memory);
                image_views.push(image_view);
                image_requirements.push(requirements);
            }
            (the_images, image_memories, image_views, image_requirements)
        };

        let sampler = unsafe {
            s.device
                .create_sampler(image::SamplerInfo::new(
                    match options.filtering {
                        Filter::Nearest => image::Filter::Nearest,
                        Filter::Linear => image::Filter::Linear,
                    },
                    match options.wrap_mode {
                        WrapMode::Tile => image::WrapMode::Tile,
                        WrapMode::Mirror => image::WrapMode::Mirror,
                        WrapMode::Clamp => image::WrapMode::Clamp,
                    },
                ))
                .expect("Couldn't create the sampler!")
        };

        const VERTEX_SOURCE_TEXTURE: &[u8] = include_bytes!["../_build/spirv/strtex.vert.spirv"];
        const FRAGMENT_SOURCE_TEXTURE: &[u8] = include_bytes!["../_build/spirv/strtex.frag.spirv"];

        let vertex_source_texture = pso::read_spirv(Cursor::new(VERTEX_SOURCE_TEXTURE)).unwrap();
        let fragment_source_texture =
            pso::read_spirv(Cursor::new(FRAGMENT_SOURCE_TEXTURE)).unwrap();

        let vs_module =
            { unsafe { s.device.create_shader_module(&vertex_source_texture) }.unwrap() };
        let fs_module =
            { unsafe { s.device.create_shader_module(&fragment_source_texture) }.unwrap() };

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
            depth: if options.depth_test {
                Some(pso::DepthTest {
                    fun: pso::Comparison::Less,
                    write: true,
                })
            } else {
                None
            },
            depth_bounds: false,
            stencil: None,
        };
        let blender = options.blend.clone().into_gfx_blender();
        let render_pass = {
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
        let descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> = unsafe {
            (0..s.swapconfig.image_count)
                .map(|_| {
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
                    s.device
                        .create_descriptor_set_layout(bindings, immutable_samplers)
                        .expect("Couldn't make a DescriptorSetLayout")
                })
                .collect::<Vec<_>>()
        };

        let mut descriptor_pool = unsafe {
            s.device
                .create_descriptor_pool(
                    s.swapconfig.image_count as usize, // sets
                    &[
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::SampledImage,
                            count: s.swapconfig.image_count as usize,
                        },
                        pso::DescriptorRangeDesc {
                            ty: pso::DescriptorType::Sampler,
                            count: s.swapconfig.image_count as usize,
                        },
                    ],
                    pso::DescriptorPoolCreateFlags::empty(),
                )
                .expect("Couldn't create a descriptor pool!")
        };

        let mut descriptor_sets = vec![];

        unsafe {
            descriptor_pool
                .allocate_sets(&descriptor_set_layouts, &mut descriptor_sets)
                .expect("Couldn't make a Descriptor Set!")
        };

        debug_assert![descriptor_sets.len() == image_views.len()];

        unsafe {
            for (idx, set) in descriptor_sets.iter().enumerate() {
                s.device.write_descriptor_sets(vec![
                    pso::DescriptorSetWrite {
                        set,
                        binding: 0,
                        array_offset: 0,
                        descriptors: Some(pso::Descriptor::Image(
                            &image_views[idx],
                            image::Layout::General,
                        )),
                    },
                    pso::DescriptorSetWrite {
                        set,
                        binding: 1,
                        array_offset: 0,
                        descriptors: Some(pso::Descriptor::Sampler(&sampler)),
                    },
                ]);
            }
        }

        let mut push_constants = Vec::<(pso::ShaderStageFlags, core::ops::Range<u32>)>::new();
        push_constants.push((pso::ShaderStageFlags::VERTEX, 0..16));
        let pipeline_layout = unsafe {
            s.device
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
            s.device
                .create_graphics_pipeline(&pipeline_desc, None)
                .expect("Couldn't create a graphics pipeline!")
        };

        unsafe {
            s.device.destroy_shader_module(vs_module);
            s.device.destroy_shader_module(fs_module);
        }

        unsafe {
            let barrier_fence = s.device.create_fence(false).expect("unable to make fence");
            // TODO Use a proper command buffer here
            s.device.wait_idle().unwrap();
            let buffer = &mut s.command_buffers[s.current_frame];
            buffer.begin(false);
            for the_image in &the_images {
                let image_barrier = memory::Barrier::Image {
                    states: (image::Access::empty(), image::Layout::Undefined)
                        ..(
                            // image::Access::HOST_READ | image::Access::HOST_WRITE,
                            image::Access::empty(),
                            image::Layout::General,
                        ),
                    target: the_image,
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
            }
            buffer.finish();
            s.queue_group.queues[0].submit_without_semaphores(Some(&*buffer), Some(&barrier_fence));
            s.device
                .wait_for_fence(&barrier_fence, u64::max_value())
                .unwrap();
            s.device.destroy_fence(barrier_fence);
        }

        let image_count = s.swapconfig.image_count;
        let posbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&s.device, &s.adapter))
            .collect::<Vec<_>>();
        let opacbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&s.device, &s.adapter))
            .collect::<Vec<_>>();
        let uvbuf = (0..image_count)
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

        let strtex = StreamingTexture {
            hidden: false,
            removed: vec![],

            width: options.width as u32,
            height: options.height as u32,

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

            image_buffer: the_images,
            image_memory: image_memories,
            image_requirements,
            image_view: image_views,

            circular_writes: (0..s.swapconfig.image_count).map(|_| vec![]).collect::<_>(),

            descriptor_pool: ManuallyDrop::new(descriptor_pool),
            sampler: ManuallyDrop::new(sampler),

            descriptor_sets,
            descriptor_set_layouts,
            pipeline: ManuallyDrop::new(pipeline),
            pipeline_layout: ManuallyDrop::new(pipeline_layout),
            render_pass: ManuallyDrop::new(render_pass),
        };

        let prev_layer = s.layer_holes.find_available(|x| match x {
            DrawType::StreamingTexture { .. } => true,
            _ => false,
        });

        let layer = if let Some(prev_layer) = prev_layer {
            match prev_layer {
                DrawType::StreamingTexture { id } => {
                    let old_strtex = std::mem::replace(&mut s.strtexs[id], strtex);
                    self.destroy_texture(old_strtex);
                    self.vx.draw_order.push(DrawType::StreamingTexture { id });
                    Layer(id)
                }
                _ => panic!["Got a non-strtex drawtype, should be impossible!"],
            }
        } else {
            s.strtexs.push(strtex);
            s.draw_order.push(DrawType::StreamingTexture {
                id: s.strtexs.len() - 1,
            });
            Layer(s.strtexs.len() - 1)
        };
        self.vx.strtex().write_all(&layer, (0, 0, 0, 0));
        layer
    }

    /// Query the amount of layers of this type there are
    pub fn layer_count(&self) -> usize {
        self.vx.strtexs.len()
    }

    /// Disable drawing of the sprites at this layer
    pub fn hide(&mut self, layer: &Layer) {
        self.vx.strtexs[layer.0].hidden = true;
    }

    /// Enable drawing of the sprites at this layer
    pub fn show(&mut self, layer: &Layer) {
        self.vx.strtexs[layer.0].hidden = false;
    }

    /// Remove a texture (layer)
    ///
    /// This also stops drawing all associated sprites, so the sprite handles that use this layer
    /// that still exist will be invalidated.
    pub fn remove_layer(&mut self, texture: Layer) {
        let s = &mut *self.vx;
        let mut index = None;
        for (idx, x) in s.draw_order.iter().enumerate() {
            match x {
                DrawType::StreamingTexture { id } if *id == texture.0 => {
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

    fn destroy_texture(&mut self, mut strtex: StreamingTexture) {
        let s = &mut *self.vx;
        unsafe {
            for mut indices in strtex.indices.drain(..) {
                indices.destroy(&s.device);
            }
            for mut posbuf in strtex.posbuf.drain(..) {
                posbuf.destroy(&s.device);
            }
            for mut opacbuf in strtex.opacbuf.drain(..) {
                opacbuf.destroy(&s.device);
            }
            for mut uvbuf in strtex.uvbuf.drain(..) {
                uvbuf.destroy(&s.device);
            }
            for mut tranbuf in strtex.tranbuf.drain(..) {
                tranbuf.destroy(&s.device);
            }
            for mut rotbuf in strtex.rotbuf.drain(..) {
                rotbuf.destroy(&s.device);
            }
            for mut scalebuf in strtex.scalebuf.drain(..) {
                scalebuf.destroy(&s.device);
            }
            for image_buffer in strtex.image_buffer.drain(..) {
                s.device.destroy_image(image_buffer);
            }
            for image_memory in strtex.image_memory.drain(..) {
                s.device.free_memory(image_memory);
            }
            s.device
                .destroy_render_pass(ManuallyDrop::into_inner(ptr::read(&strtex.render_pass)));
            s.device
                .destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(
                    &strtex.pipeline_layout,
                )));
            s.device
                .destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(&strtex.pipeline)));
            for dsl in strtex.descriptor_set_layouts.drain(..) {
                s.device.destroy_descriptor_set_layout(dsl);
            }
            s.device
                .destroy_descriptor_pool(ManuallyDrop::into_inner(ptr::read(
                    &strtex.descriptor_pool,
                )));
            s.device
                .destroy_sampler(ManuallyDrop::into_inner(ptr::read(&strtex.sampler)));
            for image_view in strtex.image_view.drain(..) {
                s.device.destroy_image_view(image_view);
            }
        }
    }

    /// Add a sprite (a rectangular view of a texture) to the system
    pub fn add(&mut self, layer: &Layer, sprite: Sprite) -> Handle {
        // Derive xy from the sprite's initial UV
        let uv_a = sprite.uv_begin;
        let uv_b = sprite.uv_end;

        let width = sprite.width;
        let height = sprite.height;

        let topleft = (
            -width / 2f32 - sprite.origin.0,
            -height / 2f32 - sprite.origin.1,
        );
        let topleft_uv = uv_a;

        let topright = (
            width / 2f32 - sprite.origin.0,
            -height / 2f32 - sprite.origin.1,
        );
        let topright_uv = (uv_b.0, uv_a.1);

        let bottomleft = (
            -width / 2f32 - sprite.origin.0,
            height / 2f32 - sprite.origin.1,
        );
        let bottomleft_uv = (uv_a.0, uv_b.1);

        let bottomright = (
            width / 2f32 - sprite.origin.0,
            height / 2f32 - sprite.origin.1,
        );
        let bottomright_uv = (uv_b.0, uv_b.1);

        let replace = self.vx.strtexs.get(layer.0).map(|x| !x.removed.is_empty());
        if replace.is_none() {
            panic!["Layer does not exist"];
        }

        let handle = if replace.unwrap() {
            let hole = self.vx.strtexs[layer.0].removed.pop().unwrap();
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
            self.set_opacity_raw(
                &handle,
                [
                    sprite.opacity[0],
                    sprite.opacity[1],
                    sprite.opacity[2],
                    sprite.opacity[3],
                ],
            );
            self.set_translation(&handle, (sprite.translation.0, sprite.translation.1));
            self.set_rotation(&handle, Rad(sprite.rotation));
            self.set_scale(&handle, sprite.scale);
            self.set_uv(&handle, sprite.uv_begin, sprite.uv_end);
            hole
        } else {
            let tex = &mut self.vx.strtexs[layer.0];
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
            tex.opacbuffer.push([
                sprite.opacity[0],
                sprite.opacity[1],
                sprite.opacity[2],
                sprite.opacity[3],
            ]);
            tex.tranbuffer.push([
                sprite.translation.0,
                sprite.translation.1,
                sprite.translation.0,
                sprite.translation.1,
                sprite.translation.0,
                sprite.translation.1,
                sprite.translation.0,
                sprite.translation.1,
            ]);
            tex.rotbuffer.push([
                sprite.rotation,
                sprite.rotation,
                sprite.rotation,
                sprite.rotation,
            ]);
            tex.scalebuffer
                .push([sprite.scale, sprite.scale, sprite.scale, sprite.scale]);
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
            tex.posbuffer.len() - 1
        };

        let tex = &mut self.vx.strtexs[layer.0];
        tex.posbuf_touch = self.vx.swapconfig.image_count;
        tex.opacbuf_touch = self.vx.swapconfig.image_count;
        tex.uvbuf_touch = self.vx.swapconfig.image_count;
        tex.tranbuf_touch = self.vx.swapconfig.image_count;
        tex.rotbuf_touch = self.vx.swapconfig.image_count;
        tex.scalebuf_touch = self.vx.swapconfig.image_count;

        Handle(layer.0, handle)
    }

    /// Removes a single sprite, making it not be drawn
    ///
    /// The sprite is set to a scale of 0 and its handle is stored internally in a list of
    /// `holes`. Calling [Strtex::add] with available holes will fill the first available hole
    /// with the new sprite.
    pub fn remove(&mut self, handle: Handle) {
        self.vx.strtexs[handle.0].scalebuf_touch = self.vx.swapconfig.image_count;
        if let Some(strtex) = self.vx.strtexs.get_mut(handle.0) {
            strtex.removed.push(handle.1);
            strtex.scalebuffer[handle.1].copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        }
    }

    /// Removes all sprites, clearing the layer
    pub fn remove_all(&mut self, layer: &Layer) {
        self.vx.strtexs[layer.0].scalebuf_touch = self.vx.swapconfig.image_count;
        if let Some(strtex) = self.vx.strtexs.get_mut(layer.0) {
            strtex.removed.clear();
            strtex.opacbuffer.clear();
            strtex.posbuffer.clear();
            strtex.rotbuffer.clear();
            strtex.scalebuffer.clear();
            strtex.tranbuffer.clear();
            strtex.uvbuffer.clear();
        }
    }

    /// Get the current number of sprites
    pub fn sprite_count(&mut self, layer: &Layer) -> usize {
        self.vx.strtexs[layer.0].posbuffer.len() - self.vx.strtexs[layer.0].removed.len()
    }

    // ---

    /// Set the color of a specific pixel
    pub fn set_pixel(&mut self, id: &Layer, w: u32, h: u32, color: Color) {
        let s = &mut *self.vx;
        if let Some(strtex) = s.strtexs.get_mut(id.0) {
            if !(w < strtex.width && h < strtex.height) {
                return;
            }
            strtex.circular_writes[s.current_frame]
                .push(StreamingTextureWrite::Single((w, h), color.into()));
        }
    }

    /// Set multiple pixels in the texture
    pub fn set_pixels(&mut self, id: &Layer, modifier: impl Iterator<Item = (u32, u32, Color)>) {
        let s = &mut *self.vx;
        if let Some(strtex) = s.strtexs.get_mut(id.0) {
            for item in modifier {
                let w = item.0;
                let h = item.1;
                let color = item.2;
                strtex.circular_writes[s.current_frame]
                    .push(StreamingTextureWrite::Single((w, h), color.into()));
            }
        }
    }

    /// Set a block of pixels
    pub fn set_pixels_block(
        &mut self,
        id: &Layer,
        start: (u32, u32),
        wh: (u32, u32),
        color: Color,
    ) {
        let s = &mut *self.vx;
        if let Some(strtex) = s.strtexs.get_mut(id.0) {
            if start.0 + wh.0 > strtex.width || start.1 + wh.1 > strtex.height {
                return;
            }
            strtex.circular_writes[s.current_frame].push(StreamingTextureWrite::Block(
                start,
                wh,
                color.into(),
            ));
        }
    }

    // ---

    /// Change the vertices of the model-space
    ///
    /// The name `set_deform` is used to keep consistent [Strtex::deform].
    /// What this function does is just setting absolute vertex positions for each vertex in the
    /// sprite.
    pub fn set_deform(&mut self, handle: &Handle, points: [(f32, f32); 4]) {
        self.vx.strtexs[handle.0].posbuf_touch = self.vx.swapconfig.image_count;
        let vertex = &mut self.vx.strtexs[handle.0].posbuffer[handle.1];
        for (idx, point) in points.iter().enumerate() {
            vertex[idx * 2] = point.0;
            vertex[idx * 2 + 1] = point.1;
        }
    }

    /// Set a solid color each vertex of a sprite
    pub fn set_opacity(&mut self, handle: &Handle, opacity: u8) {
        self.vx.strtexs[handle.0].opacbuf_touch = self.vx.swapconfig.image_count;
        self.vx.strtexs[handle.0].opacbuffer[handle.1].copy_from_slice(&[opacity; 4]);
    }

    /// Set a solid color each vertex of a sprite
    pub fn set_opacity_raw(&mut self, handle: &Handle, opacity: [u8; 4]) {
        self.vx.strtexs[handle.0].opacbuf_touch = self.vx.swapconfig.image_count;
        self.vx.strtexs[handle.0].opacbuffer[handle.1].copy_from_slice(&opacity);
    }

    /// Set the position of a sprite
    pub fn set_translation(&mut self, handle: &Handle, position: (f32, f32)) {
        self.vx.strtexs[handle.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            self.vx.strtexs[handle.0].tranbuffer[handle.1][idx * 2] = position.0;
            self.vx.strtexs[handle.0].tranbuffer[handle.1][idx * 2 + 1] = position.1;
        }
    }

    /// Set the rotation of a sprite
    ///
    /// Positive rotation goes counter-clockwise. The value of the rotation is in radians.
    pub fn set_rotation<T: Copy + Into<Rad<f32>>>(&mut self, handle: &Handle, rotation: T) {
        let angle = rotation.into().0;
        self.vx.strtexs[handle.0].rotbuf_touch = self.vx.swapconfig.image_count;
        self.vx.strtexs[handle.0].rotbuffer[handle.1]
            .copy_from_slice(&[angle, angle, angle, angle]);
    }

    /// Set the scale of a sprite
    pub fn set_scale(&mut self, handle: &Handle, scale: f32) {
        self.vx.strtexs[handle.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for sc in &mut self.vx.strtexs[handle.0].scalebuffer[handle.1] {
            *sc = scale;
        }
    }

    /// Set the UV values of a single sprite
    pub fn set_uv(&mut self, handle: &Handle, uv_begin: (f32, f32), uv_end: (f32, f32)) {
        self.vx.strtexs[handle.0].uvbuf_touch = self.vx.swapconfig.image_count;
        self.vx.strtexs[handle.0].uvbuffer[handle.1].copy_from_slice(&[
            uv_begin.0, uv_begin.1, uv_begin.0, uv_end.1, uv_end.0, uv_end.1, uv_end.0, uv_begin.1,
        ]);
    }

    /// Set the raw UV values of each vertex in a sprite
    ///
    /// This may be used to repeat a texture multiple times over the same sprite, or to do
    /// something exotic with uv coordinates.
    pub fn set_uv_raw(&mut self, handle: &Handle, uvs: [(f32, f32); 4]) {
        self.vx.strtexs[handle.0].uvbuf_touch = self.vx.swapconfig.image_count;
        self.vx.strtexs[handle.0].uvbuffer[handle.1].copy_from_slice(&[
            uvs[0].0, uvs[0].1, uvs[1].0, uvs[1].1, uvs[2].0, uvs[2].1, uvs[3].0, uvs[3].1,
        ]);
    }

    // ---

    /// Deform a sprite by adding delta vertices
    ///
    /// Adds the delta vertices to the sprite. Beware: This changes model space form.
    pub fn deform(&mut self, handle: &Handle, delta: [(f32, f32); 4]) {
        self.vx.strtexs[handle.0].posbuf_touch = self.vx.swapconfig.image_count;
        let points = &mut self.vx.strtexs[handle.0].posbuffer[handle.1];
        points[0] += delta[0].0;
        points[1] += delta[0].1;
        points[2] += delta[1].0;
        points[3] += delta[1].1;
        points[4] += delta[2].0;
        points[5] += delta[2].1;
        points[6] += delta[3].0;
        points[7] += delta[3].1;
    }

    /// Translate a sprite by a vector
    ///
    /// Translation does not mutate the model-space of a sprite.
    pub fn translate(&mut self, handle: &Handle, movement: (f32, f32)) {
        self.vx.strtexs[handle.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            self.vx.strtexs[handle.0].tranbuffer[handle.1][idx * 2] += movement.0;
            self.vx.strtexs[handle.0].tranbuffer[handle.1][idx * 2 + 1] += movement.1;
        }
    }

    /// Rotate a sprite
    ///
    /// Rotation does not mutate the model-space of a sprite.
    pub fn rotate<T: Copy + Into<Rad<f32>>>(&mut self, handle: &Handle, angle: T) {
        self.vx.strtexs[handle.0].rotbuf_touch = self.vx.swapconfig.image_count;
        for rot in &mut self.vx.strtexs[handle.0].rotbuffer[handle.1] {
            *rot += angle.into().0;
        }
    }

    /// Scale a sprite
    ///
    /// Scale does not mutate the model-space of a sprite.
    pub fn scale(&mut self, handle: &Handle, scale: f32) {
        self.vx.strtexs[handle.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for sc in &mut self.vx.strtexs[handle.0].scalebuffer[handle.1] {
            *sc *= scale;
        }
    }

    // ---

    /// Deform all strtexs by adding delta vertices
    ///
    /// Applies [Strtex::deform] to each dynamic texture.
    pub fn deform_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> [(f32, f32); 4]) {
        self.vx.strtexs[layer.0].posbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.strtexs[layer.0].posbuffer.iter_mut().enumerate() {
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

    /// Translate all strtexs by adding delta vertices
    ///
    /// Applies [Strtex::translate] to each dynamic texture.
    pub fn translate_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> (f32, f32)) {
        self.vx.strtexs[layer.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.strtexs[layer.0].tranbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for idx in 0..4 {
                quad[idx * 2] += delta.0;
                quad[idx * 2 + 1] += delta.1;
            }
        }
    }

    /// Set opacity on all sprites
    pub fn set_opacity_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> u8) {
        self.vx.strtexs[layer.0].opacbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.strtexs[layer.0].opacbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            quad.copy_from_slice(&[delta; 4]);
        }
    }

    /// Rotate all strtexs by adding delta rotations
    ///
    /// Applies [Strtex::rotate] to each dynamic texture.
    pub fn rotate_all<T: Copy + Into<Rad<f32>>>(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> T,
    ) {
        self.vx.strtexs[layer.0].rotbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.strtexs[layer.0].rotbuffer.iter_mut().enumerate() {
            let delta = delta(idx).into().0;
            for rotation in quad.iter_mut() {
                *rotation += delta;
            }
        }
    }

    /// Scale all strtexs by multiplying a delta scale
    ///
    /// Applies [Strtex::scale] to each dynamic texture.
    pub fn scale_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> f32) {
        self.vx.strtexs[layer.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.strtexs[layer.0].scalebuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for scale in quad.iter_mut() {
                *scale *= delta;
            }
        }
    }

    // ---

    /// Deform all strtexs by setting delta vertices
    ///
    /// Applies [Strtex::set_deform] to each dynamic texture.
    pub fn set_deform_all(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> [(f32, f32); 4],
    ) {
        self.vx.strtexs[layer.0].posbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.strtexs[layer.0].posbuffer.iter_mut().enumerate() {
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

    /// Set the translation on all strtexs
    ///
    /// Applies [Strtex::set_translation] to each dynamic texture.
    pub fn set_translation_all(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> (f32, f32),
    ) {
        self.vx.strtexs[layer.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.strtexs[layer.0].tranbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for idx in 0..4 {
                quad[idx * 2] = delta.0;
                quad[idx * 2 + 1] = delta.1;
            }
        }
    }

    /// Set the uv on all strtexs
    ///
    /// Applies [Strtex::set_uv] to each dynamic texture.
    pub fn set_uv_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> [(f32, f32); 2]) {
        self.vx.strtexs[layer.0].uvbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.strtexs[layer.0].uvbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            let uv_begin = delta[0];
            let uv_end = delta[1];
            quad.copy_from_slice(&[
                uv_begin.0, uv_begin.1, uv_begin.0, uv_end.1, uv_end.0, uv_end.1, uv_end.0,
                uv_begin.1,
            ]);
        }
    }

    /// Set the rotation on all strtexs
    ///
    /// Applies [Strtex::set_rotation] to each dynamic texture.
    pub fn set_rotation_all<T: Copy + Into<Rad<f32>>>(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> T,
    ) {
        self.vx.strtexs[layer.0].rotbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.strtexs[layer.0].rotbuffer.iter_mut().enumerate() {
            let delta = delta(idx).into().0;
            quad.copy_from_slice(&[delta; 4]);
        }
    }

    /// Set the scale on all strtexs
    ///
    /// Applies [Strtex::set_scale] to each dynamic texture.
    /// Note: This may re-enable removed sprites, see [Strtex::remove].
    pub fn set_scale_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> f32) {
        self.vx.strtexs[layer.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.strtexs[layer.0].scalebuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            quad.copy_from_slice(&[delta; 4]);
        }
    }
    // ---

    /// Read pixels from arbitrary coordinates
    pub fn read(&mut self, id: &Layer, mut map: impl FnMut(&[(u8, u8, u8, u8)], usize)) {
        let s = &mut *self.vx;
        let frame_number = s
            .current_frame
            .checked_sub(1)
            .map_or(s.swapconfig.image_count as usize - 1, |x| x);
        if let Some(ref strtex) = s.strtexs.get(id.0) {
            unsafe {
                let subres = s.device.get_image_subresource_footprint(
                    &strtex.image_buffer[frame_number],
                    gfx_hal::image::Subresource {
                        aspects: gfx_hal::format::Aspects::COLOR,
                        level: 0,
                        layer: 0,
                    },
                );

                let target = s
                    .device
                    .acquire_mapping_reader::<(u8, u8, u8, u8)>(
                        &strtex.image_memory[frame_number],
                        0..strtex.image_requirements[frame_number].size,
                    )
                    .expect("unable to acquire mapping writer");

                map(&target, (subres.row_pitch / 4) as usize);

                s.device.release_mapping_reader(target);
            }
        }
    }

    /// Write pixels to arbitrary coordinates
    pub fn write(&mut self, id: &Layer, mut map: impl FnMut(&mut [(u8, u8, u8, u8)], usize)) {
        let s = &mut *self.vx;
        if let Some(ref strtex) = s.strtexs.get(id.0) {
            for frame in 0..s.swapconfig.image_count {
                let frame = frame as usize;
                unsafe {
                    let subres = s.device.get_image_subresource_footprint(
                        &strtex.image_buffer[frame],
                        gfx_hal::image::Subresource {
                            aspects: gfx_hal::format::Aspects::COLOR,
                            level: 0,
                            layer: 0,
                        },
                    );

                    let mut target = s
                        .device
                        .acquire_mapping_writer::<(u8, u8, u8, u8)>(
                            &strtex.image_memory[frame],
                            0..strtex.image_requirements[frame].size,
                        )
                        .expect("unable to acquire mapping writer");

                    map(&mut target, (subres.row_pitch / 4) as usize);

                    s.device
                        .release_mapping_writer(target)
                        .expect("Unable to release mapping writer");
                }
            }
        }
    }

    /// Write a color to all pixels
    pub fn write_all(&mut self, id: &Layer, color: (u8, u8, u8, u8)) {
        let width = self.vx.strtexs[id.0].width;
        let height = self.vx.strtexs[id.0].height;
        self.write(id, |col, pitch| {
            for y in 0..height as usize {
                for x in 0..width as usize {
                    col[x + y * pitch] = color;
                }
            }
        });
    }

    /// Fills the streaming texture with perlin noise generated from an input seed
    pub fn fill_with_perlin_noise(&mut self, blitid: &Layer, seed: [f32; 3]) {
        let s = &mut *self.vx;
        for circ in &mut s.strtexs[blitid.0].circular_writes {
            circ.clear();
        }
        static VERTEX_SOURCE: &[u8] = include_bytes!("../_build/spirv/proc1.vert.spirv");
        static FRAGMENT_SOURCE: &[u8] = include_bytes!("../_build/spirv/proc1.frag.spirv");
        let w = s.strtexs[blitid.0].width;
        let h = s.strtexs[blitid.0].height;

        let vertex_source = pso::read_spirv(Cursor::new(VERTEX_SOURCE)).unwrap();
        let fragment_source = pso::read_spirv(Cursor::new(FRAGMENT_SOURCE)).unwrap();

        let vs_module = { unsafe { s.device.create_shader_module(&vertex_source) }.unwrap() };
        let fs_module = { unsafe { s.device.create_shader_module(&fragment_source) }.unwrap() };
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

        let vertex_buffers: Vec<pso::VertexBufferDesc> = vec![pso::VertexBufferDesc {
            binding: 0,
            stride: 8u32,
            rate: pso::VertexInputRate::Vertex,
        }];

        let attributes: Vec<pso::AttributeDesc> = vec![pso::AttributeDesc {
            location: 0,
            binding: 0,
            element: pso::Element {
                format: format::Format::Rg32Sfloat,
                offset: 0,
            },
        }];

        let rasterizer = pso::Rasterizer {
            depth_clamping: false,
            polygon_mode: pso::PolygonMode::Fill,
            cull_face: pso::Face::NONE,
            front_face: pso::FrontFace::CounterClockwise,
            depth_bias: None,
            conservative: false,
        };

        let depth_stencil = pso::DepthStencilDesc {
            depth: None,
            depth_bounds: false,
            stencil: None,
        };

        let blender = {
            let blend_state = pso::BlendState {
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
                targets: vec![pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: Some(blend_state),
                }],
            }
        };

        let extent = image::Extent {
            // width: s.swapconfig.extent.width,
            // height: s.swapconfig.extent.height,
            width: w,
            height: h,
            depth: 1,
        }
        .rect();

        let mapgen_render_pass = {
            let attachment = pass::Attachment {
                format: Some(format::Format::Rgba8Srgb),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: image::Layout::General..image::Layout::General,
            };

            let subpass = pass::SubpassDesc {
                colors: &[(0, image::Layout::General)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            unsafe { s.device.create_render_pass(&[attachment], &[subpass], &[]) }
                .expect("Can't create render pass")
        };

        let baked_states = pso::BakedStates {
            viewport: Some(pso::Viewport {
                rect: extent,
                depth: (0.0..1.0),
            }),
            scissor: Some(extent),
            blend_color: None,
            depth_bounds: None,
        };
        let bindings = Vec::<pso::DescriptorSetLayoutBinding>::new();
        let immutable_samplers = Vec::<<back::Backend as Backend>::Sampler>::new();
        let mut mapgen_descriptor_set_layouts: Vec<
            <back::Backend as Backend>::DescriptorSetLayout,
        > = vec![unsafe {
            s.device
                .create_descriptor_set_layout(bindings, immutable_samplers)
                .expect("Couldn't make a DescriptorSetLayout")
        }];
        let mut push_constants = Vec::<(pso::ShaderStageFlags, core::ops::Range<u32>)>::new();
        push_constants.push((pso::ShaderStageFlags::FRAGMENT, 0..4));

        let mapgen_pipeline_layout = unsafe {
            s.device
                .create_pipeline_layout(&mapgen_descriptor_set_layouts, push_constants)
                .expect("Couldn't create a pipeline layout")
        };

        // Describe the pipeline (rasterization, mapgen interpretation)
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
            layout: &mapgen_pipeline_layout,
            subpass: pass::Subpass {
                index: 0,
                main_pass: &mapgen_render_pass,
            },
            flags: pso::PipelineCreationFlags::empty(),
            parent: pso::BasePipeline::None,
        };

        let mapgen_pipeline = unsafe {
            s.device
                .create_graphics_pipeline(&pipeline_desc, None)
                .expect("Couldn't create a graphics pipeline!")
        };

        unsafe {
            s.device.destroy_shader_module(vs_module);
            s.device.destroy_shader_module(fs_module);
        }

        // ---

        unsafe {
            let mut image = s
                .device
                .create_image(
                    image::Kind::D2(w, h, 1, 1),
                    1,
                    format::Format::Rgba8Srgb,
                    // image::Tiling::Linear: `usage` only includes VK_IMAGE_USAGE_TRANSFER_SRC_BIT and/or VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    image::Tiling::Optimal,
                    image::Usage::COLOR_ATTACHMENT
                        | image::Usage::TRANSFER_SRC
                        | image::Usage::SAMPLED,
                    image::ViewCapabilities::empty(),
                )
                .expect("Unable to create image");
            let requirements = s.device.get_image_requirements(&image);
            let memory_type_id =
                find_memory_type_id(&s.adapter, requirements, memory::Properties::DEVICE_LOCAL);
            let memory = s
                .device
                .allocate_memory(memory_type_id, requirements.size)
                .expect("Unable to allocate memory");
            let image_view = {
                s.device
                    .bind_image_memory(&memory, 0, &mut image)
                    .expect("Unable to bind memory");

                s.device
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

            let framebuffer = s
                .device
                .create_framebuffer(
                    &mapgen_render_pass,
                    vec![&image_view],
                    image::Extent {
                        width: w,
                        height: h,
                        depth: 1,
                    },
                )
                .expect("fbo");

            #[rustfmt::skip]
            let (pt_buffer, pt_memory, _) = make_vertex_buffer_with_data(
                s,
                &[
                    -1.0, -1.0,
                    1.0, -1.0,
                    1.0, 1.0,
                    1.0, 1.0,
                    -1.0, 1.0,
                    -1.0, -1.0,
                ],
            );

            let mut cmd_buffer = s.command_pool.acquire_command_buffer::<command::OneShot>();
            let clear_values = [command::ClearValue::Color(command::ClearColor::Sfloat([
                1.0f32, 0.25, 0.5, 0.75,
            ]))];
            cmd_buffer.begin();
            {
                let image_barrier = memory::Barrier::Image {
                    states: (image::Access::empty(), image::Layout::Undefined)
                        ..(image::Access::SHADER_WRITE, image::Layout::General),
                    target: &image,
                    families: None,
                    range: image::SubresourceRange {
                        aspects: format::Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                };
                cmd_buffer.pipeline_barrier(
                    pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::FRAGMENT_SHADER,
                    memory::Dependencies::empty(),
                    &[image_barrier],
                );
                let mut enc = cmd_buffer.begin_render_pass_inline(
                    &mapgen_render_pass,
                    &framebuffer,
                    extent,
                    clear_values.iter(),
                );
                enc.bind_graphics_pipeline(&mapgen_pipeline);
                enc.push_graphics_constants(
                    &mapgen_pipeline_layout,
                    pso::ShaderStageFlags::FRAGMENT,
                    0,
                    &(std::mem::transmute::<[f32; 4], [u32; 4]>([
                        w as f32, seed[0], seed[1], seed[2],
                    ])),
                );
                let buffers: ArrayVec<[_; 1]> = [(&pt_buffer, 0)].into();
                enc.bind_vertex_buffers(0, buffers);
                enc.draw(0..6, 0..1);
            }
            cmd_buffer.finish();
            let upload_fence = s
                .device
                .create_fence(false)
                .expect("Couldn't create an upload fence!");
            s.queue_group.queues[0]
                .submit_without_semaphores(Some(&cmd_buffer), Some(&upload_fence));
            s.device
                .wait_for_fence(&upload_fence, u64::max_value())
                .expect("Unable to wait for fence");
            s.device
                .reset_fence(&upload_fence)
                .expect("Unable to wait for fence");

            cmd_buffer.begin();
            for image_buffer in &s.strtexs[blitid.0].image_buffer {
                cmd_buffer.copy_image(
                    &image,
                    image::Layout::General,
                    image_buffer,
                    image::Layout::General,
                    once(command::ImageCopy {
                        src_subresource: image::SubresourceLayers {
                            aspects: format::Aspects::COLOR,
                            level: 0,
                            layers: 0..1,
                        },
                        src_offset: image::Offset { x: 0, y: 0, z: 0 },
                        dst_subresource: image::SubresourceLayers {
                            aspects: format::Aspects::COLOR,
                            level: 0,
                            layers: 0..1,
                        },
                        dst_offset: image::Offset { x: 0, y: 0, z: 0 },
                        extent: image::Extent {
                            width: w,
                            height: h,
                            depth: 1,
                        },
                    }),
                );
            }
            cmd_buffer.finish();
            s.queue_group.queues[0]
                .submit_without_semaphores(Some(&cmd_buffer), Some(&upload_fence));
            s.device
                .wait_for_fence(&upload_fence, u64::max_value())
                .expect("Unable to wait for fence");

            s.device.destroy_fence(upload_fence);
            s.command_pool.free(once(cmd_buffer));

            s.device.destroy_buffer(pt_buffer);
            s.device.free_memory(pt_memory);
            s.device.destroy_pipeline_layout(mapgen_pipeline_layout);
            s.device.destroy_graphics_pipeline(mapgen_pipeline);
            for desc_set_layout in mapgen_descriptor_set_layouts.drain(..) {
                s.device.destroy_descriptor_set_layout(desc_set_layout);
            }
            s.device.destroy_render_pass(mapgen_render_pass);
            s.device.destroy_framebuffer(framebuffer);
            s.device.destroy_image_view(image_view);
            s.device.destroy_image(image);
            s.device.free_memory(memory);
        }
    }
}

// ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use cgmath::Deg;
    use fast_logger::{Generic, GenericLogger, Logger};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg as random;
    use test::{black_box, Bencher};

    #[test]
    fn generate_map_randomly() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(&LayerOptions::new().width(1000).height(1000));
        strtex.add(&id, Sprite::new());
        strtex.fill_with_perlin_noise(&id, [0.0, 0.0, 0.0]);

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "generate_map_randomly", img);
    }

    #[test]
    fn with_origin_11() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(&LayerOptions::new().width(1000).height(1000));
        strtex.add(&id, Sprite::new().origin((1.0, 1.0)));
        strtex.fill_with_perlin_noise(&id, [0.0, 0.0, 0.0]);

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "with_origin_11", img);
    }

    #[test]
    fn streaming_texture_blocks() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();

        let id = strtex.add_layer(&LayerOptions::new().width(1000).height(1000));
        strtex.add(&id, strtex::Sprite::new());

        strtex.set_pixels_block(&id, (0, 0), (500, 500), Color::Rgba(255, 0, 0, 255));
        strtex.set_pixels_block(&id, (500, 0), (500, 500), Color::Rgba(0, 255, 0, 255));
        strtex.set_pixels_block(&id, (0, 500), (500, 500), Color::Rgba(0, 0, 255, 255));
        strtex.set_pixels_block(&id, (500, 500), (500, 500), Color::Rgba(0, 0, 0, 0));

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "streaming_texture_blocks", img);
    }

    #[test]
    fn streaming_texture_blocks_off_by_one() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(&LayerOptions::new().width(10).height(1));
        strtex.add(&id, strtex::Sprite::new());

        strtex.set_pixels_block(&id, (0, 0), (10, 1), Color::Rgba(0, 255, 0, 255));

        strtex.set_pixels_block(&id, (3, 0), (1, 1), Color::Rgba(0, 0, 255, 255));

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "streaming_texture_blocks_off_by_one", img);

        let mut strtex = vx.strtex();
        strtex.set_pixels_block(&id, (3, 0), (0, 1), Color::Rgba(255, 0, 255, 255));

        strtex.set_pixels_block(&id, (3, 0), (0, 0), Color::Rgba(255, 0, 255, 255));

        strtex.set_pixels_block(&id, (3, 0), (1, 0), Color::Rgba(255, 0, 255, 255));

        strtex.set_pixels_block(&id, (30, 0), (800, 0), Color::Rgba(255, 0, 255, 255));

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "streaming_texture_blocks_off_by_one", img);
    }

    #[test]
    fn use_read_only_works_after_frame_drawn() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(&LayerOptions::new().width(10).height(10));
        strtex.set_pixel(&id, 3, 2, Color::Rgba(0, 123, 0, 255));
        let mut green_value = 0;
        strtex.read(&id, |arr, pitch| {
            green_value = arr[3 + 2 * pitch].1;
        });
        assert_ne![123, green_value];
        vx.draw_frame();
        vx.strtex().read(&id, |arr, pitch| {
            green_value = arr[3 + 2 * pitch].1;
        });
        assert_eq![123, green_value];
    }

    #[test]
    fn use_write() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(&LayerOptions::new().width(10).height(10));
        strtex.set_pixel(&id, 3, 2, Color::Rgba(0, 123, 0, 255));
        strtex.write(&id, |arr, pitch| {
            arr[3 + 2 * pitch].1 = 124;
        });

        let mut green_value = 0;
        strtex.read(&id, |arr, pitch| {
            green_value = arr[3 + 2 * pitch].1;
        });
        assert_eq![124, green_value];
    }

    #[test]
    fn streaming_texture_weird_pixel_accesses() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();

        let id = strtex.add_layer(&LayerOptions::new().width(20).height(20));
        strtex.add(&id, strtex::Sprite::new());

        let mut rng = random::new(0);

        for _ in 0..1000 {
            let x = rng.gen_range(0, 30);
            let y = rng.gen_range(0, 30);

            strtex.set_pixel(&id, x, y, Color::Rgba(0, 255, 0, 255));
            strtex.set_pixels(&id, once((x, y, Color::Rgba(0, 255, 0, 255))));
        }
    }

    #[test]
    fn streaming_texture_weird_block_accesses() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(&LayerOptions::new().width(64).height(64));
        strtex.add(&id, strtex::Sprite::new());

        let mut rng = random::new(0);

        for _ in 0..1000 {
            let start = (rng.gen_range(0, 100), rng.gen_range(0, 100));
            let wh = (rng.gen_range(0, 100), rng.gen_range(0, 100));

            strtex.set_pixels_block(&id, start, wh, Color::Rgba(0, 255, 0, 255));
        }
    }

    #[test]
    fn strtex_mass_manip() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let layer = vx
            .strtex()
            .add_layer(&LayerOptions::new().width(10).height(5));
        vx.strtex().write(&layer, |color, pitch| {
            color[5 + 0 * pitch].2 = 255;
            color[5 + 0 * pitch].3 = 255;

            color[5 + 1 * pitch].2 = 255;
            color[5 + 1 * pitch].3 = 255;

            color[4 + 1 * pitch].1 = 255;
            color[4 + 1 * pitch].3 = 255;
            color[6 + 1 * pitch].1 = 255;
            color[6 + 1 * pitch].3 = 255;

            color[0 + 4 * pitch].0 = 255;
            color[0 + 4 * pitch].3 = 255;
            color[9 + 4 * pitch].0 = 255;
            color[9 + 4 * pitch].3 = 255;
        });

        use rand::Rng;
        use rand_pcg::Pcg64Mcg as random;
        let mut rng = random::new(0);

        let quad = strtex::Sprite::new();

        for _ in 0..1000 {
            vx.strtex().add(&layer, quad);
        }

        for _ in 0..vx.buffer_count() {
            vx.draw_frame();
        }

        vx.strtex().set_translation_all(&layer, |idx| {
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

        vx.strtex()
            .set_scale_all(&layer, |idx| if idx < 500 { 0.1 } else { 0.2 });

        vx.strtex().set_opacity_all(&layer, |idx| {
            if idx < 250 {
                64
            } else if idx < 500 {
                128
            } else if idx < 750 {
                196
            } else {
                255
            }
        });

        vx.strtex()
            .set_rotation_all(&layer, |idx| if idx < 500 { Deg(0.0) } else { Deg(30.0) });

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "strtex_mass_manip", img);
    }

    #[test]
    fn wrap_mode_clamp() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();
        let options = &LayerOptions::new()
            .width(1000)
            .height(1000)
            .wrap_mode(WrapMode::Clamp);
        let testure = strtex.add_layer(options);
        strtex.fill_with_perlin_noise(&testure, [1.0, 2.0, 3.0]);
        let sprite = strtex.add(&testure, Sprite::new());
        strtex.set_uv_raw(&sprite, [(-0.5, 0.0), (-0.5, 1.0), (1.0, 1.0), (1.0, 0.0)]);

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "wrap_mode_clamp_strtex", img);
    }

    #[test]
    fn wrap_mode_mirror() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();
        let options = &LayerOptions::new()
            .width(1000)
            .height(1000)
            .wrap_mode(WrapMode::Mirror);
        let testure = strtex.add_layer(options);
        strtex.fill_with_perlin_noise(&testure, [1.0, 2.0, 3.0]);
        let sprite = strtex.add(&testure, Sprite::new());
        strtex.set_uv_raw(&sprite, [(-0.5, 0.0), (-0.5, 1.0), (1.0, 1.0), (1.0, 0.0)]);

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "wrap_mode_mirror_strtex", img);
    }

    #[test]
    fn rapidly_add_remove_layer() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let options = &LayerOptions::new();

        for _ in 0..10 {
            let mut strtex = vx.strtex();
            let layer = strtex.add_layer(options);

            strtex.add(&layer, Sprite::new());

            vx.draw_frame();

            vx.strtex().remove_layer(layer);
            assert![vx.swapconfig.image_count + 1 >= vx.strtex().layer_count() as u32];
            assert![0 < vx.strtex().layer_count()];
        }
    }

    // ---

    #[bench]
    fn bench_streaming_texture_set_single_pixel_while_drawing(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let id = vx
            .strtex()
            .add_layer(&LayerOptions::new().width(50).height(50));
        vx.strtex().add(&id, strtex::Sprite::new());

        b.iter(|| {
            vx.strtex()
                .set_pixel(&id, black_box(1), black_box(2), Color::Rgba(255, 0, 0, 255));
            vx.draw_frame();
        });
    }

    #[bench]
    fn bench_streaming_texture_set_500x500_area(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let id = vx
            .strtex()
            .add_layer(&LayerOptions::new().width(1000).height(1000));
        vx.strtex().add(&id, strtex::Sprite::new());

        b.iter(|| {
            vx.strtex()
                .set_pixels_block(&id, (0, 0), (500, 500), Color::Rgba(255, 0, 0, 255));
        });
    }

    #[bench]
    fn bench_streaming_texture_set_500x500_area_using_iterator(b: &mut Bencher) {
        use itertools::Itertools;
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let id = vx
            .strtex()
            .add_layer(&LayerOptions::new().width(1000).height(1000));
        vx.strtex().add(&id, strtex::Sprite::new());

        b.iter(|| {
            vx.strtex().set_pixels(
                &id,
                (0..500)
                    .cartesian_product(0..500)
                    .map(|(x, y)| (x, y, Color::Rgba(255, 0, 0, 255))),
            );
        });
    }

    #[bench]
    fn bench_streaming_texture_set_single_pixel(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let id = vx
            .strtex()
            .add_layer(&LayerOptions::new().width(1000).height(1000));
        vx.strtex().add(&id, strtex::Sprite::new());

        b.iter(|| {
            vx.strtex()
                .set_pixel(&id, black_box(1), black_box(2), Color::Rgba(255, 0, 0, 255));
        });
    }

    #[bench]
    fn adding_sprites(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let layer = vx
            .strtex()
            .add_layer(&LayerOptions::new().width(1000).height(1000));

        b.iter(|| {
            vx.strtex().add(&layer, strtex::Sprite::new());
            if vx.strtex().sprite_count(&layer) > 1000 {
                vx.strtex().remove_all(&layer);
            }
        });
    }

    #[bench]
    fn generate_perlin_noise(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let layer = vx
            .strtex()
            .add_layer(&LayerOptions::new().width(1000).height(1000));

        b.iter(|| {
            vx.strtex().fill_with_perlin_noise(&layer, [1.0, 2.0, 3.0]);
        });
    }
}
