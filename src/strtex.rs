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
//! use cgmath::{prelude::*, Deg, Matrix4};
//! use logger::{Generic, GenericLogger, Logger};
//! use vxdraw::{strtex::{LayerOptions, Sprite}, ShowWindow, VxDraw};
//! fn main() {
//!     let mut vx = VxDraw::new(Logger::<Generic>::spawn_test().to_compatibility(),
//!         ShowWindow::Headless1k); // Change this to ShowWindow::Enable to show the window
//!
//!     // Create a new layer/streaming texture, each streaming texture is on its own layer
//!     let clock = vx.strtex().add_layer(LayerOptions::new().width(8));
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
//!                     (0, (256 / 8 * idx) as u8, 0, 255)
//!                 } else {
//!                     (0, 0, 0, 128)
//!                 }
//!             );
//!         }
//!
//!         // Draw the frame with the identity matrix transformation (meaning no transformations)
//!         vx.draw_frame(&Matrix4::identity());
//!
//!         // Sleep here so we can see some animation
//!         #[cfg(not(test))]
//!         std::thread::sleep(std::time::Duration::new(0, 16_000_000));
//!     }
//! }
//! ```
use super::utils::*;
use crate::data::{DrawType, StreamingTexture, VxDraw};
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
    command,
    device::Device,
    format, image, memory, pass,
    pso::{self, DescriptorPool},
    Backend, Primitive,
};
use logger::debug;
use std::iter::once;
use std::mem::ManuallyDrop;

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

/// Options for creating a layer of a single streaming texture with sprites
#[derive(Clone, Copy)]
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
}

impl LayerOptions {
    /// Create a default layer
    pub fn new() -> Self {
        Self::default()
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
}

impl Default for LayerOptions {
    fn default() -> Self {
        Self {
            depth_test: true,
            fixed_perspective: None,
            width: 1,
            height: 1,
        }
    }
}

/// Sprite creation builder
///
/// A sprite is a rectangular view into a texture. This structure sets up the necessary data to
/// call [Strtex::add] with.
pub struct Sprite {
    colors: [(u8, u8, u8, u8); 4],
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
    pub fn colors(mut self, colors: [(u8, u8, u8, u8); 4]) -> Self {
        self.colors = colors;
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
            colors: [(0, 0, 0, 255); 4],
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

    /// Disable drawing of the sprites at this layer
    pub fn hide(&mut self, layer: &Layer) {
        self.vx.strtexs[layer.0].hidden = true;
    }

    /// Enable drawing of the sprites at this layer
    pub fn show(&mut self, layer: &Layer) {
        self.vx.strtexs[layer.0].hidden = false;
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
    pub fn add_layer(&mut self, options: LayerOptions) -> Layer {
        let s = &mut *self.vx;

        let device = &s.device;

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
                memory::Properties::CPU_VISIBLE | memory::Properties::COHERENT,
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

        let sampler = unsafe {
            s.device
                .create_sampler(image::SamplerInfo::new(
                    image::Filter::Nearest,
                    image::WrapMode::Tile,
                ))
                .expect("Couldn't create the sampler!")
        };

        const VERTEX_SOURCE_TEXTURE: &[u8] = include_bytes!["../_build/spirv/strtex.vert.spirv"];

        const FRAGMENT_SOURCE_TEXTURE: &[u8] = include_bytes!["../_build/spirv/strtex.frag.spirv"];

        let vs_module =
            { unsafe { s.device.create_shader_module(&VERTEX_SOURCE_TEXTURE) }.unwrap() };
        let fs_module =
            { unsafe { s.device.create_shader_module(&FRAGMENT_SOURCE_TEXTURE) }.unwrap() };

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
        debug![s.log, "vxdraw", "After making"];
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
                    format: format::Format::Rgba8Unorm,
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
                pso::DepthTest::On {
                    fun: pso::Comparison::Less,
                    write: true,
                }
            } else {
                pso::DepthTest::Off
            },
            depth_bounds: false,
            stencil: pso::StencilTest::Off,
        };
        let blender = {
            let blend_state = pso::BlendState::On {
                color: pso::BlendOp::Add {
                    src: pso::Factor::SrcAlpha,
                    dst: pso::Factor::OneMinusSrcAlpha,
                },
                alpha: pso::BlendOp::Add {
                    src: pso::Factor::One,
                    dst: pso::Factor::OneMinusSrcAlpha,
                },
            };
            pso::BlendDesc {
                logic_op: Some(pso::LogicOp::Copy),
                targets: vec![pso::ColorBlendDesc(pso::ColorMask::ALL, blend_state)],
            }
        };
        let triangle_render_pass = {
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
        let triangle_descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> =
            vec![unsafe {
                s.device
                    .create_descriptor_set_layout(bindings, immutable_samplers)
                    .expect("Couldn't make a DescriptorSetLayout")
            }];

        let mut descriptor_pool = unsafe {
            s.device
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
                .allocate_set(&triangle_descriptor_set_layouts[0])
                .expect("Couldn't make a Descriptor Set!")
        };

        unsafe {
            s.device.write_descriptor_sets(vec![
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

        let mut push_constants = Vec::<(pso::ShaderStageFlags, core::ops::Range<u32>)>::new();
        push_constants.push((pso::ShaderStageFlags::VERTEX, 0..16));
        let triangle_pipeline_layout = unsafe {
            s.device
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
            {
                let image_barrier = memory::Barrier::Image {
                    states: (image::Access::empty(), image::Layout::Undefined)
                        ..(
                            // image::Access::HOST_READ | image::Access::HOST_WRITE,
                            image::Access::empty(),
                            image::Layout::General,
                        ),
                    target: &the_image,
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
            s.queue_group.queues[0].submit_nosemaphores(Some(&*buffer), Some(&barrier_fence));
            s.device
                .wait_for_fence(&barrier_fence, u64::max_value())
                .unwrap();
            s.device.destroy_fence(barrier_fence);
        }

        let image_count = s.swapconfig.image_count;
        let posbuf = (0..image_count)
            .map(|_| super::utils::ResizBuf::new(&s.device, &s.adapter))
            .collect::<Vec<_>>();
        let colbuf = (0..image_count)
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

        s.strtexs.push(StreamingTexture {
            hidden: false,
            count: 0,
            removed: vec![],

            width: options.width as u32,
            height: options.height as u32,

            posbuf_touch: 0,
            colbuf_touch: 0,
            uvbuf_touch: 0,
            tranbuf_touch: 0,
            rotbuf_touch: 0,
            scalebuf_touch: 0,

            posbuffer: vec![],
            colbuffer: vec![],
            uvbuffer: vec![],
            tranbuffer: vec![],
            rotbuffer: vec![],
            scalebuffer: vec![],

            posbuf,
            colbuf,
            uvbuf,
            tranbuf,
            rotbuf,
            scalebuf,
            indices,

            image_buffer: ManuallyDrop::new(the_image),
            image_memory: ManuallyDrop::new(image_memory),
            image_requirements: requirements,

            descriptor_pool: ManuallyDrop::new(descriptor_pool),
            image_view: ManuallyDrop::new(image_view),
            sampler: ManuallyDrop::new(sampler),

            descriptor_set: ManuallyDrop::new(descriptor_set),
            descriptor_set_layouts: triangle_descriptor_set_layouts,
            pipeline: ManuallyDrop::new(triangle_pipeline),
            pipeline_layout: ManuallyDrop::new(triangle_pipeline_layout),
            render_pass: ManuallyDrop::new(triangle_render_pass),
        });
        s.draw_order.push(DrawType::StreamingTexture {
            id: s.strtexs.len() - 1,
        });
        Layer(s.strtexs.len() - 1)
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
            s.draw_order.remove(idx);
            // Can't delete here always because other textures may still be referring to later strtexs,
            // only when this is the last texture.
            if s.strtexs.len() == texture.0 + 1 {
                let strtex = s.strtexs.pop().unwrap();
                self.destroy_texture(strtex);
            }
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
            for mut colbuf in strtex.colbuf.drain(..) {
                colbuf.destroy(&s.device);
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
            s.device
                .destroy_image(ManuallyDrop::into_inner(ptr::read(&strtex.image_buffer)));
            s.device
                .free_memory(ManuallyDrop::into_inner(ptr::read(&strtex.image_memory)));
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
            s.device
                .destroy_image_view(ManuallyDrop::into_inner(ptr::read(&strtex.image_view)));
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
            self.set_color(
                &handle,
                [
                    sprite.colors[0].0,
                    sprite.colors[0].1,
                    sprite.colors[0].2,
                    sprite.colors[0].3,
                    sprite.colors[1].0,
                    sprite.colors[1].1,
                    sprite.colors[1].2,
                    sprite.colors[1].3,
                    sprite.colors[2].0,
                    sprite.colors[2].1,
                    sprite.colors[2].2,
                    sprite.colors[2].3,
                    sprite.colors[3].0,
                    sprite.colors[3].1,
                    sprite.colors[3].2,
                    sprite.colors[3].3,
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
            tex.colbuffer.push([
                sprite.colors[0].0,
                sprite.colors[0].1,
                sprite.colors[0].2,
                sprite.colors[0].3,
                sprite.colors[1].0,
                sprite.colors[1].1,
                sprite.colors[1].2,
                sprite.colors[1].3,
                sprite.colors[2].0,
                sprite.colors[2].1,
                sprite.colors[2].2,
                sprite.colors[2].3,
                sprite.colors[3].0,
                sprite.colors[3].1,
                sprite.colors[3].2,
                sprite.colors[3].3,
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
        tex.colbuf_touch = self.vx.swapconfig.image_count;
        tex.uvbuf_touch = self.vx.swapconfig.image_count;
        tex.tranbuf_touch = self.vx.swapconfig.image_count;
        tex.rotbuf_touch = self.vx.swapconfig.image_count;
        tex.scalebuf_touch = self.vx.swapconfig.image_count;

        Handle(layer.0, handle)
    }

    // ---

    /// Set the color of a specific pixel
    pub fn set_pixel(&mut self, id: &Layer, w: u32, h: u32, color: (u8, u8, u8, u8)) {
        let s = &mut *self.vx;
        if let Some(ref strtex) = s.strtexs.get(id.0) {
            if !(w < strtex.width && h < strtex.height) {
                return;
            }
            unsafe {
                let foot = s.device.get_image_subresource_footprint(
                    &strtex.image_buffer,
                    image::Subresource {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layer: 0,
                    },
                );
                let access = foot.row_pitch * u64::from(h) + u64::from(w * 4);

                let aligned = perfect_mapping_alignment(Align {
                    access_offset: access,
                    how_many_bytes_you_need: 4,
                    non_coherent_atom_size: s.device_limits.non_coherent_atom_size as u64,
                });

                s.device
                    .wait_for_fences(
                        &s.frames_in_flight_fences,
                        gfx_hal::device::WaitFor::All,
                        u64::max_value(),
                    )
                    .expect("Unable to wait for fences");

                let mut target = s
                    .device
                    .acquire_mapping_writer(&*strtex.image_memory, aligned.begin..aligned.end)
                    .expect("unable to acquire mapping writer");

                target[aligned.index_offset as usize..(aligned.index_offset + 4) as usize]
                    .copy_from_slice(&[color.0, color.1, color.2, color.3]);

                s.device
                    .release_mapping_writer(target)
                    .expect("Unable to release mapping writer");
            }
        }
    }

    /// Set multiple pixels in the texture
    pub fn set_pixels(
        &mut self,
        id: &Layer,
        modifier: impl Iterator<Item = (u32, u32, (u8, u8, u8, u8))>,
    ) {
        let s = &mut *self.vx;
        if let Some(ref strtex) = s.strtexs.get(id.0) {
            unsafe {
                let foot = s.device.get_image_subresource_footprint(
                    &strtex.image_buffer,
                    image::Subresource {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layer: 0,
                    },
                );

                s.device
                    .wait_for_fences(
                        &s.frames_in_flight_fences,
                        gfx_hal::device::WaitFor::All,
                        u64::max_value(),
                    )
                    .expect("Unable to wait for fences");

                let mut target = s
                    .device
                    .acquire_mapping_writer(
                        &*strtex.image_memory,
                        0..strtex.image_requirements.size,
                    )
                    .expect("unable to acquire mapping writer");

                for item in modifier {
                    let w = item.0;
                    let h = item.1;
                    let color = item.2;

                    if !(w < strtex.width && h < strtex.height) {
                        continue;
                    }

                    let access = foot.row_pitch * u64::from(h) + u64::from(w * 4);

                    target[access as usize..(access + 4) as usize]
                        .copy_from_slice(&[color.0, color.1, color.2, color.3]);
                }
                s.device
                    .release_mapping_writer(target)
                    .expect("Unable to release mapping writer");
            }
        }
    }

    /// Set a block of pixels
    pub fn set_pixels_block(
        &mut self,
        id: &Layer,
        start: (u32, u32),
        wh: (u32, u32),
        color: (u8, u8, u8, u8),
    ) {
        let s = &mut *self.vx;
        if let Some(ref strtex) = s.strtexs.get(id.0) {
            if start.0 + wh.0 > strtex.width || start.1 + wh.1 > strtex.height {
                return;
            }
            unsafe {
                let foot = s.device.get_image_subresource_footprint(
                    &strtex.image_buffer,
                    image::Subresource {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layer: 0,
                    },
                );

                // Vulkan 01390, Size must be a multiple of DeviceLimits:nonCoherentAtomSize, or offset
                // plus size = size of memory, if it's not VK_WHOLE_SIZE
                let access_begin = foot.row_pitch * u64::from(start.1) + u64::from(start.0 * 4);
                let access_end = foot.row_pitch
                    * u64::from(start.1 + if wh.1 == 0 { 0 } else { wh.1 - 1 })
                    + u64::from((start.0 + wh.0) * 4);

                debug_assert![access_end <= strtex.image_requirements.size];

                let aligned = perfect_mapping_alignment(Align {
                    access_offset: access_begin,
                    how_many_bytes_you_need: access_end - access_begin,
                    non_coherent_atom_size: s.device_limits.non_coherent_atom_size as u64,
                });

                s.device
                    .wait_for_fences(
                        &s.frames_in_flight_fences,
                        gfx_hal::device::WaitFor::All,
                        u64::max_value(),
                    )
                    .expect("Unable to wait for fences");

                let mut target = s
                    .device
                    .acquire_mapping_writer::<u8>(&*strtex.image_memory, aligned.begin..aligned.end)
                    .expect("unable to acquire mapping writer");

                let mut colbuff = vec![];
                for _ in start.0..start.0 + wh.0 {
                    colbuff.extend(&[color.0, color.1, color.2, color.3]);
                }

                for idx in start.1..start.1 + wh.1 {
                    let idx = (idx - start.1) as usize;
                    let pitch = foot.row_pitch as usize;
                    target[aligned.index_offset as usize + idx * pitch
                        ..aligned.index_offset as usize + idx * pitch + (wh.0) as usize * 4]
                        .copy_from_slice(&colbuff);
                }
                s.device
                    .release_mapping_writer(target)
                    .expect("Unable to release mapping writer");
            }
        }
    }

    // ---

    /// Change the vertices of the model-space
    ///
    /// The name `set_deform` is used to keep consistent [Strtex::deform].
    /// What this function does is just setting absolute vertex positions for each vertex in the
    /// triangle.
    pub fn set_deform(&mut self, handle: &Handle, points: [(f32, f32); 4]) {
        self.vx.strtexs[handle.0].posbuf_touch = self.vx.swapconfig.image_count;
        let vertex = &mut self.vx.strtexs[handle.0].posbuffer[handle.1];
        for (idx, point) in points.iter().enumerate() {
            vertex[idx * 2] = point.0;
            vertex[idx * 2 + 1] = point.1;
        }
    }

    /// Set a solid color each vertex of a sprite
    pub fn set_color(&mut self, handle: &Handle, rgba: [u8; 16]) {
        self.vx.strtexs[handle.0].colbuf_touch = self.vx.swapconfig.image_count;
        self.vx.strtexs[handle.0].colbuffer[handle.1].copy_from_slice(&rgba);
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

    /// Read pixels from arbitrary coordinates
    pub fn read(&mut self, id: &Layer, mut map: impl FnMut(&[(u8, u8, u8, u8)], usize)) {
        let s = &mut *self.vx;
        if let Some(ref strtex) = s.strtexs.get(id.0) {
            unsafe {
                let subres = s.device.get_image_subresource_footprint(
                    &*strtex.image_buffer,
                    gfx_hal::image::Subresource {
                        aspects: gfx_hal::format::Aspects::COLOR,
                        level: 0,
                        layer: 0,
                    },
                );

                let target = s
                    .device
                    .acquire_mapping_reader::<(u8, u8, u8, u8)>(
                        &*strtex.image_memory,
                        0..strtex.image_requirements.size,
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
            unsafe {
                let subres = s.device.get_image_subresource_footprint(
                    &*strtex.image_buffer,
                    gfx_hal::image::Subresource {
                        aspects: gfx_hal::format::Aspects::COLOR,
                        level: 0,
                        layer: 0,
                    },
                );

                let mut target = s
                    .device
                    .acquire_mapping_writer::<(u8, u8, u8, u8)>(
                        &*strtex.image_memory,
                        0..strtex.image_requirements.size,
                    )
                    .expect("unable to acquire mapping writer");

                map(&mut target, (subres.row_pitch / 4) as usize);

                s.device
                    .release_mapping_writer(target)
                    .expect("Unable to release mapping writer");
            }
        }
    }

    /// Fills the streaming texture with perlin noise generated from an input seed
    pub fn fill_with_perlin_noise(&mut self, blitid: &Layer, seed: [f32; 3]) {
        let s = &mut *self.vx;
        static VERTEX_SOURCE: &[u8] = include_bytes!("../_build/spirv/proc1.vert.spirv");
        static FRAGMENT_SOURCE: &[u8] = include_bytes!("../_build/spirv/proc1.frag.spirv");
        let w = s.strtexs[blitid.0].width;
        let h = s.strtexs[blitid.0].height;
        let vs_module = { unsafe { s.device.create_shader_module(&VERTEX_SOURCE) }.unwrap() };
        let fs_module = { unsafe { s.device.create_shader_module(&FRAGMENT_SOURCE) }.unwrap() };
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
                    image::Tiling::Linear,
                    image::Usage::COLOR_ATTACHMENT
                        | image::Usage::TRANSFER_SRC
                        | image::Usage::SAMPLED,
                    image::ViewCapabilities::empty(),
                )
                .expect("Unable to create image");
            let requirements = s.device.get_image_requirements(&image);
            let memory_type_id =
                find_memory_type_id(&s.adapter, requirements, memory::Properties::CPU_VISIBLE);
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
            let clear_values = [command::ClearValue::Color(command::ClearColor::Float([
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
            s.queue_group.queues[0].submit_nosemaphores(Some(&cmd_buffer), Some(&upload_fence));
            s.device
                .wait_for_fence(&upload_fence, u64::max_value())
                .expect("Unable to wait for fence");
            s.device
                .reset_fence(&upload_fence)
                .expect("Unable to wait for fence");

            cmd_buffer.begin();
            cmd_buffer.blit_image(
                &image,
                image::Layout::General,
                &s.strtexs[blitid.0].image_buffer,
                image::Layout::General,
                image::Filter::Nearest,
                once(command::ImageBlit {
                    src_subresource: image::SubresourceLayers {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    src_bounds: image::Offset { x: 0, y: 0, z: 0 }..image::Offset {
                        x: w as i32,
                        y: w as i32,
                        z: 1,
                    },
                    dst_subresource: image::SubresourceLayers {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    dst_bounds: image::Offset { x: 0, y: 0, z: 0 }..image::Offset {
                        x: w as i32,
                        y: h as i32,
                        z: 1,
                    },
                }),
            );
            cmd_buffer.finish();
            s.queue_group.queues[0].submit_nosemaphores(Some(&cmd_buffer), Some(&upload_fence));
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
    use logger::{Generic, GenericLogger, Logger};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg as random;
    use test::{black_box, Bencher};

    #[test]
    fn generate_map_randomly() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(LayerOptions::new().width(1000).height(1000));
        strtex.add(&id, Sprite::default());
        strtex.fill_with_perlin_noise(&id, [0.0, 0.0, 0.0]);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "generate_map_randomly", img);
    }

    #[test]
    fn with_origin_11() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(LayerOptions::new().width(1000).height(1000));
        strtex.add(&id, Sprite::default().origin((1.0, 1.0)));
        strtex.fill_with_perlin_noise(&id, [0.0, 0.0, 0.0]);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "with_origin_11", img);
    }

    #[test]
    fn streaming_texture_blocks() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut strtex = vx.strtex();

        let id = strtex.add_layer(LayerOptions::new().width(1000).height(1000));
        strtex.add(&id, strtex::Sprite::default());

        strtex.set_pixels_block(&id, (0, 0), (500, 500), (255, 0, 0, 255));
        strtex.set_pixels_block(&id, (500, 0), (500, 500), (0, 255, 0, 255));
        strtex.set_pixels_block(&id, (0, 500), (500, 500), (0, 0, 255, 255));
        strtex.set_pixels_block(&id, (500, 500), (500, 500), (0, 0, 0, 0));

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "streaming_texture_blocks", img);
    }

    #[test]
    fn streaming_texture_blocks_off_by_one() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(LayerOptions::new().width(10).height(1));
        strtex.add(&id, strtex::Sprite::default());

        strtex.set_pixels_block(&id, (0, 0), (10, 1), (0, 255, 0, 255));

        strtex.set_pixels_block(&id, (3, 0), (1, 1), (0, 0, 255, 255));

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "streaming_texture_blocks_off_by_one", img);

        let mut strtex = vx.strtex();
        strtex.set_pixels_block(&id, (3, 0), (0, 1), (255, 0, 255, 255));

        strtex.set_pixels_block(&id, (3, 0), (0, 0), (255, 0, 255, 255));

        strtex.set_pixels_block(&id, (3, 0), (1, 0), (255, 0, 255, 255));

        strtex.set_pixels_block(&id, (30, 0), (800, 0), (255, 0, 255, 255));

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "streaming_texture_blocks_off_by_one", img);
    }

    #[test]
    fn use_read() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(LayerOptions::new().width(10).height(10));
        strtex.set_pixel(&id, 3, 2, (0, 123, 0, 255));
        let mut green_value = 0;
        strtex.read(&id, |arr, pitch| {
            green_value = arr[3 + 2 * pitch].1;
        });
        assert_eq![123, green_value];
    }

    #[test]
    fn use_write() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(LayerOptions::new().width(10).height(10));
        strtex.set_pixel(&id, 3, 2, (0, 123, 0, 255));
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

        let id = strtex.add_layer(LayerOptions::new().width(20).height(20));
        strtex.add(&id, strtex::Sprite::default());

        let mut rng = random::new(0);

        for _ in 0..1000 {
            let x = rng.gen_range(0, 30);
            let y = rng.gen_range(0, 30);

            strtex.set_pixel(&id, x, y, (0, 255, 0, 255));
            strtex.set_pixels(&id, once((x, y, (0, 255, 0, 255))));
        }
    }

    #[test]
    fn streaming_texture_weird_block_accesses() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut strtex = vx.strtex();
        let id = strtex.add_layer(LayerOptions::new().width(64).height(64));
        strtex.add(&id, strtex::Sprite::default());

        let mut rng = random::new(0);

        for _ in 0..1000 {
            let start = (rng.gen_range(0, 100), rng.gen_range(0, 100));
            let wh = (rng.gen_range(0, 100), rng.gen_range(0, 100));

            strtex.set_pixels_block(&id, start, wh, (0, 255, 0, 255));
        }
    }

    // #[test]
    // fn streaming_texture_respects_z_ordering() {
    //     let logger = Logger::<Generic>::spawn_void().to_compatibility();
    //     let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
    //     let prspect = gen_perspective(&vx);

    //     let mut strtex = vx.strtex();
    //     let strtex1 = strtex.add_layer(LayerOptions::new().width(10).height(10));
    //     strtex.set_pixels_block(&strtex1, (0, 0), (9, 9), (255, 255, 0, 255));
    //     strtex.add(&strtex1, strtex::Sprite::default());

    //     let strtex2 = strtex.add_layer(LayerOptions::new().width(10).height(10));
    //     strtex.set_pixels_block(&strtex2, (1, 1), (9, 9), (0, 255, 255, 255));
    //     strtex.add(
    //         &strtex2,
    //         strtex::Sprite {
    //             depth: 0.1,
    //             ..strtex::Sprite::default()
    //         },
    //     );

    //     let img = vx.draw_frame_copy_framebuffer(&prspect);
    //     utils::assert_swapchain_eq(&mut vx, "streaming_texture_z_ordering", img);

    //     vx.strtex().remove_layer(strtex1);

    //     let img = vx.draw_frame_copy_framebuffer(&prspect);
    //     utils::assert_swapchain_eq(&mut vx, "streaming_texture_z_ordering_removed", img);
    // }

    // ---

    #[bench]
    fn bench_streaming_texture_set_single_pixel_while_drawing(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let id = vx
            .strtex()
            .add_layer(LayerOptions::new().width(50).height(50));
        vx.strtex().add(&id, strtex::Sprite::default());

        b.iter(|| {
            vx.strtex()
                .set_pixel(&id, black_box(1), black_box(2), (255, 0, 0, 255));
            vx.draw_frame(&prspect);
        });
    }

    #[bench]
    fn bench_streaming_texture_set_500x500_area(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let id = vx
            .strtex()
            .add_layer(LayerOptions::new().width(1000).height(1000));
        vx.strtex().add(&id, strtex::Sprite::default());

        b.iter(|| {
            vx.strtex()
                .set_pixels_block(&id, (0, 0), (500, 500), (255, 0, 0, 255));
        });
    }

    #[bench]
    fn bench_streaming_texture_set_500x500_area_using_iterator(b: &mut Bencher) {
        use itertools::Itertools;
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let id = vx
            .strtex()
            .add_layer(LayerOptions::new().width(1000).height(1000));
        vx.strtex().add(&id, strtex::Sprite::default());

        b.iter(|| {
            vx.strtex().set_pixels(
                &id,
                (0..500)
                    .cartesian_product(0..500)
                    .map(|(x, y)| (x, y, (255, 0, 0, 255))),
            );
        });
    }

    #[bench]
    fn bench_streaming_texture_set_single_pixel(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let id = vx
            .strtex()
            .add_layer(LayerOptions::new().width(1000).height(1000));
        vx.strtex().add(&id, strtex::Sprite::default());

        b.iter(|| {
            vx.strtex()
                .set_pixel(&id, black_box(1), black_box(2), (255, 0, 0, 255));
        });
    }
}
