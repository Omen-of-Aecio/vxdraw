//! Methods and types to control dynamic textures
//!
//! A dynamic texture is a texture from which you can spawn sprites. The `dynamic` part of the name
//! refers to the sprites. A sprite is a rectangular view into the texture. The sprites can be
//! changed freely during runtime. This allows movement of sprites, animations, and warping of
//! their form.
//! # Example - Drawing a sprite #
//! ```
//! use cgmath::{prelude::*, Deg, Matrix4};
//! use vxdraw::{dyntex::{ImgData, LayerOptions, Sprite}, void_logger, utils::gen_perspective, ShowWindow, VxDraw};
//! fn main() {
//!     static TESTURE: &ImgData = &ImgData::PNGBytes(include_bytes!["../images/testure.png"]);
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Headless1k); // Change this to ShowWindow::Enable to show the window
//!
//!
//!     let mut dyntex = vx.dyntex();
//!     let tex = dyntex.add_layer(TESTURE, &LayerOptions::new());
//!     vx.dyntex().add(&tex, Sprite::new().scale(0.5));
//!
//!     let prspect = gen_perspective(&vx);
//!     vx.draw_frame(&prspect);
//!     #[cfg(not(test))]
//!     std::thread::sleep(std::time::Duration::new(3, 0));
//! }
//! ```
use super::{blender, utils::*, Color};
use crate::data::{DrawType, DynamicTexture, VxDraw};
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
use std::mem::ManuallyDrop;

// ---

/// A view into a texture (a sprite)
pub struct Handle(usize, usize);

/// Handle to a layer (a single texture)
pub struct Layer(usize);

impl Layerable for Layer {
    fn get_layer(&self, vx: &VxDraw) -> usize {
        for (idx, ord) in vx.draw_order.iter().enumerate() {
            match ord {
                DrawType::DynamicTexture { id } if *id == self.0 => {
                    return idx;
                }
                _ => {}
            }
        }
        panic!["Unable to get layer"]
    }
}

/// Options for creating a layer of a dynamic texture with sprites
#[derive(Clone)]
pub struct LayerOptions {
    /// Perform depth testing (and fragment culling) when drawing sprites from this texture
    depth_test: bool,
    /// Fix the perspective, this ignores the perspective sent into draw for this texture and
    /// all its associated sprites
    fixed_perspective: Option<Matrix4<f32>>,
    /// Specify filtering mode for sampling the texture (default is [Filter::Nearest])
    filtering: Filter,
    /// Specify wrap mode for texture sampling
    wrap_mode: WrapMode,
    /// Blending mode for this layer
    blend: blender::Blender,
}

impl LayerOptions {
    /// Same as default
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the sampling filter mode for the texture
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filtering = filter;
        self
    }

    /// Enable/disable depth testing
    pub fn depth(mut self, depth: bool) -> Self {
        self.depth_test = depth;
        self
    }

    /// Set a fixed perspective for this layer
    ///
    /// The layer will ignore global perspective matrices if this is given.
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
            depth_test: true,
            fixed_perspective: None,
            filtering: Filter::Nearest,
            wrap_mode: WrapMode::Tile,
            blend: blender::Blender::default(),
        }
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

/// Sprite creation builder
///
/// A sprite is a rectangular view into a texture. This structure sets up the necessary data to
/// call [Dyntex::add] with.
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

    /// Set the opacity of the sprite
    ///
    /// The opacity per fragment is multiplied with the sprite's opacity value
    pub fn opacity(mut self, opacity: u8) -> Self {
        self.opacity = [opacity; 4];
        self
    }

    /// Set the opacity of the sprite
    ///
    /// The opacity per fragment is multiplied with the sprite's opacity value
    pub fn opacity_raw(mut self, opacity: [u8; 4]) -> Self {
        self.opacity = opacity;
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

/// Specify the type of incoming texture data
pub enum ImgData<'a> {
    /// Raw PNG bytes, no size is needed as this is included in the bytestream
    PNGBytes(&'a [u8]),
    /// Raw RGBA8 bytes
    RawBytes {
        /// Width of the image in pixels
        width: usize,
        /// Height of the image in pixels
        height: usize,
        /// Bytes of the image, if there is a width/height mismatch, the image will be truncated or
        /// tiled (extended)
        bytes: &'a [u8],
    },
}

// ---

/// Accessor object to all dynamic textures
///
/// A dynamic texture is a texture which is used to display textured sprites.
/// See [crate::dyntex] for examples.
pub struct Dyntex<'a> {
    vx: &'a mut VxDraw,
}

impl<'a> Dyntex<'a> {
    /// Prepare to edit dynamic textures
    ///
    /// You're not supposed to use this function directly (although you can).
    /// The recommended way of spawning a dyntex is via [VxDraw::dyntex()].
    pub fn new(s: &'a mut VxDraw) -> Self {
        Self { vx: s }
    }

    /// Add a texture (layer) to the system
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
    pub fn add_layer<'x>(&mut self, img_data: &ImgData<'x>, options: &LayerOptions) -> Layer {
        match img_data {
            ImgData::PNGBytes(ref bytes) => {
                let image = load_image::load_from_memory_with_format(&bytes[..], load_image::PNG)
                    .unwrap()
                    .to_rgba();
                let (width, height) = (image.width() as usize, image.height() as usize);
                let img_bytes = image.into_raw();
                self.add_layer_internal(width, height, &img_bytes[..], &options)
            }
            ImgData::RawBytes {
                width,
                height,
                bytes,
            } => self.add_layer_internal(*width, *height, bytes, options),
        }
    }

    fn add_layer_internal(
        &mut self,
        img_width: usize,
        img_height: usize,
        img: &[u8],
        options: &LayerOptions,
    ) -> Layer {
        let s = &mut *self.vx;
        let device = &s.device;

        let pixel_size = 4; //size_of::<image::Rgba<u8>>();
        let row_size = pixel_size * img_width;
        let limits = s.adapter.physical_device.limits();
        let row_alignment_mask = limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let row_pitch = ((row_size as u32 + row_alignment_mask) & !row_alignment_mask) as usize;
        debug_assert!(row_pitch as usize >= row_size);
        let required_bytes = row_pitch * img_height;

        let mut image_upload_buffer = unsafe {
            device.create_buffer(required_bytes as u64, gfx_hal::buffer::Usage::TRANSFER_SRC)
        }
        .unwrap();
        let image_mem_reqs = unsafe { device.get_buffer_requirements(&image_upload_buffer) };
        let memory_type_id = find_memory_type_id(
            &s.adapter,
            image_mem_reqs,
            Properties::CPU_VISIBLE | Properties::COHERENT,
        );
        let image_upload_memory =
            unsafe { device.allocate_memory(memory_type_id, image_mem_reqs.size) }.unwrap();
        unsafe { device.bind_buffer_memory(&image_upload_memory, 0, &mut image_upload_buffer) }
            .unwrap();

        unsafe {
            let mut writer = s
                .device
                .acquire_mapping_writer::<u8>(&image_upload_memory, 0..image_mem_reqs.size)
                .expect("Unable to get mapping writer");
            let mut idx = 0;
            for y in 0..img_height {
                // let row = &(*img)[y * row_size..(y + 1) * row_size];
                let dest_base = y * row_pitch;
                for row_index in 0..row_size {
                    writer[dest_base + row_index] = img[idx % img.len()];
                    idx += 1;
                }
            }
            device
                .release_mapping_writer(writer)
                .expect("Couldn't release the mapping writer to the staging buffer!");
        }

        let mut the_image = unsafe {
            device
                .create_image(
                    image::Kind::D2(img_width as u32, img_height as u32, 1, 1),
                    1,
                    format::Format::Rgba8Srgb,
                    image::Tiling::Optimal,
                    image::Usage::TRANSFER_DST | image::Usage::SAMPLED,
                    image::ViewCapabilities::empty(),
                )
                .expect("Couldn't create the image!")
        };

        let image_memory = unsafe {
            let requirements = device.get_image_requirements(&the_image);
            let memory_type_id =
                find_memory_type_id(&s.adapter, requirements, memory::Properties::DEVICE_LOCAL);
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

        unsafe {
            let mut cmd_buffer = s.command_pool.acquire_command_buffer::<command::OneShot>();
            cmd_buffer.begin();
            let image_barrier = memory::Barrier::Image {
                states: (image::Access::empty(), image::Layout::Undefined)
                    ..(
                        image::Access::TRANSFER_WRITE,
                        image::Layout::TransferDstOptimal,
                    ),
                target: &the_image,
                families: None,
                range: image::SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };
            cmd_buffer.pipeline_barrier(
                pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::TRANSFER,
                memory::Dependencies::empty(),
                &[image_barrier],
            );
            cmd_buffer.copy_buffer_to_image(
                &image_upload_buffer,
                &the_image,
                image::Layout::TransferDstOptimal,
                &[command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: (row_pitch / pixel_size) as u32,
                    buffer_height: img_height as u32,
                    image_layers: gfx_hal::image::SubresourceLayers {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: image::Offset { x: 0, y: 0, z: 0 },
                    image_extent: image::Extent {
                        width: img_width as u32,
                        height: img_height as u32,
                        depth: 1,
                    },
                }],
            );
            let image_barrier = memory::Barrier::Image {
                states: (
                    image::Access::TRANSFER_WRITE,
                    image::Layout::TransferDstOptimal,
                )
                    ..(
                        image::Access::SHADER_READ,
                        image::Layout::ShaderReadOnlyOptimal,
                    ),
                target: &the_image,
                families: None,
                range: image::SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };
            cmd_buffer.pipeline_barrier(
                pso::PipelineStage::TRANSFER..pso::PipelineStage::FRAGMENT_SHADER,
                memory::Dependencies::empty(),
                &[image_barrier],
            );
            cmd_buffer.finish();
            let upload_fence = s
                .device
                .create_fence(false)
                .expect("Couldn't create an upload fence!");
            s.queue_group.queues[0].submit_nosemaphores(Some(&cmd_buffer), Some(&upload_fence));
            s.device
                .wait_for_fence(&upload_fence, u64::max_value())
                .expect("Couldn't wait for the fence!");
            s.device.destroy_fence(upload_fence);
        }

        unsafe {
            device.destroy_buffer(image_upload_buffer);
            device.free_memory(image_upload_memory);
        }

        const VERTEX_SOURCE_TEXTURE: &[u8] = include_bytes!["../_build/spirv/dyntex.vert.spirv"];

        const FRAGMENT_SOURCE_TEXTURE: &[u8] = include_bytes!["../_build/spirv/dyntex.frag.spirv"];

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
                pso::DepthTest::On {
                    fun: pso::Comparison::LessEqual,
                    write: true,
                }
            } else {
                pso::DepthTest::Off
            },
            depth_bounds: false,
            stencil: pso::StencilTest::Off,
        };
        let blender = options.blend.clone().to_gfx_blender();
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
                    descriptors: Some(pso::Descriptor::Image(
                        &image_view,
                        image::Layout::ShaderReadOnlyOptimal,
                    )),
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

        s.dyntexs.push(DynamicTexture {
            hidden: false,

            fixed_perspective: options.fixed_perspective,
            removed: vec![],

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

            texture_image_buffer: ManuallyDrop::new(the_image),
            texture_image_memory: ManuallyDrop::new(image_memory),

            descriptor_pool: ManuallyDrop::new(descriptor_pool),
            image_view: ManuallyDrop::new(image_view),
            sampler: ManuallyDrop::new(sampler),

            descriptor_set: ManuallyDrop::new(descriptor_set),
            descriptor_set_layouts: triangle_descriptor_set_layouts,
            pipeline: ManuallyDrop::new(triangle_pipeline),
            pipeline_layout: ManuallyDrop::new(triangle_pipeline_layout),
            render_pass: ManuallyDrop::new(triangle_render_pass),
        });
        s.draw_order.push(DrawType::DynamicTexture {
            id: s.dyntexs.len() - 1,
        });
        Layer(s.dyntexs.len() - 1)
    }

    /// Disable drawing of the sprites at this layer
    pub fn hide(&mut self, layer: &Layer) {
        self.vx.dyntexs[layer.0].hidden = true;
    }

    /// Enable drawing of the sprites at this layer
    pub fn show(&mut self, layer: &Layer) {
        self.vx.dyntexs[layer.0].hidden = false;
    }

    /// Remove a layer
    ///
    /// Removes the layer from memory and destroys all sprites associated with it.
    /// All lingering sprite handles that were spawned using this layer handle will be
    /// invalidated.
    pub fn remove_layer(&mut self, layer: Layer) {
        let s = &mut *self.vx;
        let mut index = None;
        for (idx, x) in s.draw_order.iter().enumerate() {
            match x {
                DrawType::DynamicTexture { id } if *id == layer.0 => {
                    index = Some(idx);
                    break;
                }
                _ => {}
            }
        }
        if let Some(idx) = index {
            s.draw_order.remove(idx);
            // Can't delete here always because other textures may still be referring to later dyntexs,
            // only when this is the last layer.
            if s.dyntexs.len() == layer.0 + 1 {
                let dyntex = s.dyntexs.pop().unwrap();
                destroy_texture(s, dyntex);
            }
        }
    }

    /// Add a sprite (a rectangular view of a texture) to the system
    ///
    /// The sprite is automatically drawn on each [VxDraw::draw_frame] call, and must be removed by
    /// [Dyntex::remove] to stop it from being drawn.
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

        let replace = self.vx.dyntexs.get(layer.0).map(|x| !x.removed.is_empty());
        if replace.is_none() {
            panic!["Layer does not exist"];
        }

        let handle = if replace.unwrap() {
            let hole = self.vx.dyntexs[layer.0].removed.pop().unwrap();
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
            let tex = &mut self.vx.dyntexs[layer.0];
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

        let tex = &mut self.vx.dyntexs[layer.0];
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
    /// `holes`. Calling [Dyntex::add] with available holes will fill the first available hole
    /// with the new triangle.
    pub fn remove(&mut self, handle: Handle) {
        self.vx.dyntexs[handle.0].scalebuf_touch = self.vx.swapconfig.image_count;
        if let Some(dyntex) = self.vx.dyntexs.get_mut(handle.0) {
            dyntex.removed.push(handle.1);
            dyntex.scalebuffer[handle.1].copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        }
    }

    // ---

    /// Change the vertices of the model-space
    ///
    /// The name `set_deform` is used to keep consistent [Dyntex::deform].
    /// What this function does is just setting absolute vertex positions for each vertex in the
    /// triangle.
    pub fn set_deform(&mut self, handle: &Handle, points: [(f32, f32); 4]) {
        self.vx.dyntexs[handle.0].posbuf_touch = self.vx.swapconfig.image_count;
        let vertex = &mut self.vx.dyntexs[handle.0].posbuffer[handle.1];
        for (idx, point) in points.iter().enumerate() {
            vertex[idx * 2] = point.0;
            vertex[idx * 2 + 1] = point.1;
        }
    }

    /// Set a solid color of a quad
    pub fn set_solid_color(&mut self, handle: &Handle, rgba: Color) {
        self.vx.dyntexs[handle.0].opacbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            let Color::Rgba(r, g, b, a) = rgba;
            self.vx.dyntexs[handle.0].opacbuffer[handle.1][idx * 4..(idx + 1) * 4]
                .copy_from_slice(&[r, g, b, a]);
        }
    }

    /// Set an opacity each vertex of a sprite
    pub fn set_opacity(&mut self, handle: &Handle, opacity: u8) {
        self.vx.dyntexs[handle.0].opacbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            self.vx.dyntexs[handle.0].opacbuffer[handle.1].copy_from_slice(&[opacity; 4]);
        }
    }

    /// Set an opacity each vertex of a sprite
    pub fn set_opacity_raw(&mut self, handle: &Handle, opacity: [u8; 4]) {
        self.vx.dyntexs[handle.0].opacbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            self.vx.dyntexs[handle.0].opacbuffer[handle.1].copy_from_slice(&opacity);
        }
    }

    /// Set the position of a sprite
    pub fn set_translation(&mut self, handle: &Handle, position: (f32, f32)) {
        self.vx.dyntexs[handle.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            self.vx.dyntexs[handle.0].tranbuffer[handle.1][idx * 2] = position.0;
            self.vx.dyntexs[handle.0].tranbuffer[handle.1][idx * 2 + 1] = position.1;
        }
    }

    /// Set the rotation of a sprite
    ///
    /// Positive rotation goes counter-clockwise. The value of the rotation is in radians.
    pub fn set_rotation<T: Copy + Into<Rad<f32>>>(&mut self, handle: &Handle, angle: T) {
        let angle = angle.into().0;
        self.vx.dyntexs[handle.0].rotbuf_touch = self.vx.swapconfig.image_count;
        self.vx.dyntexs[handle.0].rotbuffer[handle.1]
            .copy_from_slice(&[angle, angle, angle, angle]);
    }

    /// Set the scale of a sprite
    pub fn set_scale(&mut self, handle: &Handle, scale: f32) {
        self.vx.dyntexs[handle.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for sc in &mut self.vx.dyntexs[handle.0].scalebuffer[handle.1] {
            *sc = scale;
        }
    }

    /// Set the UV values of a single sprite
    pub fn set_uv(&mut self, handle: &Handle, uv_begin: (f32, f32), uv_end: (f32, f32)) {
        self.vx.dyntexs[handle.0].uvbuf_touch = self.vx.swapconfig.image_count;
        self.vx.dyntexs[handle.0].uvbuffer[handle.1].copy_from_slice(&[
            uv_begin.0, uv_begin.1, uv_begin.0, uv_end.1, uv_end.0, uv_end.1, uv_end.0, uv_begin.1,
        ]);
    }

    /// Set the raw UV values of each vertex in a sprite
    ///
    /// This may be used to repeat a texture multiple times over the same sprite, or to do
    /// something exotic with uv coordinates.
    pub fn set_uv_raw(&mut self, handle: &Handle, uvs: [(f32, f32); 4]) {
        self.vx.dyntexs[handle.0].uvbuf_touch = self.vx.swapconfig.image_count;
        self.vx.dyntexs[handle.0].uvbuffer[handle.1].copy_from_slice(&[
            uvs[0].0, uvs[0].1, uvs[1].0, uvs[1].1, uvs[2].0, uvs[2].1, uvs[3].0, uvs[3].1,
        ]);
    }

    // ---

    /// Deform a sprite by adding delta vertices
    ///
    /// Adds the delta vertices to the sprite. Beware: This changes model space form.
    pub fn deform(&mut self, handle: &Handle, delta: [(f32, f32); 4]) {
        self.vx.dyntexs[handle.0].posbuf_touch = self.vx.swapconfig.image_count;
        let points = &mut self.vx.dyntexs[handle.0].posbuffer[handle.1];
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
        self.vx.dyntexs[handle.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            self.vx.dyntexs[handle.0].tranbuffer[handle.1][idx * 2] += movement.0;
            self.vx.dyntexs[handle.0].tranbuffer[handle.1][idx * 2 + 1] += movement.1;
        }
    }

    /// Rotate a sprite
    ///
    /// Rotation does not mutate the model-space of a sprite.
    pub fn rotate<T: Copy + Into<Rad<f32>>>(&mut self, handle: &Handle, angle: T) {
        self.vx.dyntexs[handle.0].rotbuf_touch = self.vx.swapconfig.image_count;
        for rot in &mut self.vx.dyntexs[handle.0].rotbuffer[handle.1] {
            *rot += angle.into().0;
        }
    }

    /// Scale a sprite
    ///
    /// Scale does not mutate the model-space of a sprite.
    pub fn scale(&mut self, handle: &Handle, scale: f32) {
        self.vx.dyntexs[handle.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for sc in &mut self.vx.dyntexs[handle.0].scalebuffer[handle.1] {
            *sc *= scale;
        }
    }

    // ---

    /// Deform all dyntexs by adding delta vertices
    ///
    /// Applies [Dyntex::deform] to each dynamic texture.
    pub fn deform_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> [(f32, f32); 4]) {
        self.vx.dyntexs[layer.0].posbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.dyntexs[layer.0].posbuffer.iter_mut().enumerate() {
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

    /// Translate all dyntexs by adding delta vertices
    ///
    /// Applies [Dyntex::translate] to each dynamic texture.
    pub fn translate_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> (f32, f32)) {
        self.vx.dyntexs[layer.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.dyntexs[layer.0].tranbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for idx in 0..4 {
                quad[idx * 2] += delta.0;
                quad[idx * 2 + 1] += delta.1;
            }
        }
    }

    /// Rotate all dyntexs by adding delta rotations
    ///
    /// Applies [Dyntex::rotate] to each dynamic texture.
    pub fn rotate_all<T: Copy + Into<Rad<f32>>>(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> T,
    ) {
        self.vx.dyntexs[layer.0].rotbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.dyntexs[layer.0].rotbuffer.iter_mut().enumerate() {
            let delta = delta(idx).into().0;
            for idx in 0..4 {
                quad[idx] += delta;
            }
        }
    }

    /// Scale all dyntexs by multiplying a delta scale
    ///
    /// Applies [Dyntex::scale] to each dynamic texture.
    pub fn scale_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> f32) {
        self.vx.dyntexs[layer.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.dyntexs[layer.0].scalebuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for idx in 0..4 {
                quad[idx] *= delta;
            }
        }
    }

    // ---

    /// Deform all dyntexs by setting delta vertices
    ///
    /// Applies [Dyntex::set_deform] to each dynamic texture.
    pub fn set_deform_all(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> [(f32, f32); 4],
    ) {
        self.vx.dyntexs[layer.0].posbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.dyntexs[layer.0].posbuffer.iter_mut().enumerate() {
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

    /// Set the color on all dyntexs
    ///
    /// Applies [Dyntex::set_solid_color] to each dynamic texture.
    pub fn set_solid_color_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> Color) {
        self.vx.dyntexs[layer.0].opacbuf_touch = self.vx.swapconfig.image_count;
        for (idx, dyntex) in self.vx.dyntexs[layer.0].opacbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for idx in 0..4 {
                let Color::Rgba(r, g, b, a) = delta;
                dyntex[idx * 4..(idx + 1) * 4].copy_from_slice(&[r, g, b, a]);
            }
        }
    }

    /// Set the color on all dyntexs
    ///
    /// Applies [Dyntex::set_color] to each dynamic texture.
    pub fn set_color_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> [Color; 4]) {
        self.vx.dyntexs[layer.0].opacbuf_touch = self.vx.swapconfig.image_count;
        for (idx, dyntex) in self.vx.dyntexs[layer.0].opacbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for (idx, dt) in delta.iter().enumerate() {
                let Color::Rgba(r, g, b, a) = dt;
                dyntex[idx * 4..(idx + 1) * 4].copy_from_slice(&[*r, *g, *b, *a]);
            }
        }
    }

    /// Set the translation on all dyntexs
    ///
    /// Applies [Dyntex::set_translation] to each dynamic texture.
    pub fn set_translation_all(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> (f32, f32),
    ) {
        self.vx.dyntexs[layer.0].tranbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.dyntexs[layer.0].tranbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for idx in 0..4 {
                quad[idx * 2] = delta.0;
                quad[idx * 2 + 1] = delta.1;
            }
        }
    }

    /// Set the uv on all dyntexs
    ///
    /// Applies [Dyntex::set_uv] to each dynamic texture.
    pub fn set_uv_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> [(f32, f32); 2]) {
        self.vx.dyntexs[layer.0].uvbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.dyntexs[layer.0].uvbuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            let uv_begin = delta[0];
            let uv_end = delta[1];
            quad.copy_from_slice(&[
                uv_begin.0, uv_begin.1, uv_begin.0, uv_end.1, uv_end.0, uv_end.1, uv_end.0,
                uv_begin.1,
            ]);
        }
    }

    /// Set the rotation on all dyntexs
    ///
    /// Applies [Dyntex::set_rotation] to each dynamic texture.
    pub fn set_rotation_all<T: Copy + Into<Rad<f32>>>(
        &mut self,
        layer: &Layer,
        mut delta: impl FnMut(usize) -> T,
    ) {
        self.vx.dyntexs[layer.0].rotbuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.dyntexs[layer.0].rotbuffer.iter_mut().enumerate() {
            let delta = delta(idx).into().0;
            for idx in 0..4 {
                quad[idx] = delta;
            }
        }
    }

    /// Set the scale on all dyntexs
    ///
    /// Applies [Dyntex::set_scale] to each dynamic texture.
    /// Note: This may re-enable removed sprites, see [Dyntex::remove].
    pub fn set_scale_all(&mut self, layer: &Layer, mut delta: impl FnMut(usize) -> f32) {
        self.vx.dyntexs[layer.0].scalebuf_touch = self.vx.swapconfig.image_count;
        for (idx, quad) in self.vx.dyntexs[layer.0].scalebuffer.iter_mut().enumerate() {
            let delta = delta(idx);
            for idx in 0..4 {
                quad[idx] = delta;
            }
        }
    }

    // ---

    /// Set the UV values of multiple sprites
    pub fn set_uvs<'b>(
        &mut self,
        mut uvs: impl Iterator<Item = (&'b Handle, (f32, f32), (f32, f32))>,
    ) {
        if let Some(first) = uvs.next() {
            let current_texture_handle = first.0;
            self.set_uv(current_texture_handle, first.1, first.2);
            for handle in uvs {
                self.set_uv(handle.0, handle.1, handle.2);
            }
        }
    }
}

// ---

fn destroy_texture(s: &mut VxDraw, mut dyntex: DynamicTexture) {
    unsafe {
        for mut indices in dyntex.indices.drain(..) {
            indices.destroy(&s.device);
        }
        for mut posbuf in dyntex.posbuf.drain(..) {
            posbuf.destroy(&s.device);
        }
        for mut opacbuf in dyntex.opacbuf.drain(..) {
            opacbuf.destroy(&s.device);
        }
        for mut uvbuf in dyntex.uvbuf.drain(..) {
            uvbuf.destroy(&s.device);
        }
        for mut tranbuf in dyntex.tranbuf.drain(..) {
            tranbuf.destroy(&s.device);
        }
        for mut rotbuf in dyntex.rotbuf.drain(..) {
            rotbuf.destroy(&s.device);
        }
        for mut scalebuf in dyntex.scalebuf.drain(..) {
            scalebuf.destroy(&s.device);
        }
        s.device
            .destroy_image(ManuallyDrop::into_inner(read(&dyntex.texture_image_buffer)));
        s.device
            .free_memory(ManuallyDrop::into_inner(read(&dyntex.texture_image_memory)));
        s.device
            .destroy_render_pass(ManuallyDrop::into_inner(read(&dyntex.render_pass)));
        s.device
            .destroy_pipeline_layout(ManuallyDrop::into_inner(read(&dyntex.pipeline_layout)));
        s.device
            .destroy_graphics_pipeline(ManuallyDrop::into_inner(read(&dyntex.pipeline)));
        for dsl in dyntex.descriptor_set_layouts.drain(..) {
            s.device.destroy_descriptor_set_layout(dsl);
        }
        s.device
            .destroy_descriptor_pool(ManuallyDrop::into_inner(read(&dyntex.descriptor_pool)));
        s.device
            .destroy_sampler(ManuallyDrop::into_inner(read(&dyntex.sampler)));
        s.device
            .destroy_image_view(ManuallyDrop::into_inner(read(&dyntex.image_view)));
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
    use std::f32::consts::PI;
    use test::Bencher;

    // ---

    static LOGO: &ImgData = &ImgData::PNGBytes(include_bytes!["../images/logo.png"]);
    static FOREST: &ImgData = &ImgData::PNGBytes(include_bytes!["../images/forest-light.png"]);
    static TESTURE: &ImgData = &ImgData::PNGBytes(include_bytes!["../images/testure.png"]);
    static TREE: &ImgData = &ImgData::PNGBytes(include_bytes!["../images/treetest.png"]);
    static FIREBALL: &ImgData = &ImgData::PNGBytes(include_bytes!["../images/Fireball_68x9.png"]);

    // ---

    #[test]
    fn simple_texture() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut dyntex = vx.dyntex();
        let tex = dyntex.add_layer(LOGO, &&LayerOptions::new());
        vx.dyntex().add(&tex, Sprite::new());

        let prspect = gen_perspective(&vx);
        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "simple_texture", img);
    }

    #[test]
    fn simple_texture_adheres_to_view() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless2x1k);
        let tex = vx.dyntex().add_layer(LOGO, &LayerOptions::new());
        vx.dyntex().add(&tex, Sprite::new());

        let prspect = gen_perspective(&vx);
        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "simple_texture_adheres_to_view", img);
    }

    #[test]
    fn colored_simple_texture1() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let tex = vx.dyntex().add_layer(LOGO, &LayerOptions::new());
        vx.dyntex().add(&tex, Sprite::new().opacity(100));

        let prspect = gen_perspective(&vx);
        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "colored_simple_texture", img);
    }

    #[test]
    fn colored_simple_texture_set_position() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let mut dyntex = vx.dyntex();
        let tex = dyntex.add_layer(LOGO, &LayerOptions::new());
        let sprite = dyntex.add(&tex, Sprite::new().opacity(100));
        dyntex.set_translation(&sprite, (0.5, 0.3));

        let prspect = gen_perspective(&vx);
        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "colored_simple_texture_set_position", img);
    }

    #[test]
    fn translated_texture() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let tex = vx.dyntex().add_layer(
            LOGO,
            &LayerOptions {
                depth_test: false,
                ..LayerOptions::new()
            },
        );

        let base = Sprite {
            width: 1.0,
            height: 1.0,
            ..Sprite::new()
        };

        let mut dyntex = vx.dyntex();

        dyntex.add(
            &tex,
            Sprite {
                translation: (-0.5, -0.5),
                rotation: 0.0,
                ..base
            },
        );
        dyntex.add(
            &tex,
            Sprite {
                translation: (0.5, -0.5),
                rotation: PI / 4.0,
                ..base
            },
        );
        dyntex.add(
            &tex,
            Sprite {
                translation: (-0.5, 0.5),
                rotation: PI / 2.0,
                ..base
            },
        );
        dyntex.add(
            &tex,
            Sprite {
                translation: (0.5, 0.5),
                rotation: PI,
                ..base
            },
        );
        dyntex.translate_all(&tex, |_| (0.25, 0.35));

        let prspect = gen_perspective(&vx);
        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "translated_texture", img);
    }

    #[test]
    fn rotated_texture() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let mut dyntex = vx.dyntex();
        let tex = dyntex.add_layer(
            LOGO,
            &LayerOptions {
                depth_test: false,
                ..LayerOptions::new()
            },
        );

        let base = Sprite {
            width: 1.0,
            height: 1.0,
            ..Sprite::new()
        };

        dyntex.add(
            &tex,
            Sprite {
                translation: (-0.5, -0.5),
                rotation: 0.0,
                ..base
            },
        );
        dyntex.add(
            &tex,
            Sprite {
                translation: (0.5, -0.5),
                rotation: PI / 4.0,
                ..base
            },
        );
        dyntex.add(
            &tex,
            Sprite {
                translation: (-0.5, 0.5),
                rotation: PI / 2.0,
                ..base
            },
        );
        dyntex.add(
            &tex,
            Sprite {
                translation: (0.5, 0.5),
                rotation: PI,
                ..base
            },
        );
        dyntex.rotate_all(&tex, |_| Deg(90.0));

        let prspect = gen_perspective(&vx);
        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "rotated_texture", img);
    }

    #[test]
    fn many_sprites() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let tex = vx.dyntex().add_layer(
            LOGO,
            &LayerOptions {
                depth_test: false,
                ..LayerOptions::new()
            },
        );
        for i in 0..360 {
            vx.dyntex().add(
                &tex,
                Sprite {
                    rotation: ((i * 10) as f32 / 180f32 * PI),
                    scale: 0.5,
                    ..Sprite::new()
                },
            );
        }

        let prspect = gen_perspective(&vx);
        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "many_sprites", img);
    }

    #[test]
    fn three_layer_scene() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let options = &LayerOptions {
            depth_test: false,
            ..LayerOptions::new()
        };
        let mut dyntex = vx.dyntex();
        let forest = dyntex.add_layer(FOREST, options);
        let player = dyntex.add_layer(LOGO, options);
        let tree = dyntex.add_layer(TREE, options);

        vx.dyntex().add(&forest, Sprite::new());
        vx.dyntex().add(
            &player,
            Sprite {
                scale: 0.4,
                ..Sprite::new()
            },
        );
        vx.dyntex().add(
            &tree,
            Sprite {
                translation: (-0.3, 0.0),
                scale: 0.4,
                ..Sprite::new()
            },
        );

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "three_layer_scene", img);
    }

    #[test]
    fn three_layer_scene_remove_middle() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let options = &LayerOptions {
            depth_test: false,
            ..LayerOptions::new()
        };
        let mut dyntex = vx.dyntex();
        let forest = dyntex.add_layer(FOREST, options);
        let player = dyntex.add_layer(LOGO, options);
        let tree = dyntex.add_layer(TREE, options);

        dyntex.add(&forest, Sprite::new());
        let middle = dyntex.add(
            &player,
            Sprite {
                scale: 0.4,
                ..Sprite::new()
            },
        );
        dyntex.add(
            &tree,
            Sprite {
                translation: (-0.3, 0.0),
                scale: 0.4,
                ..Sprite::new()
            },
        );

        dyntex.remove(middle);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "three_layer_scene_remove_middle", img);
    }

    #[test]
    fn three_layer_scene_remove_middle_texture() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let options = &LayerOptions {
            depth_test: false,
            ..LayerOptions::new()
        };
        let mut dyntex = vx.dyntex();
        let forest = dyntex.add_layer(FOREST, options);
        let player = dyntex.add_layer(LOGO, options);
        let tree = dyntex.add_layer(TREE, options);

        dyntex.add(&forest, Sprite::new());
        dyntex.add(
            &player,
            Sprite {
                scale: 0.4,
                ..Sprite::new()
            },
        );
        dyntex.add(
            &tree,
            Sprite {
                translation: (-0.3, 0.0),
                scale: 0.4,
                ..Sprite::new()
            },
        );

        dyntex.remove_layer(player);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "three_layer_scene_remove_middle_texture", img);

        vx.dyntex().remove_layer(tree);

        vx.draw_frame(&prspect);
    }

    #[test]
    fn three_layer_scene_remove_last_texture() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let options = &LayerOptions {
            depth_test: false,
            ..LayerOptions::new()
        };

        let mut dyntex = vx.dyntex();
        let forest = dyntex.add_layer(FOREST, options);
        let player = dyntex.add_layer(LOGO, options);
        let tree = dyntex.add_layer(TREE, options);

        dyntex.add(&forest, Sprite::new());
        dyntex.add(
            &player,
            Sprite {
                scale: 0.4,
                ..Sprite::new()
            },
        );
        dyntex.add(
            &tree,
            Sprite {
                translation: (-0.3, 0.0),
                scale: 0.4,
                ..Sprite::new()
            },
        );

        dyntex.remove_layer(tree);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "three_layer_scene_remove_last_texture", img);

        vx.dyntex().remove_layer(player);

        vx.draw_frame(&prspect);
    }

    #[test]
    fn fixed_perspective() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless2x1k);
        let prspect = Matrix4::from_scale(0.0) * gen_perspective(&vx);

        let options = &LayerOptions {
            depth_test: false,
            fixed_perspective: Some(Matrix4::identity()),
            ..LayerOptions::new()
        };
        let forest = vx.dyntex().add_layer(FOREST, options);

        vx.dyntex().add(&forest, Sprite::new());

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "fixed_perspective", img);
    }

    #[test]
    fn change_of_uv_works_for_first() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut dyntex = vx.dyntex();

        let options = &LayerOptions::new();
        let testure = dyntex.add_layer(TESTURE, options);
        let sprite = dyntex.add(&testure, Sprite::new());

        dyntex.set_uvs(std::iter::once((
            &sprite,
            (1.0 / 3.0, 0.0),
            (2.0 / 3.0, 1.0),
        )));

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "change_of_uv_works_for_first", img);

        vx.dyntex()
            .set_uv(&sprite, (1.0 / 3.0, 0.0), (2.0 / 3.0, 1.0));

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "change_of_uv_works_for_first", img);
    }

    #[test]
    fn bunch_of_different_opacity_sprites() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut dyntex = vx.dyntex();

        let options = &LayerOptions::new();
        let testure = dyntex.add_layer(LOGO, options);

        for idx in 0..10 {
            dyntex.add(
                &testure,
                Sprite::new()
                    .translation((idx as f32 / 5.0 - 1.0, 0.0))
                    .opacity((255.0 * idx as f32 / 20.0) as u8),
            );
        }

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "bunch_of_different_opacity_sprites", img);
    }

    #[test]
    fn set_single_sprite_rotation() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut dyntex = vx.dyntex();
        let options = &LayerOptions::new();
        let testure = dyntex.add_layer(TESTURE, options);
        let sprite = dyntex.add(&testure, Sprite::new());
        dyntex.set_rotation(&sprite, Rad(0.3));

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "set_single_sprite_rotation", img);
    }

    #[test]
    fn linear_filtering_mode() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut dyntex = vx.dyntex();
        let options = &LayerOptions::new().filter(Filter::Linear);
        let testure = dyntex.add_layer(TESTURE, options);
        let sprite = dyntex.add(&testure, Sprite::new());

        dyntex.set_rotation(&sprite, Rad(0.3));

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "linear_filtering_mode", img);
    }

    #[test]
    fn raw_uvs() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut dyntex = vx.dyntex();
        let options = &LayerOptions::new();
        let testure = dyntex.add_layer(TESTURE, options);
        let sprite = dyntex.add(&testure, Sprite::new());
        dyntex.set_uv_raw(&sprite, [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "raw_uvs", img);
    }

    #[test]
    fn wrap_mode_clamp() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut dyntex = vx.dyntex();
        let options = &LayerOptions::new().wrap_mode(WrapMode::Clamp);
        let testure = dyntex.add_layer(TESTURE, options);
        let sprite = dyntex.add(&testure, Sprite::new());
        dyntex.set_uv_raw(&sprite, [(-0.5, 0.0), (-0.5, 1.0), (1.0, 1.0), (1.0, 0.0)]);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "wrap_mode_clamp", img);
    }

    #[test]
    fn wrap_mode_mirror() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut dyntex = vx.dyntex();
        let options = &LayerOptions::new().wrap_mode(WrapMode::Mirror);
        let testure = dyntex.add_layer(TESTURE, options);
        let sprite = dyntex.add(&testure, Sprite::new());
        dyntex.set_uv_raw(&sprite, [(-1.0, 0.0), (-1.0, 1.0), (1.0, 1.0), (1.0, 0.0)]);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "wrap_mode_mirror", img);
    }

    #[test]
    fn push_and_pop_often_avoid_allocating_out_of_bounds() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let options = &LayerOptions::new();
        let testure = vx.dyntex().add_layer(TESTURE, options);

        let mut dyntex = vx.dyntex();
        for _ in 0..100_000 {
            let sprite = dyntex.add(&testure, Sprite::new());
            dyntex.remove(sprite);
        }

        vx.draw_frame(&prspect);
    }

    #[test]
    fn too_little_data_in_texture_wraps() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let options = &LayerOptions::new();
        #[rustfmt::skip]
        let tex = ImgData::RawBytes {
            width: 7,
            height: 8,
            bytes: &[
                255,   0,   0, 255,
                  0, 255,   0, 255,
                  0,   0, 255, 255,
                255,   0, 255, 255,
                  0,   0,   0, 255,
            ],
        };
        let testure = vx.dyntex().add_layer(&tex, options);
        let sprite = vx.dyntex().add(&testure, Sprite::new());

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "too_little_data_in_texture_wraps", img);
    }

    #[bench]
    fn bench_many_sprites(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let tex = vx.dyntex().add_layer(LOGO, &LayerOptions::new());
        for i in 0..1000 {
            vx.dyntex().add(
                &tex,
                Sprite {
                    rotation: ((i * 10) as f32 / 180f32 * PI),
                    scale: 0.5,
                    ..Sprite::new()
                },
            );
        }

        let prspect = gen_perspective(&vx);
        b.iter(|| {
            vx.draw_frame(&prspect);
        });
    }

    #[bench]
    fn bench_many_particles(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let tex = vx.dyntex().add_layer(LOGO, &LayerOptions::new());
        let mut rng = random::new(0);
        for i in 0..1000 {
            let (dx, dy) = (
                rng.gen_range(-1.0f32, 1.0f32),
                rng.gen_range(-1.0f32, 1.0f32),
            );
            vx.dyntex().add(
                &tex,
                Sprite {
                    translation: (dx, dy),
                    rotation: ((i * 10) as f32 / 180f32 * PI),
                    scale: 0.01,
                    ..Sprite::new()
                },
            );
        }

        let prspect = gen_perspective(&vx);
        b.iter(|| {
            vx.draw_frame(&prspect);
        });
    }

    #[bench]
    fn animated_fireballs_20x20_uvs2(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let fireball_texture = vx.dyntex().add_layer(
            FIREBALL,
            &LayerOptions {
                depth_test: false,
                ..LayerOptions::new()
            },
        );

        let mut fireballs = vec![];
        for idx in -10..10 {
            for jdx in -10..10 {
                fireballs.push(vx.dyntex().add(
                    &fireball_texture,
                    Sprite {
                        width: 0.68,
                        height: 0.09,
                        rotation: idx as f32 / 18.0 + jdx as f32 / 16.0,
                        translation: (idx as f32 / 10.0, jdx as f32 / 10.0),
                        ..Sprite::new()
                    },
                ));
            }
        }

        let width_elems = 10;
        let height_elems = 6;

        let mut counter = 0;

        b.iter(|| {
            let width_elem = counter % width_elems;
            let height_elem = counter / width_elems;
            let uv_begin = (
                width_elem as f32 / width_elems as f32,
                height_elem as f32 / height_elems as f32,
            );
            let uv_end = (
                (width_elem + 1) as f32 / width_elems as f32,
                (height_elem + 1) as f32 / height_elems as f32,
            );
            counter += 1;
            if counter > width_elems * height_elems {
                counter = 0;
            }

            vx.dyntex()
                .set_uvs(fireballs.iter().map(|id| (id, uv_begin, uv_end)));
            vx.draw_frame(&prspect);
        });
    }

    #[bench]
    fn bench_push_and_pop_sprite(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let options = &LayerOptions::new();
        let testure = vx.dyntex().add_layer(TESTURE, options);

        let mut dyntex = vx.dyntex();
        b.iter(|| {
            let sprite = dyntex.add(&testure, Sprite::new());
            dyntex.remove(sprite);
        });
    }

    #[bench]
    fn bench_push_and_pop_texture(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let mut dyntex = vx.dyntex();

        b.iter(|| {
            let options = &LayerOptions::new();
            let testure = dyntex.add_layer(TESTURE, options);
            dyntex.remove_layer(testure);
        });
    }
}
