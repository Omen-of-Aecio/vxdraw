//! Tex
use super::{utils::*, Color};
use crate::{
    blender,
    data::{DrawType, DynamicTexture, SData, Text, VxDraw},
    strtex,
};
use cgmath::Matrix4;
use cgmath::Rad;
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
pub struct LayerOptions {
    // Wrap mode does not make sense here...
    /// Specify filtering mode for sampling the grid texture (default is [Filter::Nearest])
    filtering: Filter,
    /// Blending mode for this layer
    blend: blender::Blender,
}

impl LayerOptions {
    /// Create a new options structure
    pub fn new() -> Self {
        Self {
            filtering: Filter::Linear,
            blend: blender::Blender::default(),
        }
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

/// Handle to a piece of text
pub struct Handle(usize, std::ops::Range<usize>);

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
    pub fn new(vx: &'a mut VxDraw) -> Self {
        Self { vx }
    }

    /// Add a text layer to the system
    pub fn add_layer(&mut self, font: &'static [u8], options: LayerOptions) -> Layer {
        let glyph_brush = GlyphBrushBuilder::using_font_bytes(font).build();
        let (width, height) = glyph_brush.texture_dimensions();

        /// Create an image, image view, and memory
        self.vx
            .adapter
            .physical_device
            .image_format_properties(
                format::Format::Rgba8Srgb,
                2,
                image::Tiling::Linear,
                image::Usage::TRANSFER_SRC,
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

        /// Create a sampler
        let sampler = unsafe {
            self.vx
                .device
                .create_sampler(image::SamplerInfo::new(
                    match options.filtering {
                        Filter::Nearest => image::Filter::Nearest,
                        Filter::Linear => image::Filter::Linear,
                    },
                    image::WrapMode::Tile,
                ))
                .expect("Couldn't create the sampler!")
        };

        /// Add shader
        const VERTEX_SOURCE_TEXTURE: &[u8] = include_bytes!["../_build/spirv/strtex.vert.spirv"];
        const FRAGMENT_SOURCE_TEXTURE: &[u8] = include_bytes!["../_build/spirv/strtex.frag.spirv"];

        let vs_module =
            { unsafe { self.vx.device.create_shader_module(&VERTEX_SOURCE_TEXTURE) }.unwrap() };
        let fs_module = {
            unsafe {
                self.vx
                    .device
                    .create_shader_module(&FRAGMENT_SOURCE_TEXTURE)
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

        /// Describe input data
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
            depth: pso::DepthTest::Off,
            depth_bounds: false,
            stencil: pso::StencilTest::Off,
        };

        let blender = options.blend.clone().to_gfx_blender();

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

        /// Write descriptor sets
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

        /// Push constants
        let mut push_constants = Vec::<(pso::ShaderStageFlags, core::ops::Range<u32>)>::new();
        push_constants.push((pso::ShaderStageFlags::VERTEX, 0..16));

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

        /// Clean up
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
            buffer.begin(false);
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
            self.vx.queue_group.queues[0].submit_nosemaphores(Some(&*buffer), Some(&barrier_fence));
            self.vx
                .device
                .wait_for_fence(&barrier_fence, u64::max_value())
                .unwrap();
            self.vx.device.destroy_fence(barrier_fence);
        }

        self.vx.texts.push(Text {
            hidden: false,
            removed: vec![],
            glyph_brush,

            fixed_perspective: None,

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
        });

        self.vx.draw_order.push(DrawType::Text {
            id: self.vx.texts.len() - 1,
        });
        Layer(self.vx.texts.len() - 1)
    }

    /// Add text to this layer
    pub fn add(&mut self, layer: &mut Layer, opts: TextOptions) -> Handle {
        self.vx.texts[layer.0].glyph_brush.queue(Section {
            text: opts.text,
            scale: glyph_brush::rusttype::Scale {
                x: opts.font_size_x,
                y: opts.font_size_y,
            },
            ..Section::default()
        });
        self.vx.wait_for_fences();
        self.vx.texts[layer.0].posbuf_touch = self.vx.swapconfig.image_count;
        self.vx.texts[layer.0].opacbuf_touch = self.vx.swapconfig.image_count;
        self.vx.texts[layer.0].uvbuf_touch = self.vx.swapconfig.image_count;
        self.vx.texts[layer.0].tranbuf_touch = self.vx.swapconfig.image_count;
        self.vx.texts[layer.0].rotbuf_touch = self.vx.swapconfig.image_count;
        self.vx.texts[layer.0].scalebuf_touch = self.vx.swapconfig.image_count;
        let prev_begin = self.vx.texts[layer.0].posbuffer.len();
        let mut count = 0usize;
        let mut tex_values = vec![];
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
                    tex.tranbuffer.push([0.0; 8]);
                    tex.rotbuffer.push([0.0; 4]);
                    tex.scalebuffer.push([1.0; 4]);
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
                println!["{:?}", suggested];
            }
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
                let mut target = self
                    .vx
                    .device
                    .acquire_mapping_writer(
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
                    target[access as usize..(access + 4) as usize]
                        .copy_from_slice(&[255, 255, 255, *alpha]);
                }
                self.vx
                    .device
                    .release_mapping_writer(target)
                    .expect("Unable to release mapping writer");
            }
        }
        Handle(layer.0, prev_begin + 1..prev_begin + 1 + count)
    }
}
