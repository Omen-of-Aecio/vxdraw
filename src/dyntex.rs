use super::utils::*;
use crate::data::{DrawType, SingleTexture, Windowing};
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
use std::io::Read;
use std::mem::{size_of, ManuallyDrop};

pub struct Dyntex<'a> {
    windowing: &'a mut Windowing,
}

impl<'a> Dyntex<'a> {
    pub fn new(s: &'a mut Windowing) -> Self {
        Self { windowing: s }
    }

    /// Add a texture to the system
    ///
    /// You use a texture to create sprites. Sprites are rectangular views into a texture. Sprites
    /// based on different texures are drawn in the order in which the textures were allocated, that
    /// means that the first texture's sprites are drawn first, then, the second texture's sprites,and
    /// so on.
    ///
    /// Each texture has options (See `TextureOptions`). This decides how the derivative sprites are
    /// drawn.
    ///
    /// Note: Alpha blending with depth testing will make foreground transparency not be transparent.
    /// To make sure transparency works correctly you can turn off the depth test for foreground
    /// objects and ensure that the foreground texture is allocated last.
    pub fn push_texture(&mut self, img_data: &[u8], options: TextureOptions) -> TextureHandle {
        let s = &mut *self.windowing;
        let device = &*s.device;

        let img = load_image::load_from_memory_with_format(&img_data[..], load_image::PNG)
            .unwrap()
            .to_rgba();

        let pixel_size = 4; //size_of::<image::Rgba<u8>>();
        let row_size = pixel_size * (img.width() as usize);
        let limits = s.adapter.physical_device.limits();
        let row_alignment_mask = limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let row_pitch = ((row_size as u32 + row_alignment_mask) & !row_alignment_mask) as usize;
        debug_assert!(row_pitch as usize >= row_size);
        let required_bytes = row_pitch * img.height() as usize;

        let mut image_upload_buffer = unsafe {
            device.create_buffer(required_bytes as u64, gfx_hal::buffer::Usage::TRANSFER_SRC)
        }
        .unwrap();
        let image_mem_reqs = unsafe { device.get_buffer_requirements(&image_upload_buffer) };
        let memory_type_id =
            find_memory_type_id(&s.adapter, image_mem_reqs, Properties::CPU_VISIBLE);
        let image_upload_memory =
            unsafe { device.allocate_memory(memory_type_id, image_mem_reqs.size) }.unwrap();
        unsafe { device.bind_buffer_memory(&image_upload_memory, 0, &mut image_upload_buffer) }
            .unwrap();

        unsafe {
            let mut writer = s
                .device
                .acquire_mapping_writer::<u8>(&image_upload_memory, 0..image_mem_reqs.size)
                .expect("Unable to get mapping writer");
            for y in 0..img.height() as usize {
                let row = &(*img)[y * row_size..(y + 1) * row_size];
                let dest_base = y * row_pitch;
                writer[dest_base..dest_base + row.len()].copy_from_slice(row);
            }
            device
                .release_mapping_writer(writer)
                .expect("Couldn't release the mapping writer to the staging buffer!");
        }

        let mut the_image = unsafe {
            device
                .create_image(
                    image::Kind::D2(img.width(), img.height(), 1, 1),
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
                    image::Filter::Nearest,
                    image::WrapMode::Tile,
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
                    buffer_height: img.height(),
                    image_layers: gfx_hal::image::SubresourceLayers {
                        aspects: format::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: image::Offset { x: 0, y: 0, z: 0 },
                    image_extent: image::Extent {
                        width: img.width(),
                        height: img.height(),
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

        const VERTEX_SOURCE_TEXTURE: &str = "#version 450
        #extension GL_ARB_separate_shader_objects : enable

        layout(location = 0) in vec3 v_pos;
        layout(location = 1) in vec2 v_uv;
        layout(location = 2) in vec2 v_dxdy;
        layout(location = 3) in float rotation;
        layout(location = 4) in float scale;
        layout(location = 5) in vec4 color;

        layout(location = 0) out vec2 f_uv;
        layout(location = 1) out vec4 f_color;

        layout(push_constant) uniform PushConstant {
            mat4 view;
        } push_constant;

        out gl_PerVertex {
            vec4 gl_Position;
        };

        void main() {
            mat2 rotmatrix = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation));
            vec2 pos = rotmatrix * scale * v_pos.xy;
            f_uv = v_uv;
            f_color = color;
            gl_Position = push_constant.view * vec4(pos + v_dxdy, v_pos.z, 1.0);
        }";

        const FRAGMENT_SOURCE_TEXTURE: &str = "#version 450
        #extension GL_ARB_separate_shader_objects : enable

        layout(location = 0) in vec2 f_uv;
        layout(location = 1) in vec4 f_color;

        layout(location = 0) out vec4 color;

        layout(set = 0, binding = 0) uniform texture2D f_texture;
        layout(set = 0, binding = 1) uniform sampler f_sampler;

        void main() {
            color = texture(sampler2D(f_texture, f_sampler), f_uv);
            color.a *= f_color.a;
            color.rgb += f_color.rgb;
        }";

        let vs_module = {
            let glsl = VERTEX_SOURCE_TEXTURE;
            let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex)
                .unwrap()
                .bytes()
                .map(Result::unwrap)
                .collect();
            unsafe { s.device.create_shader_module(&spirv) }.unwrap()
        };
        let fs_module = {
            let glsl = FRAGMENT_SOURCE_TEXTURE;
            let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Fragment)
                .unwrap()
                .bytes()
                .map(Result::unwrap)
                .collect();
            unsafe { s.device.create_shader_module(&spirv) }.unwrap()
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

        let vertex_buffers: Vec<pso::VertexBufferDesc> = vec![pso::VertexBufferDesc {
            binding: 0,
            stride: (size_of::<f32>() * (3 + 2 + 2 + 2 + 1)) as u32,
            rate: pso::VertexInputRate::Vertex,
        }];
        let attributes: Vec<pso::AttributeDesc> = vec![
            pso::AttributeDesc {
                location: 0,
                binding: 0,
                element: pso::Element {
                    format: format::Format::Rgb32Sfloat,
                    offset: 0,
                },
            },
            pso::AttributeDesc {
                location: 1,
                binding: 0,
                element: pso::Element {
                    format: format::Format::Rg32Sfloat,
                    offset: 12,
                },
            },
            pso::AttributeDesc {
                location: 2,
                binding: 0,
                element: pso::Element {
                    format: format::Format::Rg32Sfloat,
                    offset: 20,
                },
            },
            pso::AttributeDesc {
                location: 3,
                binding: 0,
                element: pso::Element {
                    format: format::Format::R32Sfloat,
                    offset: 28,
                },
            },
            pso::AttributeDesc {
                location: 4,
                binding: 0,
                element: pso::Element {
                    format: format::Format::R32Sfloat,
                    offset: 32,
                },
            },
            pso::AttributeDesc {
                location: 5,
                binding: 0,
                element: pso::Element {
                    format: format::Format::Rgba8Unorm,
                    offset: 36,
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

        let (texture_vertex_buffer, texture_vertex_memory, texture_vertex_requirements) =
            make_vertex_buffer_with_data(s, &[0f32; 10 * 4 * 1000]);

        const INDEX_COUNT: usize = 1000;
        let (
            texture_vertex_buffer_indices,
            texture_vertex_memory_indices,
            texture_vertex_requirements_indices,
        ) = make_index_buffer_with_data(s, &[0f32; 3 * INDEX_COUNT]);

        unsafe {
            let mut data_target = s
                .device
                .acquire_mapping_writer(
                    &texture_vertex_memory_indices,
                    0..texture_vertex_requirements_indices.size,
                )
                .expect("Failed to acquire a memory writer!");
            for index in 0..INDEX_COUNT {
                let ver = (index * 6) as u16;
                let ind = (index * 4) as u16;
                data_target[ver as usize..(ver + 6) as usize].copy_from_slice(&[
                    ind,
                    ind + 1,
                    ind + 2,
                    ind + 2,
                    ind + 3,
                    ind,
                ]);
            }
            s.device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }

        s.dyntexs.push(SingleTexture {
            count: 0,

            fixed_perspective: options.fixed_perspective,
            mockbuffer: vec![],
            removed: vec![],

            texture_vertex_buffer: ManuallyDrop::new(texture_vertex_buffer),
            texture_vertex_memory: ManuallyDrop::new(texture_vertex_memory),
            texture_vertex_requirements,

            texture_vertex_buffer_indices: ManuallyDrop::new(texture_vertex_buffer_indices),
            texture_vertex_memory_indices: ManuallyDrop::new(texture_vertex_memory_indices),
            texture_vertex_requirements_indices,

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
        TextureHandle(s.dyntexs.len() - 1)
    }

    /// Add a sprite (a rectangular view of a texture) to the system
    ///
    /// The sprite is automatically drawn on each [draw] call, and must be removed by
    /// [remove_sprite] to stop it from being drawn.
    pub fn push_sprite(&mut self, texture: &TextureHandle, sprite: Sprite) -> SpriteHandle {
        let s = &mut *self.windowing;
        let tex = &mut s.dyntexs[texture.0];

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

        let index = if let Some(value) = tex.removed.pop() {
            value as u32
        } else {
            let old = tex.count;
            tex.count += 1;
            old
        };

        unsafe {
            let idx = (index * 4 * 10 * 4) as usize;

            while tex.mockbuffer.len() <= idx {
                tex.mockbuffer.extend([0u8; 4 * 40].iter());
            }
            for (i, (point, uv)) in [
                (topleft, topleft_uv),
                (bottomleft, bottomleft_uv),
                (bottomright, bottomright_uv),
                (topright, topright_uv),
            ]
            .iter()
            .enumerate()
            {
                let idx = idx + i * 10 * 4;
                use std::mem::transmute;
                let x = &transmute::<f32, [u8; 4]>(point.0);
                let y = &transmute::<f32, [u8; 4]>(point.1);

                let uv0 = &transmute::<f32, [u8; 4]>(uv.0);
                let uv1 = &transmute::<f32, [u8; 4]>(uv.1);

                let tr0 = &transmute::<f32, [u8; 4]>(sprite.translation.0);
                let tr1 = &transmute::<f32, [u8; 4]>(sprite.translation.1);

                let rot = &transmute::<f32, [u8; 4]>(sprite.rotation);
                let scale = &transmute::<f32, [u8; 4]>(sprite.scale);

                let colors = &transmute::<(u8, u8, u8, u8), [u8; 4]>(sprite.colors[i]);

                tex.mockbuffer[idx..idx + 4].copy_from_slice(x);
                tex.mockbuffer[idx + 4..idx + 8].copy_from_slice(y);
                tex.mockbuffer[idx + 8..idx + 12]
                    .copy_from_slice(&transmute::<f32, [u8; 4]>(sprite.depth));

                tex.mockbuffer[idx + 12..idx + 16].copy_from_slice(uv0);
                tex.mockbuffer[idx + 16..idx + 20].copy_from_slice(uv1);

                tex.mockbuffer[idx + 20..idx + 24].copy_from_slice(tr0);
                tex.mockbuffer[idx + 24..idx + 28].copy_from_slice(tr1);

                tex.mockbuffer[idx + 28..idx + 32].copy_from_slice(rot);
                tex.mockbuffer[idx + 32..idx + 36].copy_from_slice(scale);
                tex.mockbuffer[idx + 36..idx + 40].copy_from_slice(colors);
            }
        }
        SpriteHandle(texture.0, index as usize)
    }

    /// Remove a texture
    ///
    /// Removes the texture from memory and destroys all sprites associated with it.
    /// All lingering sprite handles that were spawned using this texture handle will be
    /// invalidated.
    pub fn remove_texture(&mut self, texture: TextureHandle) {
        let s = &mut *self.windowing;
        let mut index = None;
        for (idx, x) in s.draw_order.iter().enumerate() {
            match x {
                DrawType::DynamicTexture { id } if *id == texture.0 => {
                    index = Some(idx);
                    break;
                }
                _ => {}
            }
        }
        if let Some(idx) = index {
            s.draw_order.remove(idx);
            // Can't delete here always because other textures may still be referring to later dyntexs,
            // only when this is the last texture.
            if s.dyntexs.len() == texture.0 + 1 {
                let dyntex = s.dyntexs.pop().unwrap();
                destroy_texture(s, dyntex);
            }
        }
    }

    /// Removes a single sprite, making it not be drawn
    pub fn remove_sprite(&mut self, handle: SpriteHandle) {
        let s = &mut *self.windowing;
        if let Some(dyntex) = s.dyntexs.get_mut(handle.0) {
            let idx = (handle.1 * 4 * 10 * 4) as usize;
            let zero = unsafe { std::mem::transmute::<f32, [u8; 4]>(0.0) };
            for idx in (0..=3).map(|x| (x * 40) + idx) {
                dyntex.mockbuffer[idx + 32..idx + 36].copy_from_slice(&zero);
            }
            dyntex.removed.push(handle.1);
        }
    }

    /// Set the position of a sprite
    pub fn set_position(&mut self, handle: &SpriteHandle, position: (f32, f32)) {
        let s = &mut *self.windowing;
        if let Some(stex) = s.dyntexs.get_mut(handle.0) {
            unsafe {
                use std::mem::transmute;
                let position0 = &transmute::<f32, [u8; 4]>(position.0);
                let position1 = &transmute::<f32, [u8; 4]>(position.1);

                let mut idx = (handle.1 * 4 * 10 * 4) as usize;

                stex.mockbuffer[idx + 5 * 4..idx + 6 * 4].copy_from_slice(position0);
                stex.mockbuffer[idx + 6 * 4..idx + 7 * 4].copy_from_slice(position1);
                idx += 40;
                stex.mockbuffer[idx + 5 * 4..idx + 6 * 4].copy_from_slice(position0);
                stex.mockbuffer[idx + 6 * 4..idx + 7 * 4].copy_from_slice(position1);
                idx += 40;
                stex.mockbuffer[idx + 5 * 4..idx + 6 * 4].copy_from_slice(position0);
                stex.mockbuffer[idx + 6 * 4..idx + 7 * 4].copy_from_slice(position1);
                idx += 40;
                stex.mockbuffer[idx + 5 * 4..idx + 6 * 4].copy_from_slice(position0);
                stex.mockbuffer[idx + 6 * 4..idx + 7 * 4].copy_from_slice(position1);
            }
        }
    }

    /// Set the rotation of a sprite
    ///
    /// Positive rotation goes counter-clockwise. The value of the rotation is in radians.
    pub fn set_rotation<T: Copy + Into<Rad<f32>>>(&mut self, handle: &SpriteHandle, rotation: T) {
        let s = &mut *self.windowing;
        if let Some(stex) = s.dyntexs.get_mut(handle.0) {
            unsafe {
                use std::mem::transmute;
                let rot = &transmute::<f32, [u8; 4]>(rotation.into().0);

                let mut idx = (handle.1 * 4 * 10 * 4) as usize;

                stex.mockbuffer[idx + 7 * 4..idx + 8 * 4].copy_from_slice(rot);
                idx += 40;
                stex.mockbuffer[idx + 7 * 4..idx + 8 * 4].copy_from_slice(rot);
                idx += 40;
                stex.mockbuffer[idx + 7 * 4..idx + 8 * 4].copy_from_slice(rot);
                idx += 40;
                stex.mockbuffer[idx + 7 * 4..idx + 8 * 4].copy_from_slice(rot);
            }
        }
    }

    /// Translate all sprites that depend on a given texture
    ///
    /// Convenience method that translates all sprites associated with the given texture.
    pub fn sprite_translate_all(&mut self, tex: &TextureHandle, dxdy: (f32, f32)) {
        let s = &mut *self.windowing;
        if let Some(stex) = s.dyntexs.get_mut(tex.0) {
            unsafe {
                for mock in stex.mockbuffer.chunks_mut(40) {
                    use std::mem::transmute;
                    let x = transmute::<&[u8], &[f32]>(&mock[5 * 4..6 * 4]);
                    let y = transmute::<&[u8], &[f32]>(&mock[6 * 4..7 * 4]);
                    mock[5 * 4..6 * 4].copy_from_slice(&transmute::<f32, [u8; 4]>(x[0] + dxdy.0));
                    mock[6 * 4..7 * 4].copy_from_slice(&transmute::<f32, [u8; 4]>(y[0] + dxdy.1));
                }
            }
        }
    }

    /// Rotate all sprites that depend on a given texture
    ///
    /// Convenience method that rotates all sprites associated with the given texture.
    pub fn sprite_rotate_all<T: Copy + Into<Rad<f32>>>(&mut self, tex: &TextureHandle, deg: T) {
        let s = &mut *self.windowing;
        if let Some(stex) = s.dyntexs.get_mut(tex.0) {
            unsafe {
                for mock in stex.mockbuffer.chunks_mut(40) {
                    use std::mem::transmute;
                    let deggy = transmute::<&[u8], &[f32]>(&mock[28..32]);
                    mock[28..32]
                        .copy_from_slice(&transmute::<f32, [u8; 4]>(deggy[0] + deg.into().0));
                }
            }
        }
    }

    pub fn set_uv(&mut self, handle: &SpriteHandle, uv_begin: (f32, f32), uv_end: (f32, f32)) {
        let s = &mut *self.windowing;
        if let Some(stex) = s.dyntexs.get_mut(handle.0) {
            if handle.1 < stex.count as usize {
                unsafe {
                    let mut idx = (handle.1 * 4 * 10 * 4) as usize;

                    use std::mem::transmute;
                    let begin0 = &transmute::<f32, [u8; 4]>(uv_begin.0);
                    let begin1 = &transmute::<f32, [u8; 4]>(uv_begin.1);
                    let end0 = &transmute::<f32, [u8; 4]>(uv_end.0);
                    let end1 = &transmute::<f32, [u8; 4]>(uv_end.1);

                    stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(begin0);
                    stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(begin1);
                    idx += 40;
                    stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(begin0);
                    stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(end1);
                    idx += 40;
                    stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(end0);
                    stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(end1);
                    idx += 40;
                    stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(end0);
                    stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(begin1);
                }
            }
        }
    }

    pub fn set_uvs2<'b>(
        &mut self,
        mut uvs: impl Iterator<Item = (&'b SpriteHandle, (f32, f32), (f32, f32))>,
    ) {
        let s = &mut *self.windowing;
        if let Some(first) = uvs.next() {
            if let Some(ref mut stex) = s.dyntexs.get_mut((first.0).0) {
                let current_texture_handle = (first.0).0;
                unsafe {
                    if (first.0).1 < stex.count as usize {
                        let mut idx = ((first.0).1 * 4 * 10 * 4) as usize;
                        let uv_begin = first.1;
                        let uv_end = first.2;

                        use std::mem::transmute;
                        let begin0 = &transmute::<f32, [u8; 4]>(uv_begin.0);
                        let begin1 = &transmute::<f32, [u8; 4]>(uv_begin.1);
                        let end0 = &transmute::<f32, [u8; 4]>(uv_end.0);
                        let end1 = &transmute::<f32, [u8; 4]>(uv_end.1);

                        stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(begin0);
                        stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(begin1);
                        idx += 40;
                        stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(begin0);
                        stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(end1);
                        idx += 40;
                        stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(end0);
                        stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(end1);
                        idx += 40;
                        stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(end0);
                        stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(begin1);
                    }
                    for handle in uvs {
                        if (handle.0).0 != current_texture_handle {
                            panic!["The texture handles of each sprite must be identical"];
                        }
                        if (handle.0).1 < stex.count as usize {
                            let mut idx = ((handle.0).1 * 4 * 10 * 4) as usize;
                            let uv_begin = handle.1;
                            let uv_end = handle.2;

                            use std::mem::transmute;
                            let begin0 = &transmute::<f32, [u8; 4]>(uv_begin.0);
                            let begin1 = &transmute::<f32, [u8; 4]>(uv_begin.1);
                            let end0 = &transmute::<f32, [u8; 4]>(uv_end.0);
                            let end1 = &transmute::<f32, [u8; 4]>(uv_end.1);

                            stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(begin0);
                            stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(begin1);
                            idx += 40;
                            stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(begin0);
                            stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(end1);
                            idx += 40;
                            stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(end0);
                            stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(end1);
                            idx += 40;
                            stex.mockbuffer[idx + 3 * 4..idx + 4 * 4].copy_from_slice(end0);
                            stex.mockbuffer[idx + 4 * 4..idx + 5 * 4].copy_from_slice(begin1);
                        }
                    }
                }
            }
        }
    }
}

// ---

/// A view into a texture
///
/// A sprite is a rectangular view into a texture.
#[derive(Clone, Copy)]
pub struct Sprite {
    pub width: f32,
    pub height: f32,
    pub depth: f32,
    pub colors: [(u8, u8, u8, u8); 4],
    pub uv_begin: (f32, f32),
    pub uv_end: (f32, f32),
    pub translation: (f32, f32),
    pub rotation: f32,
    pub scale: f32,
    pub origin: (f32, f32),
}

impl Default for Sprite {
    fn default() -> Self {
        Sprite {
            width: 2.0,
            height: 2.0,
            depth: 0.0,
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

/// A view into a texture
pub struct SpriteHandle(usize, usize);

/// Handle to a texture
pub struct TextureHandle(usize);

#[derive(Clone, Copy)]
pub struct TextureOptions {
    /// Perform depth testing (and fragment culling) when drawing sprites from this texture
    pub depth_test: bool,
    /// Fix the perspective, this ignores the perspective sent into draw for this texture and
    /// all its associated sprites
    pub fixed_perspective: Option<Matrix4<f32>>,
}

impl Default for TextureOptions {
    fn default() -> Self {
        Self {
            depth_test: true,
            fixed_perspective: None,
        }
    }
}

// ---

fn destroy_texture(s: &mut Windowing, mut dyntex: SingleTexture) {
    unsafe {
        s.device.destroy_buffer(ManuallyDrop::into_inner(read(
            &dyntex.texture_vertex_buffer_indices,
        )));
        s.device.free_memory(ManuallyDrop::into_inner(read(
            &dyntex.texture_vertex_memory_indices,
        )));
        s.device.destroy_buffer(ManuallyDrop::into_inner(read(
            &dyntex.texture_vertex_buffer,
        )));
        s.device.free_memory(ManuallyDrop::into_inner(read(
            &dyntex.texture_vertex_memory,
        )));
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
    use logger::{Generic, GenericLogger, Logger};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg as random;
    use std::f32::consts::PI;
    use test::Bencher;

    // ---

    static LOGO: &[u8] = include_bytes!["../images/logo.png"];
    static FOREST: &[u8] = include_bytes!["../images/forest-light.png"];
    static TESTURE: &[u8] = include_bytes!["../images/testure.png"];
    static TREE: &[u8] = include_bytes!["../images/treetest.png"];
    static FIREBALL: &[u8] = include_bytes!["../images/Fireball_68x9.png"];

    // ---

    #[test]
    fn overlapping_dyntex_respect_z_order() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let mut dyntex = windowing.dyntex();

        let tree = dyntex.push_texture(TREE, TextureOptions::default());
        let logo = dyntex.push_texture(LOGO, TextureOptions::default());

        let sprite = Sprite {
            scale: 0.5,
            ..Sprite::default()
        };

        windowing.dyntex().push_sprite(
            &tree,
            Sprite {
                depth: 0.5,
                ..sprite
            },
        );
        windowing.dyntex().push_sprite(
            &logo,
            Sprite {
                depth: 0.6,
                translation: (0.25, 0.25),
                ..sprite
            },
        );

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "overlapping_dyntex_respect_z_order", img);
    }

    #[test]
    fn simple_texture() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);

        let mut dyntex = windowing.dyntex();
        let tex = dyntex.push_texture(LOGO, TextureOptions::default());
        windowing.dyntex().push_sprite(&tex, Sprite::default());

        let prspect = gen_perspective(&windowing);
        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "simple_texture", img);
    }

    #[test]
    fn simple_texture_adheres_to_view() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless2x1k);
        let tex = windowing
            .dyntex()
            .push_texture(LOGO, TextureOptions::default());
        windowing.dyntex().push_sprite(&tex, Sprite::default());

        let prspect = gen_perspective(&windowing);
        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "simple_texture_adheres_to_view", img);
    }

    #[test]
    fn colored_simple_texture() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let tex = windowing
            .dyntex()
            .push_texture(LOGO, TextureOptions::default());
        windowing.dyntex().push_sprite(
            &tex,
            Sprite {
                colors: [
                    (255, 1, 2, 255),
                    (0, 255, 0, 255),
                    (0, 0, 255, 100),
                    (255, 2, 1, 0),
                ],
                ..Sprite::default()
            },
        );

        let prspect = gen_perspective(&windowing);
        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "colored_simple_texture", img);
    }

    #[test]
    fn colored_simple_texture_set_position() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);

        let mut dyntex = windowing.dyntex();
        let tex = dyntex.push_texture(LOGO, TextureOptions::default());
        let sprite = dyntex.push_sprite(
            &tex,
            Sprite {
                colors: [
                    (255, 1, 2, 255),
                    (0, 255, 0, 255),
                    (0, 0, 255, 100),
                    (255, 2, 1, 0),
                ],
                ..Sprite::default()
            },
        );
        dyntex.set_position(&sprite, (0.5, 0.3));

        let prspect = gen_perspective(&windowing);
        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "colored_simple_texture_set_position", img);
    }

    #[test]
    fn translated_texture() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let tex = windowing.dyntex().push_texture(
            LOGO,
            TextureOptions {
                depth_test: false,
                ..TextureOptions::default()
            },
        );

        let base = Sprite {
            width: 1.0,
            height: 1.0,
            ..Sprite::default()
        };

        let mut dyntex = windowing.dyntex();

        dyntex.push_sprite(
            &tex,
            Sprite {
                translation: (-0.5, -0.5),
                rotation: 0.0,
                ..base
            },
        );
        dyntex.push_sprite(
            &tex,
            Sprite {
                translation: (0.5, -0.5),
                rotation: PI / 4.0,
                ..base
            },
        );
        dyntex.push_sprite(
            &tex,
            Sprite {
                translation: (-0.5, 0.5),
                rotation: PI / 2.0,
                ..base
            },
        );
        dyntex.push_sprite(
            &tex,
            Sprite {
                translation: (0.5, 0.5),
                rotation: PI,
                ..base
            },
        );
        dyntex.sprite_translate_all(&tex, (0.25, 0.35));

        let prspect = gen_perspective(&windowing);
        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "translated_texture", img);
    }

    #[test]
    fn rotated_texture() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let mut dyntex = windowing.dyntex();
        let tex = dyntex.push_texture(
            LOGO,
            TextureOptions {
                depth_test: false,
                ..TextureOptions::default()
            },
        );

        let base = Sprite {
            width: 1.0,
            height: 1.0,
            ..Sprite::default()
        };

        dyntex.push_sprite(
            &tex,
            Sprite {
                translation: (-0.5, -0.5),
                rotation: 0.0,
                ..base
            },
        );
        dyntex.push_sprite(
            &tex,
            Sprite {
                translation: (0.5, -0.5),
                rotation: PI / 4.0,
                ..base
            },
        );
        dyntex.push_sprite(
            &tex,
            Sprite {
                translation: (-0.5, 0.5),
                rotation: PI / 2.0,
                ..base
            },
        );
        dyntex.push_sprite(
            &tex,
            Sprite {
                translation: (0.5, 0.5),
                rotation: PI,
                ..base
            },
        );
        dyntex.sprite_rotate_all(&tex, Deg(90.0));

        let prspect = gen_perspective(&windowing);
        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "rotated_texture", img);
    }

    #[test]
    fn many_sprites() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let tex = windowing.dyntex().push_texture(
            LOGO,
            TextureOptions {
                depth_test: false,
                ..TextureOptions::default()
            },
        );
        for i in 0..360 {
            windowing.dyntex().push_sprite(
                &tex,
                Sprite {
                    rotation: ((i * 10) as f32 / 180f32 * PI),
                    scale: 0.5,
                    ..Sprite::default()
                },
            );
        }

        let prspect = gen_perspective(&windowing);
        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "many_sprites", img);
    }

    #[test]
    fn three_layer_scene() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let options = TextureOptions {
            depth_test: false,
            ..TextureOptions::default()
        };
        let mut dyntex = windowing.dyntex();
        let forest = dyntex.push_texture(FOREST, options);
        let player = dyntex.push_texture(LOGO, options);
        let tree = dyntex.push_texture(TREE, options);

        windowing.dyntex().push_sprite(&forest, Sprite::default());
        windowing.dyntex().push_sprite(
            &player,
            Sprite {
                scale: 0.4,
                ..Sprite::default()
            },
        );
        windowing.dyntex().push_sprite(
            &tree,
            Sprite {
                translation: (-0.3, 0.0),
                scale: 0.4,
                ..Sprite::default()
            },
        );

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "three_layer_scene", img);
    }

    #[test]
    fn three_layer_scene_remove_middle() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let options = TextureOptions {
            depth_test: false,
            ..TextureOptions::default()
        };
        let mut dyntex = windowing.dyntex();
        let forest = dyntex.push_texture(FOREST, options);
        let player = dyntex.push_texture(LOGO, options);
        let tree = dyntex.push_texture(TREE, options);

        dyntex.push_sprite(&forest, Sprite::default());
        let middle = dyntex.push_sprite(
            &player,
            Sprite {
                scale: 0.4,
                ..Sprite::default()
            },
        );
        dyntex.push_sprite(
            &tree,
            Sprite {
                translation: (-0.3, 0.0),
                scale: 0.4,
                ..Sprite::default()
            },
        );

        dyntex.remove_sprite(middle);

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "three_layer_scene_remove_middle", img);
    }

    #[test]
    fn three_layer_scene_remove_middle_texture() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let options = TextureOptions {
            depth_test: false,
            ..TextureOptions::default()
        };
        let mut dyntex = windowing.dyntex();
        let forest = dyntex.push_texture(FOREST, options);
        let player = dyntex.push_texture(LOGO, options);
        let tree = dyntex.push_texture(TREE, options);

        dyntex.push_sprite(&forest, Sprite::default());
        dyntex.push_sprite(
            &player,
            Sprite {
                scale: 0.4,
                ..Sprite::default()
            },
        );
        dyntex.push_sprite(
            &tree,
            Sprite {
                translation: (-0.3, 0.0),
                scale: 0.4,
                ..Sprite::default()
            },
        );

        dyntex.remove_texture(player);

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(
            &mut windowing,
            "three_layer_scene_remove_middle_texture",
            img,
        );

        windowing.dyntex().remove_texture(tree);

        draw_frame(&mut windowing, &prspect);
    }

    #[test]
    fn three_layer_scene_remove_last_texture() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let options = TextureOptions {
            depth_test: false,
            ..TextureOptions::default()
        };

        let mut dyntex = windowing.dyntex();
        let forest = dyntex.push_texture(FOREST, options);
        let player = dyntex.push_texture(LOGO, options);
        let tree = dyntex.push_texture(TREE, options);

        dyntex.push_sprite(&forest, Sprite::default());
        dyntex.push_sprite(
            &player,
            Sprite {
                scale: 0.4,
                ..Sprite::default()
            },
        );
        dyntex.push_sprite(
            &tree,
            Sprite {
                translation: (-0.3, 0.0),
                scale: 0.4,
                ..Sprite::default()
            },
        );

        dyntex.remove_texture(tree);

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "three_layer_scene_remove_last_texture", img);

        windowing.dyntex().remove_texture(player);

        draw_frame(&mut windowing, &prspect);
    }

    #[test]
    fn fixed_perspective() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless2x1k);
        let prspect = Matrix4::from_scale(0.0) * gen_perspective(&windowing);

        let options = TextureOptions {
            depth_test: false,
            fixed_perspective: Some(Matrix4::identity()),
            ..TextureOptions::default()
        };
        let forest = windowing.dyntex().push_texture(FOREST, options);

        windowing.dyntex().push_sprite(&forest, Sprite::default());

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "fixed_perspective", img);
    }

    #[test]
    fn change_of_uv_works_for_first() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let mut dyntex = windowing.dyntex();

        let options = TextureOptions::default();
        let testure = dyntex.push_texture(TESTURE, options);
        let sprite = dyntex.push_sprite(&testure, Sprite::default());

        dyntex.set_uvs2(std::iter::once((
            &sprite,
            (1.0 / 3.0, 0.0),
            (2.0 / 3.0, 1.0),
        )));

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "change_of_uv_works_for_first", img);

        windowing
            .dyntex()
            .set_uv(&sprite, (1.0 / 3.0, 0.0), (2.0 / 3.0, 1.0));

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "change_of_uv_works_for_first", img);
    }

    #[test]
    fn set_single_sprite_rotation() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let mut dyntex = windowing.dyntex();
        let options = TextureOptions::default();
        let testure = dyntex.push_texture(TESTURE, options);
        let sprite = dyntex.push_sprite(&testure, Sprite::default());
        dyntex.set_rotation(&sprite, Rad(0.3));

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "set_single_sprite_rotation", img);
    }

    #[test]
    fn push_and_pop_often_avoid_allocating_out_of_bounds() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let options = TextureOptions::default();
        let testure = windowing.dyntex().push_texture(TESTURE, options);

        let mut dyntex = windowing.dyntex();
        for _ in 0..100_000 {
            let sprite = dyntex.push_sprite(&testure, Sprite::default());
            dyntex.remove_sprite(sprite);
        }

        draw_frame(&mut windowing, &prspect);
    }

    #[bench]
    fn bench_many_sprites(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let tex = windowing
            .dyntex()
            .push_texture(LOGO, TextureOptions::default());
        for i in 0..1000 {
            windowing.dyntex().push_sprite(
                &tex,
                Sprite {
                    rotation: ((i * 10) as f32 / 180f32 * PI),
                    scale: 0.5,
                    ..Sprite::default()
                },
            );
        }

        let prspect = gen_perspective(&windowing);
        b.iter(|| {
            draw_frame(&mut windowing, &prspect);
        });
    }

    #[bench]
    fn bench_many_particles(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let tex = windowing
            .dyntex()
            .push_texture(LOGO, TextureOptions::default());
        let mut rng = random::new(0);
        for i in 0..1000 {
            let (dx, dy) = (
                rng.gen_range(-1.0f32, 1.0f32),
                rng.gen_range(-1.0f32, 1.0f32),
            );
            windowing.dyntex().push_sprite(
                &tex,
                Sprite {
                    translation: (dx, dy),
                    rotation: ((i * 10) as f32 / 180f32 * PI),
                    scale: 0.01,
                    ..Sprite::default()
                },
            );
        }

        let prspect = gen_perspective(&windowing);
        b.iter(|| {
            draw_frame(&mut windowing, &prspect);
        });
    }

    #[bench]
    fn animated_fireballs_20x20_uvs2(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let fireball_texture = windowing.dyntex().push_texture(
            FIREBALL,
            TextureOptions {
                depth_test: false,
                ..TextureOptions::default()
            },
        );

        let mut fireballs = vec![];
        for idx in -10..10 {
            for jdx in -10..10 {
                fireballs.push(windowing.dyntex().push_sprite(
                    &fireball_texture,
                    Sprite {
                        width: 0.68,
                        height: 0.09,
                        rotation: idx as f32 / 18.0 + jdx as f32 / 16.0,
                        translation: (idx as f32 / 10.0, jdx as f32 / 10.0),
                        ..Sprite::default()
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

            windowing
                .dyntex()
                .set_uvs2(fireballs.iter().map(|id| (id, uv_begin, uv_end)));
            draw_frame(&mut windowing, &prspect);
        });
    }

    #[bench]
    fn bench_push_and_pop_sprite(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);

        let options = TextureOptions::default();
        let testure = windowing.dyntex().push_texture(TESTURE, options);

        let mut dyntex = windowing.dyntex();
        b.iter(|| {
            let sprite = dyntex.push_sprite(&testure, Sprite::default());
            dyntex.remove_sprite(sprite);
        });
    }

    #[bench]
    fn bench_push_and_pop_texture(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let mut dyntex = windowing.dyntex();

        b.iter(|| {
            let options = TextureOptions::default();
            let testure = dyntex.push_texture(TESTURE, options);
            dyntex.remove_texture(testure);
        });
    }
}
