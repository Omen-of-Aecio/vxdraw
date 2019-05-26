use super::utils::*;
use crate::data::{ColoredQuadList, Windowing};
use cgmath::Rad;
#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
use gfx_backend_gl as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use gfx_hal::{device::Device, format, image, pass, pso, Backend, Primitive};
use std::io::Read;
use std::mem::{size_of, transmute, ManuallyDrop};

// ---

pub struct QuadHandle(usize);

#[derive(Clone, Copy)]
pub struct Quad {
    pub width: f32,
    pub height: f32,
    pub depth: f32,
    pub colors: [(u8, u8, u8, u8); 4],
    pub translation: (f32, f32),
    pub rotation: f32,
    pub scale: f32,
    /// Moves the origin of the quad to some point, for instance, you may want a corner of the quad
    /// to be the origin. This affects rotation and translation of the quad.
    pub origin: (f32, f32),
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

const PTS_PER_QUAD: usize = 4;
const CART_CMPNTS: usize = 3;
const COLOR_CMPNTS: usize = 4;
const DELTA_CMPNTS: usize = 2;
const ROT_CMPNTS: usize = 1;
const SCALE_CMPNTS: usize = 1;
const BYTES_PER_VTX: usize = size_of::<f32>() * CART_CMPNTS
    + size_of::<u8>() * COLOR_CMPNTS
    + size_of::<f32>() * DELTA_CMPNTS
    + size_of::<f32>() * ROT_CMPNTS
    + size_of::<f32>() * SCALE_CMPNTS;
const QUAD_BYTE_SIZE: usize = PTS_PER_QUAD * BYTES_PER_VTX;

// ---

pub fn push(s: &mut Windowing, quad: Quad) -> QuadHandle {
    let overrun = if let Some(ref mut quads) = s.quads {
        Some((quads.count + 1) * QUAD_BYTE_SIZE > quads.capacity as usize)
    } else {
        None
    };
    if let Some(overrun) = overrun {
        // Do reallocation here
        assert_eq![false, overrun];
    }
    if let Some(ref mut quads) = s.quads {
        let device = &s.device;

        let width = quad.width;
        let height = quad.height;

        let topleft = (
            -width / 2f32 - quad.origin.0,
            -height / 2f32 - quad.origin.1,
            quad.depth,
        );
        let topright = (
            width / 2f32 - quad.origin.0,
            -height / 2f32 - quad.origin.1,
            quad.depth,
        );
        let bottomleft = (
            -width / 2f32 - quad.origin.0,
            height / 2f32 - quad.origin.1,
            quad.depth,
        );
        let bottomright = (
            width / 2f32 - quad.origin.0,
            height / 2f32 - quad.origin.1,
            quad.depth,
        );

        unsafe {
            let mut data_target = device
                .acquire_mapping_writer(
                    &quads.quads_memory_indices,
                    0..quads.quads_requirements_indices.size,
                )
                .expect("Failed to acquire a memory writer!");
            let ver = (quads.count * 6) as u16;
            let ind = (quads.count * 4) as u16;
            data_target[ver as usize..(ver + 6) as usize].copy_from_slice(&[
                ind,
                ind + 1,
                ind + 2,
                ind + 2,
                ind + 3,
                ind,
            ]);
            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
            let mut data_target = device
                .acquire_mapping_writer(&quads.quads_memory, 0..quads.memory_requirements.size)
                .expect("Failed to acquire a memory writer!");

            for (i, point) in [topleft, bottomleft, bottomright, topright]
                .iter()
                .enumerate()
            {
                let idx = (i + quads.count * 4) * 8;

                data_target[idx..idx + 3].copy_from_slice(&[point.0, point.1, point.2]);
                data_target[idx + 3..idx + 4].copy_from_slice(&transmute::<[u8; 4], [f32; 1]>([
                    quad.colors[i].0,
                    quad.colors[i].1,
                    quad.colors[i].2,
                    quad.colors[i].3,
                ]));
                data_target[idx + 4..idx + 6]
                    .copy_from_slice(&[quad.translation.0, quad.translation.1]);
                data_target[idx + 6..idx + 7].copy_from_slice(&[quad.rotation]);
                data_target[idx + 7..idx + 8].copy_from_slice(&[quad.scale]);
            }
            quads.count += 1;
            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
        QuadHandle(quads.count - 1)
    } else {
        unreachable![]
    }
}

pub fn quad_pop(s: &mut Windowing) {
    if let Some(ref mut quads) = s.quads {
        unsafe {
            s.device
                .wait_for_fences(
                    &s.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        quads.count -= 1;
    }
}

pub fn pop_n_quads(s: &mut Windowing, n: usize) {
    if let Some(ref mut quads) = s.quads {
        unsafe {
            s.device
                .wait_for_fences(
                    &s.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        quads.count -= n;
    }
}

pub fn create_quad(s: &mut Windowing) {
    pub const VERTEX_SOURCE: &[u8] = include_bytes!["../_build/spirv/quads.vert.spirv"];

    pub const FRAGMENT_SOURCE: &[u8] = include_bytes!["../_build/spirv/quads.frag.spirv"];

    let vs_module = { unsafe { s.device.create_shader_module(&VERTEX_SOURCE) }.unwrap() };
    let fs_module = { unsafe { s.device.create_shader_module(&FRAGMENT_SOURCE) }.unwrap() };

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
        stride: BYTES_PER_VTX as u32,
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
                format: format::Format::Rgba8Unorm,
                offset: 12,
            },
        },
        pso::AttributeDesc {
            location: 2,
            binding: 0,
            element: pso::Element {
                format: format::Format::Rg32Sfloat,
                offset: 16,
            },
        },
        pso::AttributeDesc {
            location: 3,
            binding: 0,
            element: pso::Element {
                format: format::Format::R32Sfloat,
                offset: 24,
            },
        },
        pso::AttributeDesc {
            location: 4,
            binding: 0,
            element: pso::Element {
                format: format::Format::R32Sfloat,
                offset: 28,
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
        depth: pso::DepthTest::On {
            fun: pso::Comparison::LessEqual,
            write: true,
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
    }
    unsafe {
        s.device.destroy_shader_module(fs_module);
    }
    let (dtbuffer, dtmemory, dtreqs) =
        make_vertex_buffer_with_data(s, &[0.0f32; QUAD_BYTE_SIZE / 4 * 1000]);

    let (quads_buffer_indices, quads_memory_indices, quads_requirements_indices) =
        make_index_buffer_with_data(s, &[0f32; 4 * 1000]);

    let quads = ColoredQuadList {
        capacity: dtreqs.size,
        count: 0,
        quads_buffer: dtbuffer,
        quads_memory: dtmemory,
        memory_requirements: dtreqs,

        quads_buffer_indices,
        quads_memory_indices,
        quads_requirements_indices,

        descriptor_set: quad_descriptor_set_layouts,
        pipeline: ManuallyDrop::new(quad_pipeline),
        pipeline_layout: ManuallyDrop::new(quad_pipeline_layout),
        render_pass: ManuallyDrop::new(quad_render_pass),
    };
    s.quads = Some(quads);
}

pub fn translate(s: &mut Windowing, handle: &QuadHandle, movement: (f32, f32)) {
    let device = &s.device;
    if let Some(ref mut quads) = s.quads {
        unsafe {
            let aligned = perfect_mapping_alignment(Align {
                access_offset: handle.0 as u64 * QUAD_BYTE_SIZE as u64,
                how_many_bytes_you_need: QUAD_BYTE_SIZE as u64,
                non_coherent_atom_size: s.device_limits.non_coherent_atom_size as u64,
                memory_size: quads.memory_requirements.size,
            });

            let data_reader = device
                .acquire_mapping_reader::<u8>(&quads.quads_memory, aligned.begin..aligned.end)
                .expect("Failed to acquire a memory writer!");
            let dxu = &data_reader[aligned.index_offset + 16..aligned.index_offset + 20];
            let dyu = &data_reader[aligned.index_offset + 20..aligned.index_offset + 24];
            let dx = transmute::<f32, [u8; 4]>(
                movement.0 + transmute::<[u8; 4], f32>([dxu[0], dxu[1], dxu[2], dxu[3]]),
            );
            let dy = transmute::<f32, [u8; 4]>(
                movement.1 + transmute::<[u8; 4], f32>([dyu[0], dyu[1], dyu[2], dyu[3]]),
            );
            device.release_mapping_reader(data_reader);

            device
                .wait_for_fences(
                    &s.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");

            let mut data_target = device
                .acquire_mapping_writer::<u8>(&quads.quads_memory, aligned.begin..aligned.end)
                .expect("Failed to acquire a memory writer!");

            let mut idx = aligned.index_offset;
            data_target[idx + 16..idx + 20].copy_from_slice(&dx);
            data_target[idx + 20..idx + 24].copy_from_slice(&dy);
            idx += BYTES_PER_VTX;
            data_target[idx + 16..idx + 20].copy_from_slice(&dx);
            data_target[idx + 20..idx + 24].copy_from_slice(&dy);
            idx += BYTES_PER_VTX;
            data_target[idx + 16..idx + 20].copy_from_slice(&dx);
            data_target[idx + 20..idx + 24].copy_from_slice(&dy);
            idx += BYTES_PER_VTX;
            data_target[idx + 16..idx + 20].copy_from_slice(&dx);
            data_target[idx + 20..idx + 24].copy_from_slice(&dy);

            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
    }
}

pub fn set_position(s: &mut Windowing, handle: &QuadHandle, position: (f32, f32)) {
    let device = &s.device;
    if let Some(ref mut quads) = s.quads {
        unsafe {
            let aligned = perfect_mapping_alignment(Align {
                access_offset: handle.0 as u64 * QUAD_BYTE_SIZE as u64,
                how_many_bytes_you_need: QUAD_BYTE_SIZE as u64,
                non_coherent_atom_size: s.device_limits.non_coherent_atom_size as u64,
                memory_size: quads.memory_requirements.size,
            });

            device
                .wait_for_fences(
                    &s.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");

            let mut data_target = device
                .acquire_mapping_writer::<u8>(&quads.quads_memory, aligned.begin..aligned.end)
                .expect("Failed to acquire a memory writer!");

            let mut idx = aligned.index_offset;
            let x = &transmute::<f32, [u8; 4]>(position.0);
            let y = &transmute::<f32, [u8; 4]>(position.1);
            data_target[idx + 16..idx + 20].copy_from_slice(x);
            data_target[idx + 20..idx + 24].copy_from_slice(y);
            idx += BYTES_PER_VTX;
            data_target[idx + 16..idx + 20].copy_from_slice(x);
            data_target[idx + 20..idx + 24].copy_from_slice(y);
            idx += BYTES_PER_VTX;
            data_target[idx + 16..idx + 20].copy_from_slice(x);
            data_target[idx + 20..idx + 24].copy_from_slice(y);
            idx += BYTES_PER_VTX;
            data_target[idx + 16..idx + 20].copy_from_slice(x);
            data_target[idx + 20..idx + 24].copy_from_slice(y);

            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
    }
}

pub fn quad_rotate_all<T: Copy + Into<Rad<f32>>>(s: &mut Windowing, deg: T) {
    let device = &s.device;
    if let Some(ref mut quads) = s.quads {
        unsafe {
            device
                .wait_for_fences(
                    &s.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        unsafe {
            let data_reader = device
                .acquire_mapping_reader::<f32>(&quads.quads_memory, 0..quads.capacity)
                .expect("Failed to acquire a memory writer!");
            let mut vertices = Vec::<f32>::with_capacity(quads.count);
            for i in 0..quads.count {
                let idx = i * QUAD_BYTE_SIZE / size_of::<f32>();
                let rotation = &data_reader[idx + 6..idx + 7];
                vertices.push(rotation[0]);
            }
            device.release_mapping_reader(data_reader);

            let mut data_target = device
                .acquire_mapping_writer::<f32>(&quads.quads_memory, 0..quads.capacity)
                .expect("Failed to acquire a memory writer!");

            for (i, vert) in vertices.iter().enumerate() {
                let mut idx = i * QUAD_BYTE_SIZE / size_of::<f32>();
                data_target[idx + 6..idx + 7].copy_from_slice(&[*vert + deg.into().0]);
                idx += BYTES_PER_VTX / 4;
                data_target[idx + 6..idx + 7].copy_from_slice(&[*vert + deg.into().0]);
                idx += BYTES_PER_VTX / 4;
                data_target[idx + 6..idx + 7].copy_from_slice(&[*vert + deg.into().0]);
                idx += BYTES_PER_VTX / 4;
                data_target[idx + 6..idx + 7].copy_from_slice(&[*vert + deg.into().0]);
            }
            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
    }
}

pub fn set_quad_color(s: &mut Windowing, inst: &QuadHandle, rgba: [u8; 4]) {
    let inst = inst.0;
    let device = &s.device;
    if let Some(ref mut quads) = s.quads {
        unsafe {
            device
                .wait_for_fences(
                    &s.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer::<f32>(&quads.quads_memory, 0..quads.capacity)
                .expect("Failed to acquire a memory writer!");

            let mut idx = inst * QUAD_BYTE_SIZE / size_of::<f32>();
            let rgba = &transmute::<[u8; 4], [f32; 1]>(rgba);
            data_target[idx + 3..idx + 4].copy_from_slice(rgba);
            idx += 7;
            data_target[idx + 3..idx + 4].copy_from_slice(rgba);
            idx += 7;
            data_target[idx + 3..idx + 4].copy_from_slice(rgba);
            idx += 7;
            data_target[idx + 3..idx + 4].copy_from_slice(rgba);
            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
    }
}

// ---

#[cfg(feature = "gfx_tests")]
#[cfg(test)]
mod tests {
    use crate::*;
    use cgmath::Deg;
    use logger::{Generic, GenericLogger, Logger};

    #[test]
    fn simple_quad() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let mut quad = quads::Quad::default();
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        quads::push(&mut windowing, quad);

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "simple_quad", img);
    }

    #[test]
    fn simple_quad_translated() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let mut quad = quads::Quad::default();
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        let handle = quads::push(&mut windowing, quad);
        quads::translate(&mut windowing, &handle, (0.25, 0.4));

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "simple_quad_translated", img);
    }

    #[test]
    fn simple_quad_rotated_with_exotic_origin() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let mut quad = quads::Quad::default();
        quad.scale = 0.2;
        quad.colors[0].0 = 255;
        quad.colors[3].0 = 255;
        quads::push(&mut windowing, quad);

        let mut quad = quads::Quad::default();
        quad.scale = 0.2;
        quad.origin = (-1.0, -1.0);
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;
        quads::push(&mut windowing, quad);

        // when
        quads::quad_rotate_all(&mut windowing, Deg(30.0));

        // then
        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(
            &mut windowing,
            "simple_quad_rotated_with_exotic_origin",
            img,
        );
    }

    #[test]
    fn a_bunch_of_quads() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let mut topright = debtri::DebugTriangle::from([-1.0, -1.0, 1.0, 1.0, 1.0, -1.0]);
        let mut bottomleft = debtri::DebugTriangle::from([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        topright.scale = 0.1;
        bottomleft.scale = 0.1;
        let radi = 0.1;
        let trans = -1f32 + radi;

        for j in 0..10 {
            for i in 0..10 {
                topright.translation =
                    (trans + i as f32 * 2.0 * radi, trans + j as f32 * 2.0 * radi);
                bottomleft.translation =
                    (trans + i as f32 * 2.0 * radi, trans + j as f32 * 2.0 * radi);
                debtri::push(&mut windowing, topright);
                debtri::push(&mut windowing, bottomleft);
            }
        }

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "a_bunch_of_quads", img);
    }

    #[test]
    fn overlapping_quads_respect_z_order() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);
        let mut quad = quads::Quad {
            scale: 0.5,
            ..quads::Quad::default()
        };

        for i in 0..4 {
            quad.colors[i] = (0, 255, 0, 255);
        }
        quad.depth = 0.0;
        quad.translation = (0.25, 0.25);
        quads::push(&mut windowing, quad);

        for i in 0..4 {
            quad.colors[i] = (255, 0, 0, 255);
        }
        quad.depth = 0.5;
        quad.translation = (0.0, 0.0);
        quads::push(&mut windowing, quad);

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "overlapping_quads_respect_z_order", img);

        // ---

        quads::pop_n_quads(&mut windowing, 2);

        // ---

        for i in 0..4 {
            quad.colors[i] = (255, 0, 0, 255);
        }
        quad.depth = 0.5;
        quad.translation = (0.0, 0.0);
        quads::push(&mut windowing, quad);

        for i in 0..4 {
            quad.colors[i] = (0, 255, 0, 255);
        }
        quad.depth = 0.0;
        quad.translation = (0.25, 0.25);
        quads::push(&mut windowing, quad);

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "overlapping_quads_respect_z_order", img);
    }
}
