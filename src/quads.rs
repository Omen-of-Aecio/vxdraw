//! Methods and types to control quads
//!
//! A quad is a renderable consisting of 4 points. Each point has a color and position associated
//! with it. By using different colors in the different points, the colors will "blend" into each
//! other. Opacity is also supported on quads.
use super::utils::*;
use crate::data::{ColoredQuadList, DrawType, VxDraw};
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
use std::mem::{size_of, transmute, ManuallyDrop};

pub struct Quads<'a> {
    vx: &'a mut VxDraw,
}

impl<'a> Quads<'a> {
    pub fn new(vx: &'a mut VxDraw) -> Self {
        Self { vx }
    }

    pub fn push(&mut self, layer: &Layer, quad: Quad) -> QuadHandle {
        if let Some(ref mut quads) = self.vx.quads.get_mut(layer.0) {
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

            quads.posbuf_touch = self.vx.swapconfig.image_count;
            quads.colbuf_touch = self.vx.swapconfig.image_count;
            quads.tranbuf_touch = self.vx.swapconfig.image_count;
            quads.rotbuf_touch = self.vx.swapconfig.image_count;
            quads.scalebuf_touch = self.vx.swapconfig.image_count;

            quads.count += 1;

            QuadHandle(layer.0, quads.count - 1)
        } else {
            unreachable![]
        }
    }

    pub fn quad_pop(&mut self, layer: &Layer) {
        if let Some(ref mut quads) = self.vx.quads.get_mut(layer.0) {
            quads.count -= 1;
        }
    }

    pub fn pop_n_quads(&mut self, layer: &Layer, n: usize) {
        if let Some(ref mut quads) = self.vx.quads.get_mut(layer.0) {
            quads.count -= n;
        }
    }

    pub fn create_quad(&mut self, options: QuadOptions) -> Layer {
        let s = &mut *self.vx;
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

        let indices = super::utils::ResizBufIdx4::new(&s.device, &s.adapter);

        let quads = ColoredQuadList {
            count: 0,

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
        s.quads.push(quads);
        s.draw_order.push(DrawType::Quad {
            id: s.quads.len() - 1,
        });
        Layer(s.quads.len() - 1)
    }

    pub fn translate(&mut self, handle: &QuadHandle, movement: (f32, f32)) {
        self.vx.debtris.tranbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            self.vx.quads[handle.0].tranbuffer[handle.1][idx * 2] += movement.0;
            self.vx.quads[handle.0].tranbuffer[handle.1][idx * 2 + 1] += movement.1;
        }
    }

    pub fn set_position(&mut self, handle: &QuadHandle, position: (f32, f32)) {
        self.vx.debtris.tranbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            self.vx.quads[handle.0].tranbuffer[handle.1][idx * 2..(idx + 1) * 2]
                .copy_from_slice(&[position.0, position.1]);
        }
    }

    pub fn quad_rotate_all<T: Copy + Into<Rad<f32>>>(&mut self, layer: &Layer, deg: T) {
        self.vx.debtris.rotbuf_touch = self.vx.swapconfig.image_count;
        for rot in self.vx.quads[layer.0].rotbuffer.iter_mut() {
            rot[0] += deg.into().0;
            rot[1] += deg.into().0;
            rot[2] += deg.into().0;
            rot[3] += deg.into().0;
        }
    }

    pub fn set_quad_color(&mut self, handle: &QuadHandle, rgba: [u8; 4]) {
        self.vx.debtris.colbuf_touch = self.vx.swapconfig.image_count;
        for idx in 0..4 {
            self.vx.quads[handle.0].colbuffer[handle.0][idx * 4..(idx + 1) * 4]
                .copy_from_slice(&rgba);
        }
    }
}

// ---

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

pub struct QuadHandle(usize, usize);

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

pub struct QuadOptions {
    depth_test: bool,
}

impl Default for QuadOptions {
    fn default() -> Self {
        Self { depth_test: false }
    }
}

// ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use cgmath::Deg;
    use logger::{Generic, GenericLogger, Logger};

    #[test]
    fn simple_quad() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut quad = quads::Quad::default();
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        let layer = vx.quads().create_quad(QuadOptions::default());
        vx.quads().push(&layer, quad);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "simple_quad", img);
    }

    #[test]
    fn simple_quad_translated() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut quad = quads::Quad::default();
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        let mut quads = vx.quads();
        let layer = quads.create_quad(QuadOptions::default());
        let handle = quads.push(&layer, quad);
        quads.translate(&handle, (0.25, 0.4));

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "simple_quad_translated", img);
    }

    #[test]
    fn simple_quad_rotated_with_exotic_origin() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

        let mut quad = quads::Quad::default();
        quad.scale = 0.2;
        quad.colors[0].0 = 255;
        quad.colors[3].0 = 255;

        let layer = vx.quads().create_quad(QuadOptions::default());
        vx.quads().push(&layer, quad);

        let mut quad = quads::Quad::default();
        quad.scale = 0.2;
        quad.origin = (-1.0, -1.0);
        quad.colors[0].1 = 255;
        quad.colors[3].1 = 255;

        let mut quads = vx.quads();
        quads.push(&layer, quad);

        // when
        quads.quad_rotate_all(&layer, Deg(30.0));

        // then
        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "simple_quad_rotated_with_exotic_origin", img);
    }

    #[test]
    fn a_bunch_of_quads() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);

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
                vx.debtri().push(topright);
                vx.debtri().push(bottomleft);
            }
        }

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "a_bunch_of_quads", img);
    }

    // DISABLED because we might disable depth buffering altogether
    // #[test]
    // fn overlapping_quads_respect_z_order() {
    //     let logger = Logger::<Generic>::spawn_void().to_logpass();
    //     let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
    //     let prspect = gen_perspective(&vx);
    //     let mut quad = quads::Quad {
    //         scale: 0.5,
    //         ..quads::Quad::default()
    //     };

    //     for i in 0..4 {
    //         quad.colors[i] = (0, 255, 0, 255);
    //     }
    //     quad.depth = 0.0;
    //     quad.translation = (0.25, 0.25);

    //     let layer = vx.quads().create_quad(QuadOptions {
    //         depth_test: true,
    //         ..QuadOptions::default()
    //     });
    //     vx.quads().push(&layer, quad);

    //     for i in 0..4 {
    //         quad.colors[i] = (255, 0, 0, 255);
    //     }
    //     quad.depth = 0.5;
    //     quad.translation = (0.0, 0.0);
    //     vx.quads().push(&layer, quad);

    //     let img = vx.draw_frame_copy_framebuffer(&prspect);
    //     utils::assert_swapchain_eq(&mut vx, "overlapping_quads_respect_z_order", img);

    //     // ---

    //     vx.quads().pop_n_quads(&layer, 2);

    //     // ---

    //     for i in 0..4 {
    //         quad.colors[i] = (255, 0, 0, 255);
    //     }
    //     quad.depth = 0.5;
    //     quad.translation = (0.0, 0.0);
    //     vx.quads().push(&layer, quad);

    //     for i in 0..4 {
    //         quad.colors[i] = (0, 255, 0, 255);
    //     }
    //     quad.depth = 0.0;
    //     quad.translation = (0.25, 0.25);
    //     vx.quads().push(&layer, quad);

    //     let img = vx.draw_frame_copy_framebuffer(&prspect);
    //     utils::assert_swapchain_eq(&mut vx, "overlapping_quads_respect_z_order", img);
    // }

    #[test]
    fn quad_layering() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);
        let mut quad = quads::Quad {
            scale: 0.5,
            ..quads::Quad::default()
        };

        for i in 0..4 {
            quad.colors[i] = (0, 255, 0, 255);
        }
        quad.depth = 0.0;
        quad.translation = (0.25, 0.25);

        let layer1 = vx.quads().create_quad(QuadOptions::default());
        let layer2 = vx.quads().create_quad(QuadOptions::default());

        vx.quads().push(&layer2, quad);

        quad.scale = 0.6;
        for i in 0..4 {
            quad.colors[i] = (0, 0, 255, 255);
        }
        vx.quads().push(&layer1, quad);

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "quad_layering", img);
    }
}
