//! Methods and types to control debug triangles
//!
//! A debug triangle is a triangle that ignores all transformations and is always shown on the
//! screen (except when a triangle's coordinates are outisde the screen). Debug triangles are meant
//! to be used to quickly find out if a state has been reached (for instance, change the color of a
//! debug triangle if collision is detected).
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
use std::mem::{size_of, transmute, ManuallyDrop};

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

    /// Add a new debug triangle to the renderer
    ///
    /// The new triangle will be drawn upon the next draw.
    pub fn push(&mut self, triangle: DebugTriangle) -> Handle {
        let s = &mut *self.vx;
        let debtris = &mut s.debtris;
        let overrun = (debtris.triangles_count + 1) * TRI_BYTE_SIZE > debtris.capacity as usize;
        if overrun {
            // TODO Do reallocation here
            assert_eq![false, overrun];
        }
        let device = &s.device;
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer(&debtris.triangles_memory, 0..debtris.capacity)
                .expect("Failed to acquire a memory writer!");
            let idx = debtris.triangles_count * TRI_BYTE_SIZE / size_of::<f32>();

            for (i, idx) in [idx, idx + 7, idx + 14].iter().enumerate() {
                data_target[*idx..*idx + 2]
                    .copy_from_slice(&[triangle.origin[i].0, triangle.origin[i].1]);
                data_target[*idx + 2..*idx + 3].copy_from_slice(&transmute::<[u8; 4], [f32; 1]>([
                    triangle.colors_rgba[i].0,
                    triangle.colors_rgba[i].1,
                    triangle.colors_rgba[i].2,
                    triangle.colors_rgba[i].3,
                ]));
                data_target[*idx + 3..*idx + 5]
                    .copy_from_slice(&[triangle.translation.0, triangle.translation.1]);
                data_target[*idx + 5..*idx + 6].copy_from_slice(&[triangle.rotation]);
                data_target[*idx + 6..*idx + 7].copy_from_slice(&[triangle.scale]);
            }
            debtris.triangles_count += 1;
            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
        Handle(debtris.triangles_count - 1)
    }

    /// Remove the last added debug triangle from rendering
    ///
    /// Has no effect if there are no debug triangles.
    pub fn pop(&mut self) {
        let vx = &mut *self.vx;
        let debtris = &mut vx.debtris;
        // TODO deallocate and move the buffer if it's small enough
        unsafe {
            vx.device
                .wait_for_fences(
                    &vx.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        debtris.triangles_count = debtris.triangles_count.checked_sub(1).unwrap_or(0);
    }

    /// Remove the last N added debug triangle from rendering
    ///
    /// If the amount to pop is bigger than the amount of debug triangles, then all debug triangles
    /// wil be removed.
    pub fn pop_many(&mut self, n: usize) {
        unsafe {
            self.vx
                .device
                .wait_for_fences(
                    &self.vx.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        self.vx.debtris.triangles_count =
            self.vx.debtris.triangles_count.checked_sub(n).unwrap_or(0);
    }

    // ---

    /// Set the position of a debug triangle
    pub fn set_position(&mut self, inst: &Handle, pos: (f32, f32)) {
        let vx = &mut *self.vx;
        let inst = inst.0;
        let device = &vx.device;
        let debtris = &mut vx.debtris;
        unsafe {
            device
                .wait_for_fences(
                    &vx.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer::<f32>(&debtris.triangles_memory, 0..debtris.capacity)
                .expect("Failed to acquire a memory writer!");

            let mut idx = inst * TRI_BYTE_SIZE / size_of::<f32>();
            data_target[idx + 3..idx + 5].copy_from_slice(&[pos.0, pos.1]);
            idx += 7;
            data_target[idx + 3..idx + 5].copy_from_slice(&[pos.0, pos.1]);
            idx += 7;
            data_target[idx + 3..idx + 5].copy_from_slice(&[pos.0, pos.1]);
            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
    }

    /// Set the scale of a debug triangle
    pub fn set_scale(&mut self, inst: &Handle, scale: f32) {
        let vx = &mut *self.vx;
        let inst = inst.0;
        let device = &vx.device;
        let debtris = &mut vx.debtris;
        unsafe {
            device
                .wait_for_fences(
                    &vx.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer::<f32>(&debtris.triangles_memory, 0..debtris.capacity)
                .expect("Failed to acquire a memory writer!");

            let mut idx = inst * TRI_BYTE_SIZE / size_of::<f32>();
            data_target[idx + 6..idx + 7].copy_from_slice(&[scale]);
            idx += 7;
            data_target[idx + 6..idx + 7].copy_from_slice(&[scale]);
            idx += 7;
            data_target[idx + 6..idx + 7].copy_from_slice(&[scale]);
            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
    }

    /// Set the rotation of a debug triangle
    pub fn set_rotation<T: Copy + Into<Rad<f32>>>(&mut self, inst: &Handle, deg: T) {
        let vx = &mut *self.vx;
        let inst = inst.0;
        let device = &vx.device;
        let debtris = &mut vx.debtris;
        unsafe {
            device
                .wait_for_fences(
                    &vx.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer::<f32>(&debtris.triangles_memory, 0..debtris.capacity)
                .expect("Failed to acquire a memory writer!");

            let mut idx = inst * TRI_BYTE_SIZE / size_of::<f32>();
            let rot = deg.into().0;
            data_target[idx + 5..idx + 6].copy_from_slice(&[rot]);
            idx += 7;
            data_target[idx + 5..idx + 6].copy_from_slice(&[rot]);
            idx += 7;
            data_target[idx + 5..idx + 6].copy_from_slice(&[rot]);
            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
    }

    /// Set a solid color of a debug triangle
    pub fn set_color(&mut self, inst: &Handle, rgba: [u8; 4]) {
        let vx = &mut *self.vx;
        let inst = inst.0;
        let device = &vx.device;
        let debtris = &mut vx.debtris;
        unsafe {
            device
                .wait_for_fences(
                    &vx.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer::<f32>(&debtris.triangles_memory, 0..debtris.capacity)
                .expect("Failed to acquire a memory writer!");

            let mut idx = inst * TRI_BYTE_SIZE / size_of::<f32>();
            let rgba = &transmute::<[u8; 4], [f32; 1]>(rgba);
            data_target[idx + 2..idx + 3].copy_from_slice(rgba);
            idx += 7;
            data_target[idx + 2..idx + 3].copy_from_slice(rgba);
            idx += 7;
            data_target[idx + 2..idx + 3].copy_from_slice(rgba);
            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
    }

    // ---

    /// Rotate all debug triangles
    pub fn rotate<T: Copy + Into<Rad<f32>>>(&mut self, handle: &Handle, deg: T) {
        let vx = &mut *self.vx;
        let device = &vx.device;
        let debtris = &mut vx.debtris;
        unsafe {
            device
                .wait_for_fences(
                    &vx.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        unsafe {
            let data_reader = device
                .acquire_mapping_reader::<f32>(&debtris.triangles_memory, 0..debtris.capacity)
                .expect("Failed to acquire a memory writer!");

            let idx = handle.0 * TRI_BYTE_SIZE / size_of::<f32>();
            let rotation = &data_reader[idx + 5..idx + 6];
            let rotation = rotation[0];

            device.release_mapping_reader(data_reader);

            let mut data_target = device
                .acquire_mapping_writer::<f32>(&debtris.triangles_memory, 0..debtris.capacity)
                .expect("Failed to acquire a memory writer!");

            let mut idx = handle.0 * TRI_BYTE_SIZE / size_of::<f32>();
            data_target[idx + 5..idx + 6].copy_from_slice(&[rotation + deg.into().0]);
            idx += 7;
            data_target[idx + 5..idx + 6].copy_from_slice(&[rotation + deg.into().0]);
            idx += 7;
            data_target[idx + 5..idx + 6].copy_from_slice(&[rotation + deg.into().0]);

            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
    }

    /// Rotate all debug triangles
    pub fn rotate_all<T: Copy + Into<Rad<f32>>>(&mut self, deg: T) {
        let vx = &mut *self.vx;
        let device = &vx.device;
        let debtris = &mut vx.debtris;
        unsafe {
            device
                .wait_for_fences(
                    &vx.frames_in_flight_fences,
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for fences");
        }
        unsafe {
            let data_reader = device
                .acquire_mapping_reader::<f32>(&debtris.triangles_memory, 0..debtris.capacity)
                .expect("Failed to acquire a memory writer!");
            let mut vertices = Vec::<f32>::with_capacity(debtris.triangles_count);
            for i in 0..debtris.triangles_count {
                let idx = i * TRI_BYTE_SIZE / size_of::<f32>();
                let rotation = &data_reader[idx + 5..idx + 6];
                vertices.push(rotation[0]);
            }
            device.release_mapping_reader(data_reader);

            let mut data_target = device
                .acquire_mapping_writer::<f32>(&debtris.triangles_memory, 0..debtris.capacity)
                .expect("Failed to acquire a memory writer!");

            for (i, vert) in vertices.iter().enumerate() {
                let mut idx = i * TRI_BYTE_SIZE / size_of::<f32>();
                data_target[idx + 5..idx + 6].copy_from_slice(&[*vert + deg.into().0]);
                idx += 7;
                data_target[idx + 5..idx + 6].copy_from_slice(&[*vert + deg.into().0]);
                idx += 7;
                data_target[idx + 5..idx + 6].copy_from_slice(&[*vert + deg.into().0]);
            }
            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
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

const PTS_PER_TRI: usize = 3;
const CART_CMPNTS: usize = 2;
const COLOR_CMPNTS: usize = 4;
const DELTA_CMPNTS: usize = 2;
const ROT_CMPNTS: usize = 1;
const SCALE_CMPNTS: usize = 1;
const BYTES_PER_VTX: usize = size_of::<f32>() * CART_CMPNTS
    + size_of::<u8>() * COLOR_CMPNTS
    + size_of::<f32>() * DELTA_CMPNTS
    + size_of::<f32>() * ROT_CMPNTS
    + size_of::<f32>() * SCALE_CMPNTS;
const TRI_BYTE_SIZE: usize = PTS_PER_TRI * BYTES_PER_VTX;

// ---

pub fn create_debug_triangle(
    device: &back::Device,
    adapter: &Adapter<back::Backend>,
    format: &format::Format,
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

    let vertex_buffers = vec![pso::VertexBufferDesc {
        binding: 0,
        stride: BYTES_PER_VTX as u32,
        rate: pso::VertexInputRate::Vertex,
    }];

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
            binding: 0,
            element: pso::Element {
                format: format::Format::Rgba8Unorm,
                offset: 8,
            },
        },
        pso::AttributeDesc {
            location: 2,
            binding: 0,
            element: pso::Element {
                format: format::Format::Rg32Sfloat,
                offset: 12,
            },
        },
        pso::AttributeDesc {
            location: 3,
            binding: 0,
            element: pso::Element {
                format: format::Format::R32Sfloat,
                offset: 20,
            },
        },
        pso::AttributeDesc {
            location: 4,
            binding: 0,
            element: pso::Element {
                format: format::Format::R32Sfloat,
                offset: 24,
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
            format: Some(*format),
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

    let (dtbuffer, dtmemory, dtreqs) =
        make_vertex_buffer_with_data2(device, adapter, &[0.0f32; TRI_BYTE_SIZE / 4 * 1000]);

    let debtris = DebugTriangleData {
        hidden: false,
        capacity: dtreqs.size,
        triangles_count: 0,
        triangles_buffer: ManuallyDrop::new(dtbuffer),
        triangles_memory: ManuallyDrop::new(dtmemory),
        memory_requirements: dtreqs,

        descriptor_set: triangle_descriptor_set_layouts,
        pipeline: ManuallyDrop::new(triangle_pipeline),
        pipeline_layout: ManuallyDrop::new(triangle_pipeline_layout),
        render_pass: ManuallyDrop::new(triangle_render_pass),
    };
    debtris
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
    fn test_single_triangle_api() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&vx);
        let tri = DebugTriangle::default();

        let mut debtri = vx.debtri();
        let handle = debtri.push(tri);
        debtri.set_scale(&handle, 0.1);
        debtri.set_rotation(&handle, Deg(25.0));
        debtri.set_position(&handle, (0.25, 0.5));
        debtri.rotate(&handle, Deg(5.0));

        let img = vx.draw_frame_copy_framebuffer(&prspect);
        utils::assert_swapchain_eq(&mut vx, "test_single_triangle_api", img);
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
    fn bench_rotating_windmills_no_render(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        utils::add_windmills(&mut vx, false);

        b.iter(|| {
            vx.debtri().rotate_all(Deg(1.0f32));
        });
    }
}
