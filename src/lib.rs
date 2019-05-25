#![feature(test)]
extern crate test;

use crate::data::{DrawType, Windowing};
#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
use gfx_backend_gl as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
// use gfx_hal::format::{AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle};
use ::image as load_image;
use arrayvec::ArrayVec;
use cgmath::prelude::*;
use cgmath::Matrix4;
use gfx_hal::{
    adapter::PhysicalDevice,
    command::{self, ClearColor, ClearValue},
    device::Device,
    format::{self, ChannelType, Swizzle},
    image, memory, pass, pool,
    pso::{
        self, AttributeDesc, BakedStates, BasePipeline, BlendDesc, BlendOp, BlendState,
        ColorBlendDesc, ColorMask, DepthStencilDesc, DepthTest, DescriptorSetLayoutBinding,
        Element, Face, Factor, FrontFace, GraphicsPipelineDesc, InputAssemblerDesc, LogicOp,
        PipelineCreationFlags, PipelineStage, PolygonMode, Rasterizer, Rect, ShaderStageFlags,
        StencilTest, VertexBufferDesc, Viewport,
    },
    queue::Submission,
    window::{Extent2D, PresentMode::*, Surface, Swapchain},
    Backbuffer, Backend, FrameSync, Instance, Primitive, SwapchainConfig,
};
use logger::{debug, info, trace, warn, Generic, InDebug, InDebugPretty, Logger};
use std::io::Read;
use std::iter::once;
use std::mem::{size_of, ManuallyDrop};
use winit::{dpi::LogicalSize, Event, EventsLoop, WindowBuilder};

pub mod data;
pub mod debtri;
pub mod dyntex;
pub mod quads;
pub mod strtex;
pub mod utils;

use utils::*;

// ---

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShowWindow {
    /// Runs vulkan in headless mode (hidden window) with a swapchain of 1000x1000
    Headless1k,
    Headless2x1k,
    Headless1x2k,
    Enable,
}

#[cfg(not(feature = "gl"))]
fn set_window_size(window: &mut winit::Window, show: ShowWindow) -> Extent2D {
    let dpi_factor = window.get_hidpi_factor();
    let (w, h): (u32, u32) = match show {
        ShowWindow::Headless1k => {
            window.set_inner_size(LogicalSize {
                width: 1000f64 / dpi_factor,
                height: 1000f64 / dpi_factor,
            });
            (1000, 1000)
        }
        ShowWindow::Headless2x1k => {
            window.set_inner_size(LogicalSize {
                width: 2000f64 / dpi_factor,
                height: 1000f64 / dpi_factor,
            });
            (2000, 1000)
        }
        ShowWindow::Headless1x2k => {
            window.set_inner_size(LogicalSize {
                width: 1000f64 / dpi_factor,
                height: 2000f64 / dpi_factor,
            });
            (1000, 2000)
        }
        ShowWindow::Enable => window
            .get_inner_size()
            .unwrap()
            .to_physical(dpi_factor)
            .into(),
    };
    Extent2D {
        width: w,
        height: h,
    }
}

#[cfg(feature = "gl")]
fn set_window_size(window: &mut glutin::GlWindow, show: ShowWindow) -> Extent2D {
    let dpi_factor = window.get_hidpi_factor();
    let (w, h): (u32, u32) = match show {
        ShowWindow::Headless1k => {
            window.set_inner_size(LogicalSize {
                width: 1000f64 / dpi_factor,
                height: 1000f64 / dpi_factor,
            });
            (1000, 1000)
        }
        ShowWindow::Headless2x1k => {
            window.set_inner_size(LogicalSize {
                width: 2000f64 / dpi_factor,
                height: 1000f64 / dpi_factor,
            });
            (2000, 1000)
        }
        ShowWindow::Headless1x2k => {
            window.set_inner_size(LogicalSize {
                width: 1000f64 / dpi_factor,
                height: 2000f64 / dpi_factor,
            });
            (1000, 2000)
        }
        ShowWindow::Enable => window
            .get_inner_size()
            .unwrap()
            .to_physical(dpi_factor)
            .into(),
    };
    Extent2D {
        width: w,
        height: h,
    }
}

impl Windowing {
    pub fn get_window_size_in_pixels(&self) -> (u32, u32) {
        (self.swapconfig.extent.width, self.swapconfig.extent.height)
    }

    pub fn get_window_size_in_pixels_float(&self) -> (f32, f32) {
        (
            self.swapconfig.extent.width as f32,
            self.swapconfig.extent.height as f32,
        )
    }
}

pub fn init_window_with_vulkan(log: &mut Logger<Generic>, show: ShowWindow) -> Windowing {
    #[cfg(feature = "gl")]
    static BACKEND: &str = "OpenGL";
    #[cfg(feature = "vulkan")]
    static BACKEND: &str = "Vulkan";
    #[cfg(feature = "metal")]
    static BACKEND: &str = "Metal";
    #[cfg(feature = "dx12")]
    static BACKEND: &str = "Dx12";

    info![log, "vxdraw", "Initializing rendering"; "show" => InDebug(&show), "backend" => BACKEND];

    let events_loop = EventsLoop::new();
    let window_builder = WindowBuilder::new().with_visibility(show == ShowWindow::Enable);

    #[cfg(feature = "gl")]
    let (mut adapters, mut surf, dims) = {
        let mut window = {
            let builder = back::config_context(
                back::glutin::ContextBuilder::new(),
                format::Format::Rgba8Srgb,
                None,
            )
            .with_vsync(true);
            back::glutin::GlWindow::new(window_builder, builder, &events_loop).unwrap()
        };

        set_window_size(&mut window, show);
        let dims = {
            let dpi_factor = window.get_hidpi_factor();
            debug![log, "vxdraw", "Window DPI factor"; "factor" => dpi_factor];
            let (w, h): (u32, u32) = window
                .get_inner_size()
                .unwrap()
                .to_physical(dpi_factor)
                .into();
            Extent2D {
                width: w,
                height: h,
            }
        };

        let surface = back::Surface::from_window(window);
        let adapters = surface.enumerate_adapters();
        (adapters, surface, dims)
    };

    #[cfg(not(feature = "gl"))]
    let (window, vk_inst, mut adapters, mut surf, dims) = {
        let mut window = window_builder.build(&events_loop).unwrap();
        let version = 1;
        let vk_inst = back::Instance::create("renderer", version);
        let surf: <back::Backend as Backend>::Surface = vk_inst.create_surface(&window);
        let adapters = vk_inst.enumerate_adapters();
        let dims = set_window_size(&mut window, show);
        let dpi_factor = window.get_hidpi_factor();
        debug![log, "vxdraw", "Window DPI factor"; "factor" => dpi_factor];
        (window, vk_inst, adapters, surf, dims)
    };

    // ---

    {
        let len = adapters.len();
        debug![log, "vxdraw", "Adapters found"; "count" => len];
    }

    for (idx, adap) in adapters.iter().enumerate() {
        let info = adap.info.clone();
        let limits = adap.physical_device.limits();
        debug![log, "vxdraw", "Adapter found"; "idx" => idx, "info" => InDebugPretty(&info), "device limits" => InDebugPretty(&limits)];
    }

    // TODO Find appropriate adapter, I've never seen a case where we have 2+ adapters, that time
    // will come one day
    let adapter = adapters.remove(0);
    let (device, queue_group) = adapter
        .open_with::<_, gfx_hal::Graphics>(1, |family| surf.supports_queue_family(family))
        .expect("Unable to find device supporting graphics");

    let phys_dev_limits = adapter.physical_device.limits();

    let (caps, formats, present_modes, composite_alpha) =
        surf.compatibility(&adapter.physical_device);

    debug![log, "vxdraw", "Surface capabilities"; "capabilities" => InDebugPretty(&caps); clone caps];
    debug![log, "vxdraw", "Formats available"; "formats" => InDebugPretty(&formats); clone formats];
    debug![log, "vxdraw", "Composition"; "alpha" => InDebugPretty(&composite_alpha); clone composite_alpha];
    let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
        formats
            .iter()
            .find(|format| format.base_format().1 == ChannelType::Srgb)
            .cloned()
            .unwrap_or(formats[0])
    });

    debug![log, "vxdraw", "Format chosen"; "format" => InDebugPretty(&format); clone format];
    debug![log, "vxdraw", "Available present modes"; "modes" => InDebugPretty(&present_modes); clone present_modes];

    // https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/VkPresentModeKHR.html
    // VK_PRESENT_MODE_FIFO_KHR ... This is the only value of presentMode that is required to be supported
    let present_mode = {
        [Mailbox, Fifo, Relaxed, Immediate]
            .iter()
            .cloned()
            .find(|pm| present_modes.contains(pm))
            .ok_or("No PresentMode values specified!")
            .unwrap()
    };
    debug![log, "vxdraw", "Using best possible present mode"; "mode" => InDebug(&present_mode)];

    let image_count = if present_mode == Mailbox {
        (caps.image_count.end - 1)
            .min(3)
            .max(caps.image_count.start)
    } else {
        (caps.image_count.end - 1)
            .min(2)
            .max(caps.image_count.start)
    };
    debug![log, "vxdraw", "Using swapchain images"; "count" => image_count];

    debug![log, "vxdraw", "Swapchain size"; "extent" => InDebug(&dims)];

    let mut swap_config = SwapchainConfig::from_caps(&caps, format, dims);
    swap_config.present_mode = present_mode;
    swap_config.image_count = image_count;
    swap_config.extent = dims;
    if caps.usage.contains(image::Usage::TRANSFER_SRC) {
        swap_config.image_usage |= gfx_hal::image::Usage::TRANSFER_SRC;
    } else {
        warn![
            log,
            "vxdraw", "Surface does not support TRANSFER_SRC, may fail during testing"
        ];
    }

    debug![log, "vxdraw", "Swapchain final configuration"; "swapchain" => InDebugPretty(&swap_config); clone swap_config];

    let (swapchain, backbuffer) =
        unsafe { device.create_swapchain(&mut surf, swap_config.clone(), None) }
            .expect("Unable to create swapchain");

    let backbuffer_string = format!["{:#?}", backbuffer];
    debug![log, "vxdraw", "Backbuffer information"; "backbuffers" => backbuffer_string];

    // NOTE: for curious people, the render_pass, used in both framebuffer creation AND command
    // buffer when drawing, only need to be _compatible_, which means the SAMPLE count and the
    // FORMAT is _the exact same_.
    // Other elements such as attachment load/store methods are irrelevant.
    // https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#renderpass-compatibility
    let render_pass = {
        let color_attachment = pass::Attachment {
            format: Some(format),
            samples: 1,
            ops: pass::AttachmentOps {
                load: pass::AttachmentLoadOp::Clear,
                store: pass::AttachmentStoreOp::Store,
            },
            stencil_ops: pass::AttachmentOps::DONT_CARE,
            layouts: image::Layout::Undefined..image::Layout::Present,
        };
        let depth = pass::Attachment {
            format: Some(format::Format::D32Float),
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
        debug![log, "vxdraw", "Render pass info"; "color attachment" => InDebugPretty(&color_attachment); clone color_attachment];
        unsafe {
            device
                .create_render_pass(&[color_attachment, depth], &[subpass], &[])
                .map_err(|_| "Couldn't create a render pass!")
                .unwrap()
        }
    };

    {
        let rpfmt = format!["{:#?}", render_pass];
        debug![log, "vxdraw", "Created render pass for framebuffers"; "renderpass" => rpfmt];
    }

    let mut depth_images: Vec<<back::Backend as Backend>::Image> = vec![];
    let mut depth_image_views: Vec<<back::Backend as Backend>::ImageView> = vec![];
    let mut depth_image_memories: Vec<<back::Backend as Backend>::Memory> = vec![];
    let mut depth_image_requirements: Vec<memory::Requirements> = vec![];

    let (image_views, framebuffers) = match backbuffer {
        Backbuffer::Images(ref images) => {
            let image_views = images
                .iter()
                .map(|image| unsafe {
                    device
                        .create_image_view(
                            &image,
                            image::ViewKind::D2,
                            format, // MUST be identical to the image's format
                            Swizzle::NO,
                            image::SubresourceRange {
                                aspects: format::Aspects::COLOR,
                                levels: 0..1,
                                layers: 0..1,
                            },
                        )
                        .map_err(|_| "Couldn't create the image_view for the image!")
                })
                .collect::<Result<Vec<_>, &str>>()
                .unwrap();

            unsafe {
                for _ in &image_views {
                    let mut depth_image = device
                        .create_image(
                            image::Kind::D2(dims.width, dims.height, 1, 1),
                            1,
                            format::Format::D32Float,
                            image::Tiling::Optimal,
                            image::Usage::DEPTH_STENCIL_ATTACHMENT,
                            image::ViewCapabilities::empty(),
                        )
                        .expect("Unable to create depth image");
                    let requirements = device.get_image_requirements(&depth_image);
                    let memory_type_id = find_memory_type_id(
                        &adapter,
                        requirements,
                        memory::Properties::DEVICE_LOCAL,
                    );
                    let memory = device
                        .allocate_memory(memory_type_id, requirements.size)
                        .expect("Couldn't allocate image memory!");
                    device
                        .bind_image_memory(&memory, 0, &mut depth_image)
                        .expect("Couldn't bind the image memory!");
                    let image_view = device
                        .create_image_view(
                            &depth_image,
                            image::ViewKind::D2,
                            format::Format::D32Float,
                            format::Swizzle::NO,
                            image::SubresourceRange {
                                aspects: format::Aspects::DEPTH,
                                levels: 0..1,
                                layers: 0..1,
                            },
                        )
                        .expect("Couldn't create the image view!");
                    depth_images.push(depth_image);
                    depth_image_views.push(image_view);
                    depth_image_requirements.push(requirements);
                    depth_image_memories.push(memory);
                }
            }
            let framebuffers: Vec<<back::Backend as Backend>::Framebuffer> = {
                image_views
                    .iter()
                    .enumerate()
                    .map(|(idx, image_view)| unsafe {
                        device
                            .create_framebuffer(
                                &render_pass,
                                vec![image_view, &depth_image_views[idx]],
                                image::Extent {
                                    width: dims.width as u32,
                                    height: dims.height as u32,
                                    depth: 1,
                                },
                            )
                            .map_err(|_| "Failed to create a framebuffer!")
                    })
                    .collect::<Result<Vec<_>, &str>>()
                    .unwrap()
            };
            (image_views, framebuffers)
        }
        #[cfg(not(feature = "gl"))]
        Backbuffer::Framebuffer(_) => unimplemented![],
        #[cfg(feature = "gl")]
        Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
    };

    {
        let image_views = format!["{:?}", image_views];
        debug![log, "vxdraw", "Created image views"; "image views" => image_views];
    }

    let framebuffers_string = format!["{:#?}", framebuffers];
    debug![log, "vxdraw", "Framebuffer information"; "framebuffers" => framebuffers_string];

    let max_frames_in_flight = 3;
    assert![max_frames_in_flight > 0];

    let mut frames_in_flight_fences = vec![];
    let mut present_wait_semaphores = vec![];
    for _ in 0..max_frames_in_flight {
        frames_in_flight_fences.push(device.create_fence(true).expect("Can't create fence"));
        present_wait_semaphores.push(device.create_semaphore().expect("Can't create semaphore"));
    }

    let acquire_image_semaphores = (0..image_count)
        .map(|_| device.create_semaphore().expect("Can't create semaphore"))
        .collect::<Vec<_>>();

    {
        let count = frames_in_flight_fences.len();
        debug![log, "vxdraw", "Allocated fences and semaphores"; "count" => count];
    }

    let mut command_pool = unsafe {
        device
            .create_command_pool_typed(&queue_group, pool::CommandPoolCreateFlags::RESET_INDIVIDUAL)
            .unwrap()
    };

    let command_buffers: Vec<_> = framebuffers
        .iter()
        .map(|_| command_pool.acquire_command_buffer::<command::MultiShot>())
        .collect();

    let mut windowing = Windowing {
        acquire_image_semaphores,
        acquire_image_semaphore_free: ManuallyDrop::new(
            device
                .create_semaphore()
                .expect("Unable to create semaphore"),
        ),
        adapter,
        backbuffer,
        command_buffers,
        command_pool: ManuallyDrop::new(command_pool),
        current_frame: 0,
        draw_order: vec![],
        max_frames_in_flight,
        debtris: None,
        device: ManuallyDrop::new(device),
        device_limits: phys_dev_limits,
        events_loop,
        frames_in_flight_fences,
        framebuffers,
        format,
        image_count: image_count as usize,
        image_views,
        present_wait_semaphores,
        queue_group: ManuallyDrop::new(queue_group),
        render_area: Rect {
            x: 0,
            y: 0,
            w: dims.width as i16,
            h: dims.height as i16,
        },
        render_pass: ManuallyDrop::new(render_pass),
        surf,
        swapchain: ManuallyDrop::new(swapchain),
        swapconfig: swap_config,
        strtexs: vec![],
        dyntexs: vec![],
        quads: None,
        depth_images,
        depth_image_views,
        depth_image_requirements,
        depth_image_memories,
        #[cfg(not(feature = "gl"))]
        vk_inst: ManuallyDrop::new(vk_inst),
        #[cfg(not(feature = "gl"))]
        window,
    };
    debtri::create_debug_triangle(&mut windowing);
    quads::create_quad(&mut windowing);
    windowing
}

pub fn collect_input(windowing: &mut Windowing) -> Vec<Event> {
    let mut inputs = vec![];
    windowing.events_loop.poll_events(|evt| {
        inputs.push(evt);
    });
    inputs
}

pub fn draw_frame_copy_framebuffer(
    s: &mut Windowing,
    log: &mut Logger<Generic>,
    view: &Matrix4<f32>,
) -> Vec<u8> {
    draw_frame_internal(s, log, view, copy_image_to_rgb)
}

pub fn draw_frame(s: &mut Windowing, log: &mut Logger<Generic>, view: &Matrix4<f32>) {
    draw_frame_internal(s, log, view, |_, _| {});
}

fn draw_frame_internal<T>(
    s: &mut Windowing,
    log: &mut Logger<Generic>,
    view: &Matrix4<f32>,
    postproc: fn(&mut Windowing, gfx_hal::window::SwapImageIndex) -> T,
) -> T {
    let postproc_res = unsafe {
        let swap_image = s
            .swapchain
            .acquire_image(
                u64::max_value(),
                FrameSync::Semaphore(&*s.acquire_image_semaphore_free),
            )
            .unwrap();

        core::mem::swap(
            &mut *s.acquire_image_semaphore_free,
            &mut s.acquire_image_semaphores[swap_image as usize],
        );

        s.device
            .wait_for_fence(
                &s.frames_in_flight_fences[s.current_frame],
                u64::max_value(),
            )
            .unwrap();

        s.device
            .reset_fence(&s.frames_in_flight_fences[s.current_frame])
            .unwrap();

        {
            let current_frame = s.current_frame;
            let texture_count = s.dyntexs.len();
            let debugtris_cnt = s.debtris.as_ref().map_or(0, |x| x.triangles_count);
            trace![log, "vxdraw", "Drawing frame"; "swapchain image" => swap_image, "flight" => current_frame, "textures" => texture_count, "debug triangles" => debugtris_cnt];
        }

        {
            let buffer = &mut s.command_buffers[s.current_frame];
            let clear_values = [
                ClearValue::Color(ClearColor::Float([1.0f32, 0.25, 0.5, 0.75])),
                ClearValue::DepthStencil(gfx_hal::command::ClearDepthStencil(1.0, 0)),
            ];
            buffer.begin(false);
            for strtex in s.strtexs.iter() {
                let image_barrier = memory::Barrier::Image {
                    states: (image::Access::empty(), image::Layout::General)
                        ..(
                            image::Access::SHADER_READ,
                            image::Layout::ShaderReadOnlyOptimal,
                        ),
                    target: &*strtex.image_buffer,
                    families: None,
                    range: image::SubresourceRange {
                        aspects: format::Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                };
                buffer.pipeline_barrier(
                    PipelineStage::TOP_OF_PIPE..PipelineStage::FRAGMENT_SHADER,
                    memory::Dependencies::empty(),
                    &[image_barrier],
                );
                // Submit automatically makes host writes available for the device
                let image_barrier = memory::Barrier::Image {
                    states: (image::Access::empty(), image::Layout::ShaderReadOnlyOptimal)
                        ..(image::Access::empty(), image::Layout::General),
                    target: &*strtex.image_buffer,
                    families: None,
                    range: image::SubresourceRange {
                        aspects: format::Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                };
                buffer.pipeline_barrier(
                    PipelineStage::FRAGMENT_SHADER..PipelineStage::HOST,
                    memory::Dependencies::empty(),
                    &[image_barrier],
                );
            }
            {
                let mut enc = buffer.begin_render_pass_inline(
                    &s.render_pass,
                    &s.framebuffers[swap_image as usize],
                    s.render_area,
                    clear_values.iter(),
                );
                for draw_cmd in s.draw_order.iter() {
                    match draw_cmd {
                        DrawType::StreamingTexture { id } => {
                            let strtex = &s.strtexs[*id];
                            enc.bind_graphics_pipeline(&strtex.pipeline);
                            enc.push_graphics_constants(
                                &strtex.pipeline_layout,
                                ShaderStageFlags::VERTEX,
                                0,
                                &*(view.as_ptr() as *const [u32; 16]),
                            );
                            enc.bind_graphics_descriptor_sets(
                                &strtex.pipeline_layout,
                                0,
                                Some(&*strtex.descriptor_set),
                                &[],
                            );
                            let buffers: ArrayVec<[_; 1]> = [(&*strtex.vertex_buffer, 0)].into();
                            enc.bind_vertex_buffers(0, buffers);
                            enc.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
                                buffer: &strtex.vertex_buffer_indices,
                                offset: 0,
                                index_type: gfx_hal::IndexType::U16,
                            });
                            enc.draw_indexed(0..strtex.count * 6, 0, 0..1);
                        }
                        DrawType::DynamicTexture { id } => {
                            let dyntex = &s.dyntexs[*id];
                            enc.bind_graphics_pipeline(&dyntex.pipeline);
                            if let Some(persp) = dyntex.fixed_perspective {
                                enc.push_graphics_constants(
                                    &dyntex.pipeline_layout,
                                    ShaderStageFlags::VERTEX,
                                    0,
                                    &*(persp.as_ptr() as *const [u32; 16]),
                                );
                            } else {
                                enc.push_graphics_constants(
                                    &dyntex.pipeline_layout,
                                    ShaderStageFlags::VERTEX,
                                    0,
                                    &*(view.as_ptr() as *const [u32; 16]),
                                );
                            }
                            enc.bind_graphics_descriptor_sets(
                                &dyntex.pipeline_layout,
                                0,
                                Some(&*dyntex.descriptor_set),
                                &[],
                            );
                            let mut data_target = s
                                .device
                                .acquire_mapping_writer::<u8>(
                                    &dyntex.texture_vertex_memory,
                                    0..dyntex.texture_vertex_requirements.size,
                                )
                                .expect("Unable to get mapping writer");
                            data_target[..dyntex.mockbuffer.len()]
                                .copy_from_slice(&dyntex.mockbuffer);
                            s.device
                                .release_mapping_writer(data_target)
                                .expect("Unable to release mapping writer");
                            let buffers: ArrayVec<[_; 1]> =
                                [(&*dyntex.texture_vertex_buffer, 0)].into();
                            enc.bind_vertex_buffers(0, buffers);
                            enc.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
                                buffer: &dyntex.texture_vertex_buffer_indices,
                                offset: 0,
                                index_type: gfx_hal::IndexType::U16,
                            });
                            enc.draw_indexed(0..dyntex.count * 6, 0, 0..1);
                        }
                    }
                }
                if let Some(ref quads) = s.quads {
                    enc.bind_graphics_pipeline(&quads.pipeline);
                    enc.push_graphics_constants(
                        &quads.pipeline_layout,
                        ShaderStageFlags::VERTEX,
                        0,
                        &*(view.as_ptr() as *const [u32; 16]),
                    );
                    let buffers: ArrayVec<[_; 1]> = [(&quads.quads_buffer, 0)].into();
                    enc.bind_vertex_buffers(0, buffers);
                    enc.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
                        buffer: &quads.quads_buffer_indices,
                        offset: 0,
                        index_type: gfx_hal::IndexType::U16,
                    });
                    enc.draw_indexed(0..quads.count as u32 * 6, 0, 0..1);
                }

                if let Some(ref debtris) = s.debtris {
                    enc.bind_graphics_pipeline(&debtris.pipeline);
                    let ratio =
                        s.swapconfig.extent.width as f32 / s.swapconfig.extent.height as f32;
                    enc.push_graphics_constants(
                        &debtris.pipeline_layout,
                        ShaderStageFlags::VERTEX,
                        0,
                        &(std::mem::transmute::<f32, [u32; 1]>(ratio)),
                    );
                    let count = debtris.triangles_count;
                    let buffers: ArrayVec<[_; 1]> = [(&debtris.triangles_buffer, 0)].into();
                    enc.bind_vertex_buffers(0, buffers);
                    enc.draw(0..(count * 3) as u32, 0..1);
                }
            }
            buffer.finish();
        }

        let command_buffers = &s.command_buffers[s.current_frame];
        let wait_semaphores: ArrayVec<[_; 1]> = [(
            &s.acquire_image_semaphores[swap_image as usize],
            PipelineStage::COLOR_ATTACHMENT_OUTPUT,
        )]
        .into();
        {
            let present_wait_semaphore = &s.present_wait_semaphores[s.current_frame];
            let signal_semaphores: ArrayVec<[_; 1]> = [present_wait_semaphore].into();
            let submission = Submission {
                command_buffers: once(command_buffers),
                wait_semaphores,
                signal_semaphores,
            };
            s.queue_group.queues[0].submit(
                submission,
                Some(&s.frames_in_flight_fences[s.current_frame]),
            );
        }
        let postproc_res = postproc(s, swap_image);
        let present_wait_semaphore = &s.present_wait_semaphores[s.current_frame];
        let present_wait_semaphores: ArrayVec<[_; 1]> = [present_wait_semaphore].into();
        s.swapchain
            .present(
                &mut s.queue_group.queues[0],
                swap_image,
                present_wait_semaphores,
            )
            .unwrap();
        postproc_res
    };
    s.current_frame = (s.current_frame + 1) % s.max_frames_in_flight;
    postproc_res
}

pub fn generate_map(s: &mut Windowing, w: u32, h: u32) -> Vec<u8> {
    static VERTEX_SOURCE: &str = include_str!("../shaders/proc1.vert");
    static FRAGMENT_SOURCE: &str = include_str!("../shaders/proc1.frag");
    let vs_module = {
        let glsl = VERTEX_SOURCE;
        let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex)
            .unwrap()
            .bytes()
            .map(Result::unwrap)
            .collect();
        unsafe { s.device.create_shader_module(&spirv) }.unwrap()
    };
    let fs_module = {
        let glsl = FRAGMENT_SOURCE;
        let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Fragment)
            .unwrap()
            .bytes()
            .map(Result::unwrap)
            .collect();
        unsafe { s.device.create_shader_module(&spirv) }.unwrap()
    };
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

    let input_assembler = InputAssemblerDesc::new(Primitive::TriangleList);

    let vertex_buffers: Vec<VertexBufferDesc> = vec![VertexBufferDesc {
        binding: 0,
        stride: 8u32,
        rate: 0,
    }];

    let attributes: Vec<AttributeDesc> = vec![AttributeDesc {
        location: 0,
        binding: 0,
        element: Element {
            format: format::Format::Rg32Float,
            offset: 0,
        },
    }];

    let rasterizer = Rasterizer {
        depth_clamping: false,
        polygon_mode: PolygonMode::Fill,
        cull_face: Face::NONE,
        front_face: FrontFace::Clockwise,
        depth_bias: None,
        conservative: false,
    };

    let depth_stencil = DepthStencilDesc {
        depth: DepthTest::Off,
        depth_bounds: false,
        stencil: StencilTest::Off,
    };

    let blender = {
        let blend_state = BlendState::On {
            color: BlendOp::Add {
                src: Factor::One,
                dst: Factor::Zero,
            },
            alpha: BlendOp::Add {
                src: Factor::One,
                dst: Factor::Zero,
            },
        };
        BlendDesc {
            logic_op: Some(LogicOp::Copy),
            targets: vec![ColorBlendDesc(ColorMask::ALL, blend_state)],
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

    let baked_states = BakedStates {
        viewport: Some(Viewport {
            rect: extent,
            depth: (0.0..1.0),
        }),
        scissor: Some(extent),
        blend_color: None,
        depth_bounds: None,
    };
    let bindings = Vec::<DescriptorSetLayoutBinding>::new();
    let immutable_samplers = Vec::<<back::Backend as Backend>::Sampler>::new();
    let mut mapgen_descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> =
        vec![unsafe {
            s.device
                .create_descriptor_set_layout(bindings, immutable_samplers)
                .expect("Couldn't make a DescriptorSetLayout")
        }];
    let mut push_constants = Vec::<(ShaderStageFlags, core::ops::Range<u32>)>::new();
    push_constants.push((ShaderStageFlags::FRAGMENT, 0..4));

    let mapgen_pipeline_layout = unsafe {
        s.device
            .create_pipeline_layout(&mapgen_descriptor_set_layouts, push_constants)
            .expect("Couldn't create a pipeline layout")
    };

    // Describe the pipeline (rasterization, mapgen interpretation)
    let pipeline_desc = GraphicsPipelineDesc {
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
        flags: PipelineCreationFlags::empty(),
        parent: BasePipeline::None,
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
                image::Usage::COLOR_ATTACHMENT | image::Usage::TRANSFER_DST | image::Usage::SAMPLED,
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
        let clear_values = [ClearValue::Color(ClearColor::Float([
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
                PipelineStage::TOP_OF_PIPE..PipelineStage::FRAGMENT_SHADER,
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
                ShaderStageFlags::FRAGMENT,
                0,
                &(std::mem::transmute::<[f32; 4], [u32; 4]>([w as f32, 0.3, 93.0, 3.0])),
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
        s.device.destroy_fence(upload_fence);
        s.command_pool.free(once(cmd_buffer));

        let footprint = s.device.get_image_subresource_footprint(
            &image,
            image::Subresource {
                aspects: format::Aspects::COLOR,
                level: 0,
                layer: 0,
            },
        );

        let map = s
            .device
            .acquire_mapping_reader(&memory, footprint.slice)
            .expect("Mapped memory");

        let pixel_size = size_of::<load_image::Rgba<u8>>() as u32;
        let row_size = pixel_size * w;

        let mut result: Vec<u8> = Vec::new();
        for y in 0..h as usize {
            let dest_base = y * footprint.row_pitch as usize;
            result.extend(map[dest_base..dest_base + row_size as usize].iter());
        }
        s.device.release_mapping_reader(map);

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
        result
    }
}

// ---

#[cfg(feature = "gfx_tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Deg, Vector3};
    use test::Bencher;

    // ---

    static TESTURE: &[u8] = include_bytes!["../images/testure.png"];

    // ---

    #[test]
    fn setup_and_teardown() {
        let mut logger = Logger::spawn_void();
        let _ = init_window_with_vulkan(&mut logger, ShowWindow::Headless1k);
    }

    #[test]
    fn setup_and_teardown_draw_clear() {
        let mut logger = Logger::spawn_void();
        logger.set_colorize(true);
        logger.set_log_level(64);

        let mut windowing = init_window_with_vulkan(&mut logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let img = draw_frame_copy_framebuffer(&mut windowing, &mut logger, &prspect);

        assert_swapchain_eq(&mut windowing, "setup_and_teardown_draw_with_test", img);
    }

    #[test]
    fn setup_and_teardown_with_gpu_upload() {
        let mut logger = Logger::spawn_void();
        let mut windowing = init_window_with_vulkan(&mut logger, ShowWindow::Headless1k);

        let (buffer, memory, _) =
            make_vertex_buffer_with_data_on_gpu(&mut windowing, &vec![1.0f32; 10_000]);

        unsafe {
            windowing.device.destroy_buffer(buffer);
            windowing.device.free_memory(memory);
        }
    }

    #[test]
    fn init_window_and_get_input() {
        let mut logger = Logger::spawn_void();
        let mut windowing = init_window_with_vulkan(&mut logger, ShowWindow::Headless1k);
        collect_input(&mut windowing);
    }

    #[test]
    fn generate_map() {
        let mut logger = Logger::spawn_void();
        let mut windowing = init_window_with_vulkan(&mut logger, ShowWindow::Headless1k);
        let mut img = super::generate_map(&mut windowing, 1000, 1000);
        let img = img
            .drain(..)
            .enumerate()
            .filter(|(idx, _)| idx % 4 != 0)
            .map(|(_, v)| v)
            .collect::<Vec<u8>>();
        assert_swapchain_eq(&mut windowing, "genmap", img);
    }

    #[test]
    fn tearing_test() {
        let mut logger = Logger::spawn_void();
        let mut windowing = init_window_with_vulkan(&mut logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let _tri = make_centered_equilateral_triangle();
        debtri::push(&mut windowing, debtri::DebugTriangle::default());
        for i in 0..=360 {
            if i % 2 == 0 {
                add_4_screencorners(&mut windowing);
            } else {
                debtri::pop_many(&mut windowing, 4);
            }
            let rot =
                prspect * Matrix4::from_axis_angle(Vector3::new(0.0f32, 0.0, 1.0), Deg(i as f32));
            draw_frame(&mut windowing, &mut logger, &rot);
            // std::thread::sleep(std::time::Duration::new(0, 80_000_000));
        }
    }

    #[test]
    fn correct_perspective() {
        let mut logger = Logger::spawn_void();
        {
            let windowing = init_window_with_vulkan(&mut logger, ShowWindow::Headless1k);
            assert_eq![Matrix4::identity(), gen_perspective(&windowing)];
        }
        {
            let windowing = init_window_with_vulkan(&mut logger, ShowWindow::Headless1x2k);
            assert_eq![
                Matrix4::from_nonuniform_scale(1.0, 0.5, 1.0),
                gen_perspective(&windowing)
            ];
        }
        {
            let windowing = init_window_with_vulkan(&mut logger, ShowWindow::Headless2x1k);
            assert_eq![
                Matrix4::from_nonuniform_scale(0.5, 1.0, 1.0),
                gen_perspective(&windowing)
            ];
        }
    }

    #[test]
    fn strtex_and_dyntex_respect_draw_order() {
        let mut logger = Logger::spawn_void();
        let mut windowing = init_window_with_vulkan(&mut logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let options = dyntex::TextureOptions {
            depth_test: false,
            ..dyntex::TextureOptions::default()
        };
        let tex1 = dyntex::push_texture(&mut windowing, TESTURE, options);
        let tex2 = strtex::push_texture(
            &mut windowing,
            strtex::TextureOptions {
                depth_test: false,
                width: 1,
                height: 1,
                ..strtex::TextureOptions::default()
            },
            &mut logger,
        );
        let tex3 = dyntex::push_texture(&mut windowing, TESTURE, options);
        let tex4 = strtex::push_texture(
            &mut windowing,
            strtex::TextureOptions {
                depth_test: false,
                width: 1,
                height: 1,
                ..strtex::TextureOptions::default()
            },
            &mut logger,
        );

        strtex::streaming_texture_set_pixel(&mut windowing, &tex2, 0, 0, (255, 0, 255, 255));
        strtex::streaming_texture_set_pixel(&mut windowing, &tex4, 0, 0, (255, 255, 255, 255));

        dyntex::push_sprite(
            &mut windowing,
            &tex1,
            dyntex::Sprite {
                rotation: 0.0,
                ..dyntex::Sprite::default()
            },
        );
        strtex::push_sprite(
            &mut windowing,
            &tex2,
            strtex::Sprite {
                rotation: 0.5,
                ..strtex::Sprite::default()
            },
        );
        dyntex::push_sprite(
            &mut windowing,
            &tex3,
            dyntex::Sprite {
                rotation: 1.0,
                ..dyntex::Sprite::default()
            },
        );
        strtex::push_sprite(
            &mut windowing,
            &tex4,
            strtex::Sprite {
                scale: 0.5,
                rotation: 0.0,
                ..strtex::Sprite::default()
            },
        );

        let img = draw_frame_copy_framebuffer(&mut windowing, &mut logger, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "strtex_and_dyntex_respect_draw_order", img);
    }

    // ---

    #[bench]
    fn clears_per_second(b: &mut Bencher) {
        let mut logger = Logger::spawn_void();
        let mut windowing = init_window_with_vulkan(&mut logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        b.iter(|| {
            draw_frame(&mut windowing, &mut logger, &prspect);
        });
    }

    #[bench]
    fn bench_generate_map(b: &mut Bencher) {
        let mut logger = Logger::spawn_void();
        let mut windowing = init_window_with_vulkan(&mut logger, ShowWindow::Headless1k);
        b.iter(|| {
            super::generate_map(&mut windowing, 1000, 1000);
        });
    }
}
