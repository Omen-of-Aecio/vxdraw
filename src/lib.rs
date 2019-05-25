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
use arrayvec::ArrayVec;
use cgmath::prelude::*;
use cgmath::Matrix4;
use gfx_hal::{
    adapter::PhysicalDevice,
    command::{self, ClearColor, ClearValue},
    device::Device,
    format::{self, ChannelType, Swizzle},
    image, memory, pass, pool,
    pso,
    queue::Submission,
    window::{Extent2D, PresentMode::*, Surface, Swapchain},
    Backbuffer, Backend, FrameSync, Instance, SwapchainConfig,
};
use logger::{debug, info, trace, warn, InDebug, InDebugPretty, Logpass};
use std::iter::once;
use std::mem::{ManuallyDrop};
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

pub fn init_window_with_vulkan(mut log: Logpass, show: ShowWindow) -> Windowing {
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
        render_area: pso::Rect {
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
        log,
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
    view: &Matrix4<f32>,
) -> Vec<u8> {
    draw_frame_internal(s, view, copy_image_to_rgb)
}

pub fn draw_frame(s: &mut Windowing, view: &Matrix4<f32>) {
    draw_frame_internal(s, view, |_, _| {});
}

fn draw_frame_internal<T>(
    s: &mut Windowing,
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
            trace![s.log, "vxdraw", "Drawing frame"; "swapchain image" => swap_image, "flight" => current_frame, "textures" => texture_count, "debug triangles" => debugtris_cnt];
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
                    pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::FRAGMENT_SHADER,
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
                    pso::PipelineStage::FRAGMENT_SHADER..pso::PipelineStage::HOST,
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
                                pso::ShaderStageFlags::VERTEX,
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
                                    pso::ShaderStageFlags::VERTEX,
                                    0,
                                    &*(persp.as_ptr() as *const [u32; 16]),
                                );
                            } else {
                                enc.push_graphics_constants(
                                    &dyntex.pipeline_layout,
                                    pso::ShaderStageFlags::VERTEX,
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
                        pso::ShaderStageFlags::VERTEX,
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
                        pso::ShaderStageFlags::VERTEX,
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
            pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
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

// ---

#[cfg(feature = "gfx_tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Deg, Vector3};
    use test::Bencher;
    use logger::{Logger, Generic, GenericLogger};

    // ---

    static TESTURE: &[u8] = include_bytes!["../images/testure.png"];

    // ---

    #[test]
    fn setup_and_teardown() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let _ = init_window_with_vulkan(logger, ShowWindow::Headless1k);
    }

    #[test]
    fn setup_and_teardown_draw_clear() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();

        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);

        assert_swapchain_eq(&mut windowing, "setup_and_teardown_draw_with_test", img);
    }

    #[test]
    fn setup_and_teardown_with_gpu_upload() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);

        let (buffer, memory, _) =
            make_vertex_buffer_with_data_on_gpu(&mut windowing, &vec![1.0f32; 10_000]);

        unsafe {
            windowing.device.destroy_buffer(buffer);
            windowing.device.free_memory(memory);
        }
    }

    #[test]
    fn init_window_and_get_input() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        collect_input(&mut windowing);
    }

    #[test]
    fn tearing_test() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
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
            draw_frame(&mut windowing, &rot);
            // std::thread::sleep(std::time::Duration::new(0, 80_000_000));
        }
    }

    #[test]
    fn correct_perspective() {
        {
            let logger = Logger::<Generic>::spawn_void().to_logpass();
            let windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
            assert_eq![Matrix4::identity(), gen_perspective(&windowing)];
        }
        {
            let logger = Logger::<Generic>::spawn_void().to_logpass();
            let windowing = init_window_with_vulkan(logger, ShowWindow::Headless1x2k);
            assert_eq![
                Matrix4::from_nonuniform_scale(1.0, 0.5, 1.0),
                gen_perspective(&windowing)
            ];
        }
        {
            let logger = Logger::<Generic>::spawn_void().to_logpass();
            let windowing = init_window_with_vulkan(logger, ShowWindow::Headless2x1k);
            assert_eq![
                Matrix4::from_nonuniform_scale(0.5, 1.0, 1.0),
                gen_perspective(&windowing)
            ];
        }
    }

    #[test]
    fn strtex_and_dyntex_respect_draw_order() {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
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

        let img = draw_frame_copy_framebuffer(&mut windowing, &prspect);
        utils::assert_swapchain_eq(&mut windowing, "strtex_and_dyntex_respect_draw_order", img);
    }

    // ---

    #[bench]
    fn clears_per_second(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_logpass();
        let mut windowing = init_window_with_vulkan(logger, ShowWindow::Headless1k);
        let prspect = gen_perspective(&windowing);

        b.iter(|| {
            draw_frame(&mut windowing, &prspect);
        });
    }
}
