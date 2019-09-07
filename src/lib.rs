//! VxDraw: Simple vulkan renderer
//!
//! # Example - Hello Triangle #
//! To get started, spawn a window and draw a debug triangle!
//! ```
//! use cgmath::{prelude::*, Matrix4};
//! use vxdraw::{debtri::DebugTriangle, void_logger, ShowWindow, VxDraw};
//! fn main() {
//!     #[cfg(not(test))]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Enable);
//!     #[cfg(test)]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Headless1k);
//!
//!     vx.debtri().add(DebugTriangle::default());
//!     vx.draw_frame();
//!
//!     // Sleep here so the window does not instantly disappear
//!     #[cfg(not(test))]
//!     std::thread::sleep(std::time::Duration::new(3, 0));
//! }
//! ```
//! ## Animation: Rotating triangle ##
//! Here's a more interesting example:
//! ```
//! use cgmath::{prelude::*, Deg, Matrix4};
//! use vxdraw::{debtri::DebugTriangle, void_logger, ShowWindow, VxDraw};
//! fn main() {
//!     #[cfg(not(test))]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Enable);
//!     #[cfg(test)]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Headless1k);
//!
//!     // Spawn a debug triangle, the handle is used to refer to it later
//!     let handle = vx.debtri().add(DebugTriangle::default());
//!
//!     for _ in 0..360 {
//!         // Rotate the triangle by 1 degree
//!         vx.debtri().rotate(&handle, Deg(1.0));
//!
//!         // Draw the scene
//!         vx.draw_frame();
//!
//!         // Wait 10 milliseconds
//!         #[cfg(not(test))]
//!         std::thread::sleep(std::time::Duration::new(0, 10_000_000));
//!     }
//! }
//! ```
//!
//! # Logging #
//! [VxDraw] is quite a lot of machinery, so it is useful for [VxDraw] to be able to log. To avoid
//! hardcoding a logging dependency in the interface the library provides [Logger] which is a type
//! that only depends on the standard library. You can build your own "logger bridge" using this.
//!
//! ```
//! use cgmath::{prelude::*, Matrix4};
//! use vxdraw::*;
//!
//! fn main() {
//!     let log = Box::new(|lvl: u8, msg| {
//!         struct Adapter {
//!             pub msg: Box<dyn Fn(&mut std::fmt::Formatter) -> std::fmt::Result + Send + Sync>
//!         }
//!         impl std::fmt::Display for Adapter {
//!             fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//!                 (self.msg)(f)
//!             }
//!         }
//!         println!["{} @ {}", lvl, Adapter { msg }];
//!     });
//!     #[cfg(not(test))]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Enable);
//!     #[cfg(test)]
//!     let mut vx = VxDraw::new(void_logger(), ShowWindow::Headless1k);
//!
//!     vx.debtri().add(debtri::DebugTriangle::default());
//!
//!     vx.draw_frame();
//!
//!     // Sleep here so the window does not instantly disappear
//!     #[cfg(not(test))]
//!     std::thread::sleep(std::time::Duration::new(3, 0));
//! }
//! ```
#![feature(test)]
#![deny(missing_docs)]
extern crate test;

pub use crate::data::VxDraw;
use crate::data::{DrawType, LayerHoles, StreamingTextureWrite};
use arrayvec::ArrayVec;
use cgmath::prelude::*;
use cgmath::Matrix4;
use fast_logger::{debug, error, info, trace, warn, InDebug, InDebugPretty, Logpass};
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
    command::{self, ClearColor, ClearValue},
    device::Device,
    format::{self, ChannelType, Swizzle},
    image, memory, pass, pool, pso,
    queue::Submission,
    window::{Extent2D, PresentMode::*, Surface, Swapchain},
    Backend, Instance, SwapchainConfig,
};
use std::fmt;
use std::iter::once;
use std::mem::ManuallyDrop;
use winit::{dpi::LogicalSize, Event, EventsLoop, WindowBuilder};

pub mod blender;
mod data;
pub mod debtri;
pub mod dyntex;
pub mod quads;
pub mod strtex;
pub mod text;
pub mod utils;

use utils::*;

// ---

/// A description of a color
pub enum Color {
    /// Red green blue alpha color, represented by u8
    Rgba(u8, u8, u8, u8),
}

impl From<Color> for (u8, u8, u8, u8) {
    fn from(color: Color) -> Self {
        let Color::Rgba(r, g, b, a) = color;
        (r, g, b, a)
    }
}

impl From<&Color> for (u8, u8, u8, u8) {
    fn from(color: &Color) -> Self {
        let Color::Rgba(r, g, b, a) = color;
        (*r, *g, *b, *a)
    }
}

// ---

/// Logger bridge type used when initializing [VxDraw]
///
/// The first argument is the log level, with 0 being severe and 255 being trace.
/// The second argument is the context label.
/// The final argument is an arbitrary formatter that outputs text, this text may be on multiple
/// lines.
///
/// In most tests we use the logger from another crate because it has `spawn_test`, which is useful
/// for quickly debugging a failing test.
pub type Logger = Box<dyn FnMut(u8, Box<dyn Fn(&mut fmt::Formatter) -> fmt::Result + Send + Sync>)>;

/// Create an empty logger bridge
pub fn void_logger() -> Logger {
    Box::new(|_, _| {})
}

/// Information regarding window visibility
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShowWindow {
    /// Runs vulkan in headless mode (hidden window) with a swapchain of 1000x1000
    Headless1k,
    /// Runs vulkan in headless mode (hidden window) with a swapchain of 2000x1000
    Headless2x1k,
    /// Runs vulkan in headless mode (hidden window) with a swapchain of 1000x2000
    Headless1x2k,
    /// Runs vulkan with a visible window
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
fn set_window_size(window: &glutin::Window, show: ShowWindow) -> Extent2D {
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

impl VxDraw {
    /// Spawn a new VxDraw context with a window
    ///
    /// This method sets up all that is necessary for drawing.
    pub fn new(log: Logger, show: ShowWindow) -> VxDraw {
        let mut log = Logpass::from_compatibility(log);

        #[cfg(feature = "gl")]
        static BACKEND: &str = "OpenGL";
        #[cfg(feature = "vulkan")]
        static BACKEND: &str = "Vulkan";
        #[cfg(feature = "metal")]
        static BACKEND: &str = "Metal";
        #[cfg(feature = "dx12")]
        static BACKEND: &str = "Dx12";

        info![log, "Initializing rendering"; "show" => InDebug(&show), "backend" => BACKEND];

        let events_loop = EventsLoop::new();
        let window_builder = WindowBuilder::new().with_visibility(show == ShowWindow::Enable);

        #[cfg(feature = "gl")]
        let (mut adapters, mut surf, dims) = {
            let window = {
                let builder = back::config_context(
                    back::glutin::ContextBuilder::new(),
                    format::Format::Rgba8Srgb,
                    None,
                )
                .with_vsync(true);
                back::glutin::WindowedContext::new_windowed(window_builder, builder, &events_loop)
                    .unwrap()
            };

            set_window_size(window.window(), show);
            let dims = {
                let dpi_factor = window.get_hidpi_factor();
                debug![log, "Window DPI factor"; "factor" => dpi_factor];
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
            debug![log, "Window DPI factor"; "factor" => dpi_factor];
            (window, vk_inst, adapters, surf, dims)
        };

        // ---

        {
            let len = adapters.len();
            debug![log, "Adapters found"; "count" => len];
        }

        for (idx, adap) in adapters.iter().enumerate() {
            let info = adap.info.clone();
            let limits = adap.physical_device.limits();
            debug![log, "Adapter found"; "idx" => idx, "info" => InDebugPretty(&info), "device limits" => InDebugPretty(&limits)];
        }

        // TODO Find appropriate adapter, I've never seen a case where we have 2+ adapters, that time
        // will come one day
        let adapter = adapters.remove(0);
        let (device, queue_group) = adapter
            .open_with::<_, gfx_hal::Graphics>(1, |family| surf.supports_queue_family(family))
            .expect("Unable to find device supporting graphics");

        let _phys_dev_limits = adapter.physical_device.limits();

        let (caps, formats, present_modes) = surf.compatibility(&adapter.physical_device);

        debug![log, "Surface capabilities"; "capabilities" => InDebugPretty(&caps); clone caps];
        debug![log, "Formats available"; "formats" => InDebugPretty(&formats); clone formats];
        let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .cloned()
                .unwrap_or(formats[0])
        });

        assert![adapter
            .physical_device
            .format_properties(Some(format))
            .optimal_tiling
            .contains(format::ImageFeature::BLIT_SRC)];

        debug![log, "Format chosen"; "format" => InDebugPretty(&format); clone format];
        debug![log, "Available present modes"; "modes" => InDebugPretty(&present_modes); clone present_modes];

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
        debug![log, "Using best possible present mode"; "mode" => InDebug(&present_mode)];

        let image_count = if present_mode == Mailbox {
            (caps.image_count.end() - 1)
                .min(3)
                .max(*caps.image_count.start())
        } else {
            (caps.image_count.end() - 1)
                .min(2)
                .max(*caps.image_count.start())
        };
        debug![log, "Using swapchain images"; "count" => image_count];

        debug![log, "Swapchain size"; "extent" => InDebug(&dims)];

        let mut swap_config = SwapchainConfig::from_caps(&caps, format, dims);
        swap_config.present_mode = present_mode;
        swap_config.image_count = image_count;
        swap_config.extent = dims;
        if caps.usage.contains(image::Usage::TRANSFER_SRC) {
            swap_config.image_usage |= gfx_hal::image::Usage::TRANSFER_SRC;
        } else {
            warn![
                log,
                "Surface does not support TRANSFER_SRC, may fail during testing"
            ];
        }

        debug![log, "Swapchain final configuration"; "swapchain" => InDebugPretty(&swap_config); clone swap_config];

        let (swapchain, images) =
            unsafe { device.create_swapchain(&mut surf, swap_config.clone(), None) }
                .expect("Unable to create swapchain");

        let images_string = format!["{:#?}", images];
        debug![log, "Image information"; "images" => images_string];

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
            debug![log, "Render pass info"; "color attachment" => InDebugPretty(&color_attachment); clone color_attachment];
            unsafe {
                device
                    .create_render_pass(&[color_attachment, depth], &[subpass], &[])
                    .map_err(|_| "Couldn't create a render pass!")
                    .unwrap()
            }
        };

        {
            let rpfmt = format!["{:#?}", render_pass];
            debug![log, "Created render pass for framebuffers"; "renderpass" => rpfmt];
        }

        let mut depth_images: Vec<<back::Backend as Backend>::Image> = vec![];
        let mut depth_image_views: Vec<<back::Backend as Backend>::ImageView> = vec![];
        let mut depth_image_memories: Vec<<back::Backend as Backend>::Memory> = vec![];
        let mut depth_image_requirements: Vec<memory::Requirements> = vec![];

        let (image_views, framebuffers) = {
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
                            format::Format::D32Sfloat,
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
                            format::Format::D32Sfloat,
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
        };

        {
            let image_views = format!["{:?}", image_views];
            debug![log, "Created image views"; "image views" => image_views];
        }

        let framebuffers_string = format!["{:#?}", framebuffers];
        debug![log, "Framebuffer information"; "framebuffers" => framebuffers_string];

        let max_frames_in_flight = image_count as usize;
        assert![max_frames_in_flight > 0];

        let mut frames_in_flight_fences = vec![];
        let mut present_wait_semaphores = vec![];
        for _ in 0..max_frames_in_flight {
            frames_in_flight_fences.push(device.create_fence(true).expect("Can't create fence"));
            present_wait_semaphores
                .push(device.create_semaphore().expect("Can't create semaphore"));
        }

        let acquire_image_semaphores = (0..image_count)
            .map(|_| device.create_semaphore().expect("Can't create semaphore"))
            .collect::<Vec<_>>();

        {
            let count = frames_in_flight_fences.len();
            debug![log, "Allocated fences and semaphores"; "count" => count];
        }

        let mut command_pool = unsafe {
            device
                .create_command_pool_typed(
                    &queue_group,
                    pool::CommandPoolCreateFlags::RESET_INDIVIDUAL,
                )
                .unwrap()
        };

        let command_buffers: Vec<_> = framebuffers
            .iter()
            .map(|_| command_pool.acquire_command_buffer::<command::MultiShot>())
            .collect();

        let debtris = debtri::create_debug_triangle(&device, &adapter, format, images.len());

        VxDraw {
            acquire_image_semaphores,
            acquire_image_semaphore_free: ManuallyDrop::new(
                device
                    .create_semaphore()
                    .expect("Unable to create semaphore"),
            ),
            adapter,
            images,
            command_buffers,
            command_pool: ManuallyDrop::new(command_pool),
            current_frame: 0,
            draw_order: vec![],
            layer_holes: LayerHoles::new(image_count as usize),
            max_frames_in_flight,
            device,
            // device_limits: phys_dev_limits,
            texts: vec![],
            events_loop,
            frames_in_flight_fences,
            framebuffers,
            format,
            image_views,
            perspective: Matrix4::identity(),
            present_wait_semaphores,
            queue_group,
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
            quads: vec![],
            depth_images,
            depth_image_views,
            depth_image_memories,
            #[cfg(not(feature = "gl"))]
            vk_inst,
            #[cfg(not(feature = "gl"))]
            window,
            log,
            debtris,

            clear_color: ClearColor::Sfloat([1.0f32, 0.25, 0.5, 0.0]),
        }
    }

    /// Set the perspective to be used when drawing geometry
    pub fn set_perspective(&mut self, perspective: Matrix4<f32>) {
        self.perspective = perspective;
    }

    /// Translate a pixel to the world coordinates according to the current perspective
    ///
    /// To set the current perspective see [VxDraw::set_perspective].
    pub fn to_world_coords(&self, screen_coord: (f32, f32)) -> (f32, f32) {
        if let Some(inverse) = self.perspective.invert() {
            let size = self.get_window_size_in_pixels_float();
            let pos = cgmath::vec4(
                (screen_coord.0 / (size.0 / 2.0) - 1.0),
                (screen_coord.1 / (size.1 / 2.0) - 1.0),
                0.0,
                1.0,
            );
            let vec4 = inverse * pos;
            (vec4.x, vec4.y)
        } else {
            (0.0, 0.0)
        }
    }

    pub(crate) fn wait_for_fences(&self) {
        unsafe {
            self.device
                .wait_for_fences(
                    &self.frames_in_flight_fences[..],
                    gfx_hal::device::WaitFor::All,
                    u64::max_value(),
                )
                .expect("Unable to wait for frame fences");
        }
    }

    /// Set the clear color when clearing a frame
    pub fn set_clear_color(&mut self, color: Color) {
        let (r, g, b, a) = color.into();
        self.clear_color = ClearColor::Sfloat([
            f32::from(r) / 255.0,
            f32::from(g) / 255.0,
            f32::from(b) / 255.0,
            f32::from(a) / 255.0,
        ]);
    }

    /// Get how many swapchain images there exist
    ///
    /// 2 indicates double-buffering, 3 triple-buffering. May return 1 in the case of an OpenGL backend.
    pub fn buffer_count(&self) -> usize {
        self.swapconfig.image_count as usize
    }

    /// Swap two layer orders
    pub fn swap_layers(&mut self, layer1: &impl Layerable, layer2: &impl Layerable) {
        let idx1 = layer1.get_layer(self);
        let idx2 = layer2.get_layer(self);
        self.draw_order.swap(idx1, idx2);
    }

    /// Get the size of the display window in floats
    #[cfg(feature = "gl")]
    pub fn get_window_size_in_pixels(&self) -> (u32, u32) {
        let dpi_factor = self.surf.get_window().get_hidpi_factor();
        let (w, h): (u32, u32) = self.surf.get_window().get_inner_size().unwrap().into();
        (w, h)
    }

    /// Get the size of the display window in floats
    #[cfg(feature = "vulkan")]
    pub fn get_window_size_in_pixels(&self) -> (u32, u32) {
        let dpi_factor = self.window.get_hidpi_factor();
        let (w, h): (u32, u32) = self.window.get_inner_size().unwrap().into();
        (w, h)
    }

    /// Get the size of the display window in floats
    pub fn get_window_size_in_pixels_float(&self) -> (f32, f32) {
        let pixels = self.get_window_size_in_pixels();
        (pixels.0 as f32, pixels.1 as f32)
    }

    /// Set the size of the display window
    #[cfg(feature = "gl")]
    pub fn set_window_size(&mut self, size: (u32, u32)) {
        let dpi_factor = self.surf.get_window().get_hidpi_factor();
        self.surf.get_window().set_inner_size(LogicalSize {
            width: f64::from(size.0) / dpi_factor,
            height: f64::from(size.1) / dpi_factor,
        });
    }

    /// Set the size of the display window
    #[cfg(feature = "vulkan")]
    pub fn set_window_size(&mut self, size: (u32, u32)) {
        let dpi_factor = self.window.get_hidpi_factor();
        self.window.set_inner_size(LogicalSize {
            width: f64::from(size.0) / dpi_factor,
            height: f64::from(size.1) / dpi_factor,
        });
    }

    /// Get a handle to all debug triangles, allows editing, removal, or creation of debtris
    /// See [debtri::Debtri] for more details.
    pub fn debtri(&mut self) -> debtri::Debtri {
        debtri::Debtri::new(self)
    }

    /// Get a handle to all quads, allows editing, removal, or creation of new quads and
    /// layers. See [quads::Quads] for more details.
    pub fn quads(&mut self) -> quads::Quads {
        quads::Quads::new(self)
    }

    /// Get a handle to all dynamic textures, allows editing, removal, or creation of new dynamic
    /// textures. See [dyntex::Dyntex] for more details.
    pub fn dyntex(&mut self) -> dyntex::Dyntex {
        dyntex::Dyntex::new(self)
    }

    /// Get a handle to all streaming textures, allows editing, removal, or creation of new
    /// streaming textures. See [strtex::Strtex] for more details.
    pub fn strtex(&mut self) -> strtex::Strtex {
        strtex::Strtex::new(self)
    }

    /// Drawing text
    pub fn text(&mut self) -> text::Texts {
        text::Texts::new(self)
    }

    /// Collect all pending input events to this window
    pub fn collect_input(&mut self) -> Vec<Event> {
        let mut inputs = vec![];
        let log = &mut self.log;
        self.events_loop.poll_events(|evt| match evt {
            winit::Event::WindowEvent { ref event, .. } => match event {
                winit::WindowEvent::Resized(dims) => {
                    info![log, "Window resized event"; "dimensions" => InDebug(&dims); clone dims];
                }
                _ => inputs.push(evt),
            },
            _ => inputs.push(evt),
        });
        inputs
    }

    /// Draw a frame but also copy the resulting image out
    pub fn draw_frame_copy_framebuffer(&mut self) -> Vec<u8> {
        self.draw_frame_internal(copy_image_to_rgb)
    }

    /// Draw a single frame and present it to the screen
    ///
    /// The view matrix is used to translate all elements on the screen with the exception of debug
    /// triangles and layers that have their own view.
    pub fn draw_frame(&mut self) {
        self.draw_frame_internal(|_, _| {});
    }

    /// Recreate the swapchain, must be called after a window resize
    fn window_resized_recreate_swapchain(&mut self) {
        self.device.wait_idle().unwrap();

        let (caps, formats, present_modes) = self.surf.compatibility(&self.adapter.physical_device);
        debug![
            self.log, "Surface capabilities";
            "capabilities" => InDebugPretty(&caps),
            "formats" => InDebugPretty(&formats);
            clone caps, formats
        ];

        assert![formats.iter().any(|f| f.contains(&self.swapconfig.format))];

        let pixels = self.get_window_size_in_pixels();
        info![self.log, "New window size"; "size" => InDebug(&pixels)];

        let extent = Extent2D {
            width: pixels.0,
            height: pixels.1,
        };

        let mut swap_config = SwapchainConfig::from_caps(&caps, self.swapconfig.format, extent);
        self.swapconfig.extent = swap_config.extent;

        let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .cloned()
                .unwrap_or(formats[0])
        });

        debug![self.log, "Format chosen"; "format" => InDebugPretty(&format); clone format];
        debug![self.log, "Available present modes"; "modes" => InDebugPretty(&present_modes); clone present_modes];

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
        debug![self.log, "Using best possible present mode"; "mode" => InDebug(&present_mode)];

        let image_count = if present_mode == Mailbox {
            (caps.image_count.end() - 1)
                .min(3)
                .max(*caps.image_count.start())
        } else {
            (caps.image_count.end() - 1)
                .min(2)
                .max(*caps.image_count.start())
        };
        debug![self.log, "Using swapchain images"; "count" => image_count];

        swap_config.present_mode = present_mode;
        swap_config.image_count = image_count;

        if caps.usage.contains(image::Usage::TRANSFER_SRC) {
            swap_config.image_usage |= gfx_hal::image::Usage::TRANSFER_SRC;
        } else {
            warn![
                self.log,
                "Surface does not support TRANSFER_SRC, may fail during testing"
            ];
        }

        info![self.log, "Recreating swapchain"; "config" => InDebug(&swap_config); clone swap_config];
        let (swapchain, images) = unsafe {
            self.device.create_swapchain(
                &mut self.surf,
                swap_config,
                Some(ManuallyDrop::into_inner(core::ptr::read(&self.swapchain))),
            )
        }
        .expect("Unable to create swapchain");

        self.swapchain = ManuallyDrop::new(swapchain);

        // unsafe {
        //     self.device
        //         .destroy_swapchain(std::mem::replace(&mut self.swapchain, swapchain));
        //     // self.device .destroy_swapchain(ManuallyDrop::into_inner(core::ptr::read(&self.swapchain)));
        // }

        let images_string = format!["{:#?}", images];
        debug![self.log, "Image information"; "images" => images_string];

        let mut depth_images: Vec<<back::Backend as Backend>::Image> = vec![];
        let mut depth_image_views: Vec<<back::Backend as Backend>::ImageView> = vec![];
        let mut depth_image_memories: Vec<<back::Backend as Backend>::Memory> = vec![];
        let mut depth_image_requirements: Vec<memory::Requirements> = vec![];

        let (image_views, framebuffers) = {
            let image_views = images
                .iter()
                .map(|image| unsafe {
                    self.device
                        .create_image_view(
                            &image,
                            image::ViewKind::D2,
                            self.swapconfig.format, // MUST be identical to the image's format
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
                    let mut depth_image = self
                        .device
                        .create_image(
                            image::Kind::D2(
                                self.swapconfig.extent.width,
                                self.swapconfig.extent.height,
                                1,
                                1,
                            ),
                            1,
                            format::Format::D32Sfloat,
                            image::Tiling::Optimal,
                            image::Usage::DEPTH_STENCIL_ATTACHMENT,
                            image::ViewCapabilities::empty(),
                        )
                        .expect("Unable to create depth image");
                    let requirements = self.device.get_image_requirements(&depth_image);
                    let memory_type_id = find_memory_type_id(
                        &self.adapter,
                        requirements,
                        memory::Properties::DEVICE_LOCAL,
                    );
                    let memory = self
                        .device
                        .allocate_memory(memory_type_id, requirements.size)
                        .expect("Couldn't allocate image memory!");
                    self.device
                        .bind_image_memory(&memory, 0, &mut depth_image)
                        .expect("Couldn't bind the image memory!");
                    let image_view = self
                        .device
                        .create_image_view(
                            &depth_image,
                            image::ViewKind::D2,
                            format::Format::D32Sfloat,
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
                        self.device
                            .create_framebuffer(
                                &self.render_pass,
                                vec![image_view, &depth_image_views[idx]],
                                image::Extent {
                                    width: self.swapconfig.extent.width,
                                    height: self.swapconfig.extent.height,
                                    depth: 1,
                                },
                            )
                            .map_err(|_| "Failed to create a framebuffer!")
                    })
                    .collect::<Result<Vec<_>, &str>>()
                    .unwrap()
            };
            (image_views, framebuffers)
        };

        unsafe {
            for fb in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(fb);
            }
            for iv in self.image_views.drain(..) {
                self.device.destroy_image_view(iv);
            }
            for di in self.depth_images.drain(..) {
                self.device.destroy_image(di);
            }
            for div in self.depth_image_views.drain(..) {
                self.device.destroy_image_view(div);
            }
            for div in self.depth_image_memories.drain(..) {
                self.device.free_memory(div);
            }
        }

        {
            let image_views = format!["{:?}", image_views];
            debug![self.log, "Created image views"; "image views" => image_views];
        }

        let framebuffers_string = format!["{:#?}", framebuffers];
        debug![self.log, "Framebuffer information"; "framebuffers" => framebuffers_string];

        self.images = images;
        self.framebuffers = framebuffers;
        self.image_views = image_views;
        self.depth_images = depth_images;
        self.depth_image_views = depth_image_views;
        self.depth_image_memories = depth_image_memories;
        self.render_area.w = self.swapconfig.extent.width as i16;
        self.render_area.h = self.swapconfig.extent.height as i16;

        unsafe {
            self.device.destroy_semaphore(std::mem::replace(
                &mut self.acquire_image_semaphore_free,
                self.device.create_semaphore().unwrap(),
            ));
        }
    }

    /// Internal drawing routine
    fn draw_frame_internal<T>(
        &mut self,
        postproc: fn(&mut VxDraw, gfx_hal::window::SwapImageIndex) -> T,
    ) -> T {
        let view = self.perspective;
        let postproc_res = unsafe {
            let swap_image: (_, Option<gfx_hal::window::Suboptimal>) = match self
                .swapchain
                .acquire_image(
                    u64::max_value(),
                    Some(&*self.acquire_image_semaphore_free),
                    None,
                ) {
                Ok((index, None)) => (index, None),
                Ok((_index, Some(_suboptimal))) => {
                    info![self.log, "Swapchain in suboptimal state, recreating" ; "type" => "acquire_image"];
                    self.window_resized_recreate_swapchain();
                    return self.draw_frame_internal(postproc);
                }
                Err(gfx_hal::window::AcquireError::OutOfDate) => {
                    info![self.log, "Swapchain out of date, recreating"; "type" => "acquire_image"];
                    self.window_resized_recreate_swapchain();
                    return self.draw_frame_internal(postproc);
                }
                Err(err) => {
                    error![self.log, "Acquire image error"; "error" => err, "type" => "acquire_image"];
                    unimplemented![]
                }
            };

            core::mem::swap(
                &mut *self.acquire_image_semaphore_free,
                &mut self.acquire_image_semaphores[swap_image.0 as usize],
            );

            self.device
                .wait_for_fence(
                    &self.frames_in_flight_fences[self.current_frame],
                    u64::max_value(),
                )
                .unwrap();

            self.device
                .reset_fence(&self.frames_in_flight_fences[self.current_frame])
                .unwrap();

            {
                let current_frame = self.current_frame;
                let texture_count = self.dyntexs.len();
                let debugtris_cnt = self.debtris.posbuffer.len();
                let swap_image = swap_image.0;
                trace![self.log, "Drawing frame"; "swapchain image" => swap_image, "flight" => current_frame, "textures" => texture_count, "debug triangles" => debugtris_cnt];
            }

            {
                let buffer = &mut self.command_buffers[self.current_frame];
                let clear_values = [
                    ClearValue::Color(self.clear_color),
                    ClearValue::DepthStencil(gfx_hal::command::ClearDepthStencil(1.0, 0)),
                ];
                buffer.begin(false);
                let rect = pso::Rect {
                    x: 0,
                    y: 0,
                    w: self.swapconfig.extent.width as i16,
                    h: self.swapconfig.extent.height as i16,
                };
                buffer.set_viewports(
                    0,
                    std::iter::once(pso::Viewport {
                        rect,
                        depth: (0.0..1.0),
                    }),
                );
                buffer.set_scissors(0, std::iter::once(&rect));
                for text in self.texts.iter() {
                    let image_barrier = memory::Barrier::Image {
                        states: (image::Access::empty(), image::Layout::General)
                            ..(
                                image::Access::SHADER_READ,
                                image::Layout::ShaderReadOnlyOptimal,
                            ),
                        target: &*text.image_buffer,
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
                        target: &*text.image_buffer,
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
                for strtex in self.strtexs.iter() {
                    let image_barrier = memory::Barrier::Image {
                        states: (image::Access::empty(), image::Layout::General)
                            ..(
                                image::Access::SHADER_READ,
                                image::Layout::ShaderReadOnlyOptimal,
                            ),
                        target: &strtex.image_buffer[self.current_frame],
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
                        target: &strtex.image_buffer[self.current_frame],
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
                        &self.render_pass,
                        &self.framebuffers[swap_image.0 as usize],
                        self.render_area,
                        clear_values.iter(),
                    );
                    for draw_cmd in self.draw_order.iter() {
                        match draw_cmd {
                            DrawType::Text { id } => {
                                let text = &mut self.texts[*id];
                                if !text.hidden {
                                    enc.bind_graphics_pipeline(&text.pipeline);
                                    if text.posbuf_touch != 0 {
                                        text.posbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &text.posbuffer[..],
                                            );
                                        text.posbuf_touch -= 1;
                                    }
                                    if text.opacbuf_touch != 0 {
                                        text.opacbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &text.opacbuffer[..],
                                            );
                                        text.opacbuf_touch -= 1;
                                    }
                                    if text.uvbuf_touch != 0 {
                                        text.uvbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &text.uvbuffer[..],
                                            );
                                        text.uvbuf_touch -= 1;
                                    }
                                    if text.tranbuf_touch != 0 {
                                        text.tranbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &text.tranbuffer[..],
                                            );
                                        text.tranbuf_touch -= 1;
                                    }
                                    if text.rotbuf_touch != 0 {
                                        text.rotbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &text.rotbuffer[..],
                                            );
                                        text.rotbuf_touch -= 1;
                                    }
                                    if text.scalebuf_touch != 0 {
                                        text.scalebuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &text.scalebuffer[..],
                                            );
                                        text.scalebuf_touch -= 1;
                                    }
                                    let count = text.posbuffer.len();
                                    text.indices[self.current_frame].ensure_capacity(
                                        &self.device,
                                        &self.adapter,
                                        count,
                                    );
                                    let buffers: ArrayVec<[_; 6]> = [
                                        (text.posbuf[self.current_frame].buffer(), 0),
                                        (text.uvbuf[self.current_frame].buffer(), 0),
                                        (text.tranbuf[self.current_frame].buffer(), 0),
                                        (text.rotbuf[self.current_frame].buffer(), 0),
                                        (text.scalebuf[self.current_frame].buffer(), 0),
                                        (text.opacbuf[self.current_frame].buffer(), 0),
                                    ]
                                    .into();
                                    enc.push_graphics_constants(
                                        &text.pipeline_layout,
                                        pso::ShaderStageFlags::VERTEX,
                                        0,
                                        &*(view.as_ptr() as *const [u32; 16]),
                                    );
                                    enc.bind_graphics_descriptor_sets(
                                        &text.pipeline_layout,
                                        0,
                                        Some(&*text.descriptor_set),
                                        &[],
                                    );
                                    enc.bind_vertex_buffers(0, buffers);
                                    enc.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
                                        buffer: text.indices[self.current_frame].buffer(),
                                        offset: 0,
                                        index_type: gfx_hal::IndexType::U32,
                                    });
                                    enc.draw_indexed(0..text.posbuffer.len() as u32 * 6, 0, 0..1);
                                }
                            }
                            DrawType::StreamingTexture { id } => {
                                let strtex = &mut self.strtexs[*id];
                                let foot = self.device.get_image_subresource_footprint(
                                    &strtex.image_buffer[self.current_frame],
                                    image::Subresource {
                                        aspects: format::Aspects::COLOR,
                                        level: 0,
                                        layer: 0,
                                    },
                                );

                                let mut target = self
                                    .device
                                    .acquire_mapping_writer(
                                        &strtex.image_memory[self.current_frame],
                                        0..strtex.image_requirements[self.current_frame].size,
                                    )
                                    .expect("unable to acquire mapping writer");

                                for items in &strtex.circular_writes {
                                    for item in items {
                                        match item {
                                            StreamingTextureWrite::Single((x, y), color) => {
                                                if !(*x < strtex.width && *y < strtex.height) {
                                                    continue;
                                                }
                                                let access = foot.row_pitch * u64::from(*y)
                                                    + u64::from(*x * 4);
                                                target[access as usize..(access + 4) as usize]
                                                    .copy_from_slice(&[
                                                        color.0, color.1, color.2, color.3,
                                                    ]);
                                            }
                                            StreamingTextureWrite::Block((x, y), (w, h), color) => {
                                                for idx in *y..*y + h {
                                                    let pitch = foot.row_pitch as usize;
                                                    for x in *x..*x + w {
                                                        let idx = (idx as usize * pitch
                                                            + x as usize * 4)
                                                            as usize;
                                                        target[idx..idx + 4].copy_from_slice(&[
                                                            color.0, color.1, color.2, color.3,
                                                        ]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                self.device
                                    .release_mapping_writer(target)
                                    .expect("Unable to release mapping writer");
                                if !strtex.hidden {
                                    enc.bind_graphics_pipeline(&strtex.pipeline);
                                    if strtex.posbuf_touch != 0 {
                                        strtex.posbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &strtex.posbuffer[..],
                                            );
                                        strtex.posbuf_touch -= 1;
                                    }
                                    if strtex.opacbuf_touch != 0 {
                                        strtex.opacbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &strtex.opacbuffer[..],
                                            );
                                        strtex.opacbuf_touch -= 1;
                                    }
                                    if strtex.uvbuf_touch != 0 {
                                        strtex.uvbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &strtex.uvbuffer[..],
                                            );
                                        strtex.uvbuf_touch -= 1;
                                    }
                                    if strtex.tranbuf_touch != 0 {
                                        strtex.tranbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &strtex.tranbuffer[..],
                                            );
                                        strtex.tranbuf_touch -= 1;
                                    }
                                    if strtex.rotbuf_touch != 0 {
                                        strtex.rotbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &strtex.rotbuffer[..],
                                            );
                                        strtex.rotbuf_touch -= 1;
                                    }
                                    if strtex.scalebuf_touch != 0 {
                                        strtex.scalebuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &strtex.scalebuffer[..],
                                            );
                                        strtex.scalebuf_touch -= 1;
                                    }
                                    let count = strtex.posbuffer.len();
                                    strtex.indices[self.current_frame].ensure_capacity(
                                        &self.device,
                                        &self.adapter,
                                        count,
                                    );
                                    let buffers: ArrayVec<[_; 6]> = [
                                        (strtex.posbuf[self.current_frame].buffer(), 0),
                                        (strtex.uvbuf[self.current_frame].buffer(), 0),
                                        (strtex.tranbuf[self.current_frame].buffer(), 0),
                                        (strtex.rotbuf[self.current_frame].buffer(), 0),
                                        (strtex.scalebuf[self.current_frame].buffer(), 0),
                                        (strtex.opacbuf[self.current_frame].buffer(), 0),
                                    ]
                                    .into();
                                    enc.push_graphics_constants(
                                        &strtex.pipeline_layout,
                                        pso::ShaderStageFlags::VERTEX,
                                        0,
                                        &*(view.as_ptr() as *const [u32; 16]),
                                    );
                                    enc.bind_graphics_descriptor_sets(
                                        &strtex.pipeline_layout,
                                        0,
                                        Some(&strtex.descriptor_sets[self.current_frame]),
                                        &[],
                                    );
                                    enc.bind_vertex_buffers(0, buffers);
                                    enc.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
                                        buffer: strtex.indices[self.current_frame].buffer(),
                                        offset: 0,
                                        index_type: gfx_hal::IndexType::U32,
                                    });
                                    enc.draw_indexed(0..strtex.posbuffer.len() as u32 * 6, 0, 0..1);
                                }
                            }
                            DrawType::DynamicTexture { id } => {
                                let dyntex = &mut self.dyntexs[*id];
                                if !dyntex.hidden {
                                    enc.bind_graphics_pipeline(&dyntex.pipeline);
                                    if dyntex.posbuf_touch != 0 {
                                        dyntex.posbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &dyntex.posbuffer[..],
                                            );
                                        dyntex.posbuf_touch -= 1;
                                    }
                                    if dyntex.opacbuf_touch != 0 {
                                        dyntex.opacbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &dyntex.opacbuffer[..],
                                            );
                                        dyntex.opacbuf_touch -= 1;
                                    }
                                    if dyntex.uvbuf_touch != 0 {
                                        dyntex.uvbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &dyntex.uvbuffer[..],
                                            );
                                        dyntex.uvbuf_touch -= 1;
                                    }
                                    if dyntex.tranbuf_touch != 0 {
                                        dyntex.tranbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &dyntex.tranbuffer[..],
                                            );
                                        dyntex.tranbuf_touch -= 1;
                                    }
                                    if dyntex.rotbuf_touch != 0 {
                                        dyntex.rotbuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &dyntex.rotbuffer[..],
                                            );
                                        dyntex.rotbuf_touch -= 1;
                                    }
                                    if dyntex.scalebuf_touch != 0 {
                                        dyntex.scalebuf[self.current_frame]
                                            .copy_from_slice_and_maybe_resize(
                                                &self.device,
                                                &self.adapter,
                                                &dyntex.scalebuffer[..],
                                            );
                                        dyntex.scalebuf_touch -= 1;
                                    }
                                    let count = dyntex.posbuffer.len();
                                    dyntex.indices[self.current_frame].ensure_capacity(
                                        &self.device,
                                        &self.adapter,
                                        count,
                                    );
                                    let buffers: ArrayVec<[_; 6]> = [
                                        (dyntex.posbuf[self.current_frame].buffer(), 0),
                                        (dyntex.uvbuf[self.current_frame].buffer(), 0),
                                        (dyntex.tranbuf[self.current_frame].buffer(), 0),
                                        (dyntex.rotbuf[self.current_frame].buffer(), 0),
                                        (dyntex.scalebuf[self.current_frame].buffer(), 0),
                                        (dyntex.opacbuf[self.current_frame].buffer(), 0),
                                    ]
                                    .into();
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
                                    enc.bind_vertex_buffers(0, buffers);
                                    enc.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
                                        buffer: dyntex.indices[self.current_frame].buffer(),
                                        offset: 0,
                                        index_type: gfx_hal::IndexType::U32,
                                    });
                                    enc.draw_indexed(0..dyntex.posbuffer.len() as u32 * 6, 0, 0..1);
                                }
                            }
                            DrawType::Quad { id } => {
                                if let Some(quad) = self.quads.get_mut(*id) {
                                    if !quad.hidden {
                                        enc.bind_graphics_pipeline(&quad.pipeline);
                                        {
                                            let view =
                                                if let Some(ref view) = quad.fixed_perspective {
                                                    view
                                                } else {
                                                    &view
                                                };
                                            enc.push_graphics_constants(
                                                &quad.pipeline_layout,
                                                pso::ShaderStageFlags::VERTEX,
                                                0,
                                                &*(view.as_ptr() as *const [u32; 16]),
                                            );
                                        }
                                        if quad.posbuf_touch != 0 {
                                            quad.posbuf[self.current_frame]
                                                .copy_from_slice_and_maybe_resize(
                                                    &self.device,
                                                    &self.adapter,
                                                    &quad.posbuffer[..],
                                                );
                                            quad.posbuf_touch -= 1;
                                        }
                                        if quad.colbuf_touch != 0 {
                                            quad.colbuf[self.current_frame]
                                                .copy_from_slice_and_maybe_resize(
                                                    &self.device,
                                                    &self.adapter,
                                                    &quad.colbuffer[..],
                                                );
                                            quad.colbuf_touch -= 1;
                                        }
                                        if quad.tranbuf_touch != 0 {
                                            quad.tranbuf[self.current_frame]
                                                .copy_from_slice_and_maybe_resize(
                                                    &self.device,
                                                    &self.adapter,
                                                    &quad.tranbuffer[..],
                                                );
                                            quad.tranbuf_touch -= 1;
                                        }
                                        if quad.rotbuf_touch != 0 {
                                            quad.rotbuf[self.current_frame]
                                                .copy_from_slice_and_maybe_resize(
                                                    &self.device,
                                                    &self.adapter,
                                                    &quad.rotbuffer[..],
                                                );
                                            quad.rotbuf_touch -= 1;
                                        }
                                        if quad.scalebuf_touch != 0 {
                                            quad.scalebuf[self.current_frame]
                                                .copy_from_slice_and_maybe_resize(
                                                    &self.device,
                                                    &self.adapter,
                                                    &quad.scalebuffer[..],
                                                );
                                            quad.scalebuf_touch -= 1;
                                        }
                                        let count = quad.posbuffer.len();
                                        quad.indices[self.current_frame].ensure_capacity(
                                            &self.device,
                                            &self.adapter,
                                            count,
                                        );
                                        let buffers: ArrayVec<[_; 5]> = [
                                            (quad.posbuf[self.current_frame].buffer(), 0),
                                            (quad.colbuf[self.current_frame].buffer(), 0),
                                            (quad.tranbuf[self.current_frame].buffer(), 0),
                                            (quad.rotbuf[self.current_frame].buffer(), 0),
                                            (quad.scalebuf[self.current_frame].buffer(), 0),
                                        ]
                                        .into();
                                        enc.bind_vertex_buffers(0, buffers);
                                        enc.bind_index_buffer(gfx_hal::buffer::IndexBufferView {
                                            buffer: &quad.indices[self.current_frame].buffer(),
                                            offset: 0,
                                            index_type: gfx_hal::IndexType::U32,
                                        });
                                        enc.draw_indexed(
                                            0..quad.posbuffer.len() as u32 * 6,
                                            0,
                                            0..1,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    if !self.debtris.hidden {
                        enc.bind_graphics_pipeline(&self.debtris.pipeline);
                        let ratio = self.swapconfig.extent.width as f32
                            / self.swapconfig.extent.height as f32;
                        enc.push_graphics_constants(
                            &self.debtris.pipeline_layout,
                            pso::ShaderStageFlags::VERTEX,
                            0,
                            &(std::mem::transmute::<f32, [u32; 1]>(ratio)),
                        );
                        if self.debtris.posbuf_touch != 0 {
                            self.debtris.posbuf[self.current_frame]
                                .copy_from_slice_and_maybe_resize(
                                    &self.device,
                                    &self.adapter,
                                    &self.debtris.posbuffer[..],
                                );
                            self.debtris.posbuf_touch -= 1;
                        }
                        if self.debtris.colbuf_touch != 0 {
                            self.debtris.colbuf[self.current_frame]
                                .copy_from_slice_and_maybe_resize(
                                    &self.device,
                                    &self.adapter,
                                    &self.debtris.colbuffer[..],
                                );
                            self.debtris.colbuf_touch -= 1;
                        }
                        if self.debtris.tranbuf_touch != 0 {
                            self.debtris.tranbuf[self.current_frame]
                                .copy_from_slice_and_maybe_resize(
                                    &self.device,
                                    &self.adapter,
                                    &self.debtris.tranbuffer[..],
                                );
                            self.debtris.tranbuf_touch -= 1;
                        }
                        if self.debtris.rotbuf_touch != 0 {
                            self.debtris.rotbuf[self.current_frame]
                                .copy_from_slice_and_maybe_resize(
                                    &self.device,
                                    &self.adapter,
                                    &self.debtris.rotbuffer[..],
                                );
                            self.debtris.rotbuf_touch -= 1;
                        }
                        if self.debtris.scalebuf_touch != 0 {
                            self.debtris.scalebuf[self.current_frame]
                                .copy_from_slice_and_maybe_resize(
                                    &self.device,
                                    &self.adapter,
                                    &self.debtris.scalebuffer[..],
                                );
                            self.debtris.scalebuf_touch -= 1;
                        }
                        let count = self.debtris.posbuffer.len();
                        let buffers: ArrayVec<[_; 5]> = [
                            (self.debtris.posbuf[self.current_frame].buffer(), 0),
                            (self.debtris.colbuf[self.current_frame].buffer(), 0),
                            (self.debtris.tranbuf[self.current_frame].buffer(), 0),
                            (self.debtris.rotbuf[self.current_frame].buffer(), 0),
                            (self.debtris.scalebuf[self.current_frame].buffer(), 0),
                        ]
                        .into();
                        enc.bind_vertex_buffers(0, buffers);

                        enc.draw(0..(count * 3) as u32, 0..1);
                    }
                }
                buffer.finish();
            }

            let command_buffers = &self.command_buffers[self.current_frame];
            let wait_semaphores: ArrayVec<[_; 1]> = [(
                &self.acquire_image_semaphores[swap_image.0 as usize],
                pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            )]
            .into();
            {
                let present_wait_semaphore = &self.present_wait_semaphores[self.current_frame];
                let signal_semaphores: ArrayVec<[_; 1]> = [present_wait_semaphore].into();
                let submission = Submission {
                    command_buffers: once(command_buffers),
                    wait_semaphores,
                    signal_semaphores,
                };
                self.queue_group.queues[0].submit(
                    submission,
                    Some(&self.frames_in_flight_fences[self.current_frame]),
                );
            }
            let postproc_res = postproc(self, swap_image.0);
            let present_wait_semaphore = &self.present_wait_semaphores[self.current_frame];
            let present_wait_semaphores: ArrayVec<[_; 1]> = [present_wait_semaphore].into();
            match self.swapchain.present(
                &mut self.queue_group.queues[0],
                swap_image.0,
                present_wait_semaphores,
            ) {
                Ok(None) => {}
                Ok(Some(_suboptimal)) => {
                    info![
                        self.log,
                        "Swapchain in suboptimal state, recreating"; "type" => "present"
                    ];
                    self.window_resized_recreate_swapchain();
                    return self.draw_frame_internal(postproc);
                }
                Err(gfx_hal::window::PresentError::OutOfDate) => {
                    info![self.log, "Swapchain out of date, recreating"; "type" => "present"];
                    self.window_resized_recreate_swapchain();
                    return self.draw_frame_internal(postproc);
                }
                Err(err) => {
                    error![self.log, "Acquire image error"; "error" => err, "type" => "present"];
                    unimplemented![]
                }
            }
            postproc_res
        };
        self.current_frame = (self.current_frame + 1) % self.max_frames_in_flight;
        for strtex in self.strtexs.iter_mut() {
            strtex.circular_writes[self.current_frame].clear();
        }
        self.layer_holes.advance_state();
        postproc_res
    }

    /// Generate the perspective projection so that the window's size does not stretch its
    /// elements. This perspective clamps the shorter axis to -1..1 and the longer axis to whatever
    /// its aspect ratio is.
    ///
    /// This means that a window wider than tall will show a little more on the left and right edges
    /// instead of stretching the image to fill the window.
    pub fn perspective_projection(&self) -> Matrix4<f32> {
        let size = self.swapconfig.extent;
        let w_over_h = size.width as f32 / size.height as f32;
        let h_over_w = size.height as f32 / size.width as f32;
        if w_over_h >= 1.0 {
            Matrix4::from_nonuniform_scale(1.0 / w_over_h, 1.0, 1.0)
        } else {
            Matrix4::from_nonuniform_scale(1.0, 1.0 / h_over_w, 1.0)
        }
    }
}

// ---

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Deg, Rad, Vector3};
    use fast_logger::{Generic, GenericLogger, Logger};
    use test::Bencher;

    // ---

    static TESTURE: &dyntex::ImgData =
        &dyntex::ImgData::PNGBytes(include_bytes!["../images/testure.png"]);

    // ---

    #[test]
    fn setup_and_teardown() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let _ = VxDraw::new(logger, ShowWindow::Headless1k);
    }

    #[test]
    fn setup_and_teardown_draw_clear() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();

        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let img = vx.draw_frame_copy_framebuffer();

        assert_swapchain_eq(&mut vx, "setup_and_teardown_draw_with_test", img);
    }

    #[test]
    fn setup_and_teardown_custom_clear_color() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();

        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        vx.set_clear_color(Color::Rgba(255, 0, 255, 128));

        let img = vx.draw_frame_copy_framebuffer();

        assert_swapchain_eq(&mut vx, "setup_and_teardown_custom_clear_color", img);
    }

    #[test]
    fn setup_and_teardown_draw_resize() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let large_triangle = debtri::DebugTriangle::new().scale(3.7);
        vx.debtri().add(large_triangle);

        vx.draw_frame();

        vx.set_window_size((1500, 1000));

        vx.draw_frame();
        vx.draw_frame();
        vx.draw_frame();

        let img = vx.draw_frame_copy_framebuffer();

        assert_swapchain_eq(&mut vx, "setup_and_teardown_draw_resize", img);
    }

    #[test]
    fn setup_and_teardown_with_gpu_upload() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let (buffer, memory, _) =
            make_vertex_buffer_with_data_on_gpu(&mut vx, &vec![1.0f32; 10_000]);

        unsafe {
            vx.device.destroy_buffer(buffer);
            vx.device.free_memory(memory);
        }
    }

    #[test]
    fn check_world_coord_conversion() {
        let mut vx = VxDraw::new(void_logger(), ShowWindow::Headless1k);
        assert_eq![(-1.0, -1.0), vx.to_world_coords((0.0, 0.0))];

        vx.set_perspective(Matrix4::from_translation(Vector3::new(1.0, 2.0, 0.0)));
        assert_eq![(-2.0, -3.0), vx.to_world_coords((0.0, 0.0))];

        vx.set_perspective(
            Matrix4::from_translation(Vector3::new(1.0, 2.0, 0.0)) * Matrix4::from_scale(0.5),
        );
        assert_eq![(-4.0, -6.0), vx.to_world_coords((0.0, 0.0))];
    }

    #[test]
    fn init_window_and_get_input() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);
        vx.collect_input();
    }

    #[test]
    fn tearing_test() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let _tri = make_centered_equilateral_triangle();
        vx.debtri().add(debtri::DebugTriangle::default());
        for i in 0..=360 {
            if i % 2 == 0 {
                add_4_screencorners(&mut vx);
            } else {
                vx.debtri().pop_many(4);
            }
            let prspect = vx.perspective_projection();
            let rot =
                prspect * Matrix4::from_axis_angle(Vector3::new(0.0f32, 0.0, 1.0), Deg(i as f32));
            vx.set_perspective(rot);
            vx.draw_frame();
            // std::thread::sleep(std::time::Duration::new(0, 80_000_000));
        }
    }

    #[test]
    fn correct_perspective() {
        {
            let logger = Logger::<Generic>::spawn_void().to_compatibility();
            let vx = VxDraw::new(logger, ShowWindow::Headless1k);
            assert_eq![Matrix4::identity(), vx.perspective_projection()];
        }
        {
            let logger = Logger::<Generic>::spawn_void().to_compatibility();
            let vx = VxDraw::new(logger, ShowWindow::Headless1x2k);
            assert_eq![
                Matrix4::from_nonuniform_scale(1.0, 0.5, 1.0),
                vx.perspective_projection()
            ];
        }
        {
            let logger = Logger::<Generic>::spawn_void().to_compatibility();
            let vx = VxDraw::new(logger, ShowWindow::Headless2x1k);
            assert_eq![
                Matrix4::from_nonuniform_scale(0.5, 1.0, 1.0),
                vx.perspective_projection(),
            ];
        }
    }

    #[test]
    fn strtex_and_dyntex_respect_draw_order() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let options = dyntex::LayerOptions::new().depth(false);
        let tex1 = vx.dyntex().add_layer(TESTURE, &options);
        let tex2 = vx
            .strtex()
            .add_layer(&strtex::LayerOptions::new().width(1).height(1).depth(false));
        let tex3 = vx.dyntex().add_layer(TESTURE, &options);
        let tex4 = vx
            .strtex()
            .add_layer(&strtex::LayerOptions::new().width(1).height(1).depth(false));

        vx.strtex()
            .set_pixel(&tex2, 0, 0, Color::Rgba(255, 0, 255, 255));
        vx.strtex()
            .set_pixel(&tex4, 0, 0, Color::Rgba(255, 255, 255, 255));

        vx.dyntex().add(&tex1, dyntex::Sprite::new());
        vx.strtex()
            .add(&tex2, strtex::Sprite::default().rotation(Rad(0.5)));
        vx.dyntex()
            .add(&tex3, dyntex::Sprite::new().rotation(Rad(1.0)));
        vx.strtex().add(&tex4, strtex::Sprite::new().scale(0.5));

        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "strtex_and_dyntex_respect_draw_order", img);
    }

    #[test]
    fn log_adapter_works() {
        let log = Box::new(|lvl, msg| {
            struct LogAdapter {
                pub msg: Box<dyn Fn(&mut std::fmt::Formatter) -> std::fmt::Result + Send + Sync>,
            }
            impl std::fmt::Display for LogAdapter {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    (self.msg)(f)
                }
            }
            println!["{} @ {}", lvl, LogAdapter { msg }];
        });

        VxDraw::new(log, ShowWindow::Headless1k);
    }

    #[test]
    fn swap_layers() {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let options = dyntex::LayerOptions::new().depth(false);
        let tex1 = vx.dyntex().add_layer(TESTURE, &options);
        let tex2 = vx
            .strtex()
            .add_layer(&strtex::LayerOptions::new().width(1).height(1).depth(false));

        vx.strtex()
            .set_pixel(&tex2, 0, 0, Color::Rgba(255, 0, 255, 255));

        vx.dyntex().add(&tex1, dyntex::Sprite::new().scale(0.5));
        vx.strtex()
            .add(&tex2, strtex::Sprite::new().rotation(Rad(0.5)));

        vx.swap_layers(&tex1, &tex2);
        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "swap_layers", img);
    }

    #[test]
    fn swap_layers_quad() {
        use quads::{LayerOptions, Quad};
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        let quad1 = vx.quads().add_layer(&LayerOptions::default());
        vx.quads().add(&quad1, Quad::new().scale(0.25));

        let options = dyntex::LayerOptions::new().depth(false);
        let tex1 = vx.dyntex().add_layer(TESTURE, &options);

        vx.dyntex().add(&tex1, dyntex::Sprite::new().scale(0.5));

        vx.swap_layers(&tex1, &quad1);
        let img = vx.draw_frame_copy_framebuffer();
        utils::assert_swapchain_eq(&mut vx, "swap_layers_quad", img);
    }

    // ---

    #[bench]
    fn clears_per_second(b: &mut Bencher) {
        let logger = Logger::<Generic>::spawn_void().to_compatibility();
        let mut vx = VxDraw::new(logger, ShowWindow::Headless1k);

        b.iter(|| {
            vx.draw_frame();
        });
    }
}
