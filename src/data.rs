use cgmath::Matrix4;
use core::ptr::read;
#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
use gfx_backend_gl as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use gfx_hal::{device::Device, Adapter, Backend};
use logger::Logpass;
use std::mem::ManuallyDrop;

/// A texture that host can read/write into directly, functions similarly to a sprite
#[derive(Debug)]
pub(crate) struct StreamingTexture {
    pub(crate) hidden: bool,
    pub(crate) count: u32,

    pub(crate) width: u32,
    pub(crate) height: u32,

    pub(crate) vertex_buffer: ManuallyDrop<<back::Backend as Backend>::Buffer>,
    pub(crate) vertex_memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
    pub(crate) vertex_requirements: gfx_hal::memory::Requirements,

    pub(crate) vertex_buffer_indices: ManuallyDrop<<back::Backend as Backend>::Buffer>,
    pub(crate) vertex_memory_indices: ManuallyDrop<<back::Backend as Backend>::Memory>,
    pub(crate) vertex_requirements_indices: gfx_hal::memory::Requirements,

    pub(crate) image_buffer: ManuallyDrop<<back::Backend as Backend>::Image>,
    pub(crate) image_memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
    pub(crate) image_requirements: gfx_hal::memory::Requirements,

    pub(crate) sampler: ManuallyDrop<<back::Backend as Backend>::Sampler>,
    pub(crate) image_view: ManuallyDrop<<back::Backend as Backend>::ImageView>,
    pub(crate) descriptor_pool: ManuallyDrop<<back::Backend as Backend>::DescriptorPool>,

    pub(crate) descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub(crate) pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pub(crate) pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub(crate) render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
    pub(crate) descriptor_set: ManuallyDrop<<back::Backend as Backend>::DescriptorSet>,
}

/// Contains a single texture and associated sprites
#[derive(Debug)]
pub(crate) struct SingleTexture {
    pub(crate) hidden: bool,
    pub(crate) count: u32,
    pub(crate) removed: Vec<usize>,

    pub(crate) fixed_perspective: Option<Matrix4<f32>>,

    pub(crate) posbuf_touch: u32,
    pub(crate) colbuf_touch: u32,
    pub(crate) uvbuf_touch: u32,
    pub(crate) tranbuf_touch: u32,
    pub(crate) rotbuf_touch: u32,
    pub(crate) scalebuf_touch: u32,

    pub(crate) posbuffer: Vec<[f32; 8]>,   // 8 per quad
    pub(crate) colbuffer: Vec<[u8; 16]>,   // 16 per quad
    pub(crate) uvbuffer: Vec<[f32; 8]>,    // 8 per quad
    pub(crate) tranbuffer: Vec<[f32; 8]>,  // 8 per quad
    pub(crate) rotbuffer: Vec<[f32; 4]>,   // 4 per quad
    pub(crate) scalebuffer: Vec<[f32; 4]>, // 4 per quad

    pub(crate) posbuf: Vec<super::utils::ResizBuf>,
    pub(crate) colbuf: Vec<super::utils::ResizBuf>,
    pub(crate) uvbuf: Vec<super::utils::ResizBuf>,
    pub(crate) tranbuf: Vec<super::utils::ResizBuf>,
    pub(crate) rotbuf: Vec<super::utils::ResizBuf>,
    pub(crate) scalebuf: Vec<super::utils::ResizBuf>,

    pub(crate) indices: Vec<super::utils::ResizBufIdx4>,

    pub(crate) texture_image_buffer: ManuallyDrop<<back::Backend as Backend>::Image>,
    pub(crate) texture_image_memory: ManuallyDrop<<back::Backend as Backend>::Memory>,

    pub(crate) sampler: ManuallyDrop<<back::Backend as Backend>::Sampler>,
    pub(crate) image_view: ManuallyDrop<<back::Backend as Backend>::ImageView>,
    pub(crate) descriptor_pool: ManuallyDrop<<back::Backend as Backend>::DescriptorPool>,

    pub(crate) descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub(crate) pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pub(crate) pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub(crate) render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
    pub(crate) descriptor_set: ManuallyDrop<<back::Backend as Backend>::DescriptorSet>,
}

#[derive(Debug)]
pub(crate) struct DebugTriangleData {
    pub(crate) hidden: bool,
    pub(crate) triangles_count: usize,

    pub(crate) holes: Vec<usize>,

    pub(crate) posbuf_touch: u32,
    pub(crate) colbuf_touch: u32,
    pub(crate) tranbuf_touch: u32,
    pub(crate) rotbuf_touch: u32,
    pub(crate) scalebuf_touch: u32,

    pub(crate) posbuffer: Vec<[f32; 6]>, // 6 per triangle
    pub(crate) colbuffer: Vec<u8>,       // 12 per triangle
    pub(crate) tranbuffer: Vec<f32>,     // 6 per triangle
    pub(crate) rotbuffer: Vec<f32>,      // 3 per triangle
    pub(crate) scalebuffer: Vec<f32>,    // 3 per triangle

    pub(crate) posbuf: Vec<super::utils::ResizBuf>,
    pub(crate) colbuf: Vec<super::utils::ResizBuf>,
    pub(crate) tranbuf: Vec<super::utils::ResizBuf>,
    pub(crate) rotbuf: Vec<super::utils::ResizBuf>,
    pub(crate) scalebuf: Vec<super::utils::ResizBuf>,

    pub(crate) descriptor_set: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub(crate) pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pub(crate) pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub(crate) render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
}

#[derive(Debug)]
pub(crate) struct QuadsData {
    pub(crate) hidden: bool,
    pub(crate) count: usize,

    pub(crate) holes: Vec<usize>,

    pub(crate) posbuf_touch: u32,
    pub(crate) colbuf_touch: u32,
    pub(crate) tranbuf_touch: u32,
    pub(crate) rotbuf_touch: u32,
    pub(crate) scalebuf_touch: u32,

    pub(crate) posbuffer: Vec<[f32; 8]>,   // 8 per quad
    pub(crate) colbuffer: Vec<[u8; 16]>,   // 16 per quad
    pub(crate) tranbuffer: Vec<[f32; 8]>,  // 8 per quad
    pub(crate) rotbuffer: Vec<[f32; 4]>,   // 4 per quad
    pub(crate) scalebuffer: Vec<[f32; 4]>, // 4 per quad

    pub(crate) posbuf: Vec<super::utils::ResizBuf>,
    pub(crate) colbuf: Vec<super::utils::ResizBuf>,
    pub(crate) tranbuf: Vec<super::utils::ResizBuf>,
    pub(crate) rotbuf: Vec<super::utils::ResizBuf>,
    pub(crate) scalebuf: Vec<super::utils::ResizBuf>,

    pub(crate) indices: Vec<super::utils::ResizBufIdx4>,

    pub(crate) descriptor_set: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub(crate) pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pub(crate) pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub(crate) render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
}

#[derive(Debug)]
pub(crate) enum DrawType {
    StreamingTexture { id: usize },
    DynamicTexture { id: usize },
    Quad { id: usize },
}

/// Main structure that holds all vulkan draw states
///
/// [VxDraw] is the entry-point of the library, from here all resources are spawned and managed.
///
/// This structure can safely be dropped and all associated resources will be cleaned up correctly.
pub struct VxDraw {
    pub(crate) draw_order: Vec<DrawType>,
    pub(crate) strtexs: Vec<StreamingTexture>,
    pub(crate) dyntexs: Vec<SingleTexture>,
    pub(crate) quads: Vec<QuadsData>,
    pub(crate) debtris: DebugTriangleData,
    //
    pub(crate) current_frame: usize,
    pub(crate) max_frames_in_flight: usize,

    pub(crate) render_area: gfx_hal::pso::Rect,

    pub(crate) depth_images: Vec<<back::Backend as Backend>::Image>,
    pub(crate) depth_image_views: Vec<<back::Backend as Backend>::ImageView>,
    pub(crate) depth_image_memories: Vec<<back::Backend as Backend>::Memory>,

    pub(crate) frames_in_flight_fences: Vec<<back::Backend as Backend>::Fence>,
    pub(crate) acquire_image_semaphore_free: ManuallyDrop<<back::Backend as Backend>::Semaphore>,
    pub(crate) acquire_image_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    pub(crate) present_wait_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    pub(crate) command_pool:
        ManuallyDrop<gfx_hal::pool::CommandPool<back::Backend, gfx_hal::Graphics>>,
    pub(crate) framebuffers: Vec<<back::Backend as Backend>::Framebuffer>,
    pub(crate) command_buffers: Vec<
        gfx_hal::command::CommandBuffer<
            back::Backend,
            gfx_hal::Graphics,
            gfx_hal::command::MultiShot,
            gfx_hal::command::Primary,
        >,
    >,
    pub(crate) images: Vec<<back::Backend as Backend>::Image>,
    pub(crate) image_views: Vec<<back::Backend as Backend>::ImageView>,
    pub(crate) render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
    pub(crate) swapchain: ManuallyDrop<<back::Backend as Backend>::Swapchain>,
    pub(crate) swapconfig: gfx_hal::window::SwapchainConfig,
    pub(crate) format: gfx_hal::format::Format,

    pub(crate) log: Logpass,

    pub(crate) queue_group: gfx_hal::QueueGroup<back::Backend, gfx_hal::Graphics>,

    ////////////////////////////////////////////////////////////
    // WARNING: ORDER SENSITIVE CODE
    //
    // Re-ordering may break dependencies and cause _rare_ segmentation faults, aborts, or illegal
    // instructions.
    ////////////////////////////////////////////////////////////
    pub(crate) device_limits: gfx_hal::Limits,
    pub(crate) device: back::Device,
    pub(crate) adapter: Adapter<back::Backend>,

    pub(crate) surf: <back::Backend as Backend>::Surface,
    #[allow(dead_code)]
    #[cfg(not(feature = "gl"))]
    pub(crate) vk_inst: back::Instance,
    #[cfg(not(feature = "gl"))]
    pub(crate) window: winit::Window,

    pub(crate) events_loop: winit::EventsLoop,
}

// ---

impl Drop for VxDraw {
    fn drop(&mut self) {
        let _ = self.device.wait_idle();

        unsafe {
            for fence in self.frames_in_flight_fences.drain(..) {
                self.device.destroy_fence(fence);
            }
            for sema in self.acquire_image_semaphores.drain(..) {
                self.device.destroy_semaphore(sema);
            }
            for sema in self.present_wait_semaphores.drain(..) {
                self.device.destroy_semaphore(sema);
            }
            for fb in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(fb);
            }
            for iv in self.image_views.drain(..) {
                self.device.destroy_image_view(iv);
            }
            self.device.destroy_semaphore(ManuallyDrop::into_inner(read(
                &self.acquire_image_semaphore_free,
            )));

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

        unsafe {
            {
                let debtris = &mut self.debtris;
                for mut posbuf in debtris.posbuf.drain(..) {
                    posbuf.destroy(&self.device);
                }
                for mut colbuf in debtris.colbuf.drain(..) {
                    colbuf.destroy(&self.device);
                }
                for mut tranbuf in debtris.tranbuf.drain(..) {
                    tranbuf.destroy(&self.device);
                }
                for mut rotbuf in debtris.rotbuf.drain(..) {
                    rotbuf.destroy(&self.device);
                }
                for mut scalebuf in debtris.scalebuf.drain(..) {
                    scalebuf.destroy(&self.device);
                }
                for dsl in debtris.descriptor_set.drain(..) {
                    self.device.destroy_descriptor_set_layout(dsl);
                }
                self.device
                    .destroy_graphics_pipeline(ManuallyDrop::into_inner(read(&debtris.pipeline)));
                self.device
                    .destroy_pipeline_layout(ManuallyDrop::into_inner(read(
                        &debtris.pipeline_layout,
                    )));
                self.device
                    .destroy_render_pass(ManuallyDrop::into_inner(read(&debtris.render_pass)));
            }

            for mut quad in self.quads.drain(..) {
                for mut indices in quad.indices.drain(..) {
                    indices.destroy(&self.device);
                }
                for mut posbuf in quad.posbuf.drain(..) {
                    posbuf.destroy(&self.device);
                }
                for mut colbuf in quad.colbuf.drain(..) {
                    colbuf.destroy(&self.device);
                }
                for mut tranbuf in quad.tranbuf.drain(..) {
                    tranbuf.destroy(&self.device);
                }
                for mut rotbuf in quad.rotbuf.drain(..) {
                    rotbuf.destroy(&self.device);
                }
                for mut scalebuf in quad.scalebuf.drain(..) {
                    scalebuf.destroy(&self.device);
                }
                for dsl in quad.descriptor_set.drain(..) {
                    self.device.destroy_descriptor_set_layout(dsl);
                }
                self.device
                    .destroy_graphics_pipeline(ManuallyDrop::into_inner(read(&quad.pipeline)));
                self.device
                    .destroy_pipeline_layout(ManuallyDrop::into_inner(read(&quad.pipeline_layout)));
                self.device
                    .destroy_render_pass(ManuallyDrop::into_inner(read(&quad.render_pass)));
            }

            self.device.destroy_command_pool(
                ManuallyDrop::into_inner(read(&self.command_pool)).into_raw(),
            );
            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(read(&self.render_pass)));
            self.device
                .destroy_swapchain(ManuallyDrop::into_inner(read(&self.swapchain)));

            for mut simple_tex in self.dyntexs.drain(..) {
                for mut indices in simple_tex.indices.drain(..) {
                    indices.destroy(&self.device);
                }
                for mut posbuf in simple_tex.posbuf.drain(..) {
                    posbuf.destroy(&self.device);
                }
                for mut colbuf in simple_tex.colbuf.drain(..) {
                    colbuf.destroy(&self.device);
                }
                for mut uvbuf in simple_tex.uvbuf.drain(..) {
                    uvbuf.destroy(&self.device);
                }
                for mut tranbuf in simple_tex.tranbuf.drain(..) {
                    tranbuf.destroy(&self.device);
                }
                for mut rotbuf in simple_tex.rotbuf.drain(..) {
                    rotbuf.destroy(&self.device);
                }
                for mut scalebuf in simple_tex.scalebuf.drain(..) {
                    scalebuf.destroy(&self.device);
                }
                self.device.destroy_image(ManuallyDrop::into_inner(read(
                    &simple_tex.texture_image_buffer,
                )));
                self.device.free_memory(ManuallyDrop::into_inner(read(
                    &simple_tex.texture_image_memory,
                )));
                self.device
                    .destroy_render_pass(ManuallyDrop::into_inner(read(&simple_tex.render_pass)));
                self.device
                    .destroy_pipeline_layout(ManuallyDrop::into_inner(read(
                        &simple_tex.pipeline_layout,
                    )));
                self.device
                    .destroy_graphics_pipeline(ManuallyDrop::into_inner(read(
                        &simple_tex.pipeline,
                    )));
                for dsl in simple_tex.descriptor_set_layouts.drain(..) {
                    self.device.destroy_descriptor_set_layout(dsl);
                }
                self.device
                    .destroy_descriptor_pool(ManuallyDrop::into_inner(read(
                        &simple_tex.descriptor_pool,
                    )));
                self.device
                    .destroy_sampler(ManuallyDrop::into_inner(read(&simple_tex.sampler)));
                self.device
                    .destroy_image_view(ManuallyDrop::into_inner(read(&simple_tex.image_view)));
            }

            for mut strtex in self.strtexs.drain(..) {
                self.device.destroy_buffer(ManuallyDrop::into_inner(read(
                    &strtex.vertex_buffer_indices,
                )));
                self.device.free_memory(ManuallyDrop::into_inner(read(
                    &strtex.vertex_memory_indices,
                )));
                self.device
                    .destroy_buffer(ManuallyDrop::into_inner(read(&strtex.vertex_buffer)));
                self.device
                    .free_memory(ManuallyDrop::into_inner(read(&strtex.vertex_memory)));
                self.device
                    .destroy_image(ManuallyDrop::into_inner(read(&strtex.image_buffer)));
                self.device
                    .free_memory(ManuallyDrop::into_inner(read(&strtex.image_memory)));
                self.device
                    .destroy_render_pass(ManuallyDrop::into_inner(read(&strtex.render_pass)));
                self.device
                    .destroy_pipeline_layout(ManuallyDrop::into_inner(read(
                        &strtex.pipeline_layout,
                    )));
                self.device
                    .destroy_graphics_pipeline(ManuallyDrop::into_inner(read(&strtex.pipeline)));
                for dsl in strtex.descriptor_set_layouts.drain(..) {
                    self.device.destroy_descriptor_set_layout(dsl);
                }
                self.device
                    .destroy_descriptor_pool(ManuallyDrop::into_inner(read(
                        &strtex.descriptor_pool,
                    )));
                self.device
                    .destroy_sampler(ManuallyDrop::into_inner(read(&strtex.sampler)));
                self.device
                    .destroy_image_view(ManuallyDrop::into_inner(read(&strtex.image_view)));
            }
        }
    }
}
