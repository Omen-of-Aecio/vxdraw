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
pub struct StreamingTexture {
    pub count: u32,

    pub width: u32,
    pub height: u32,

    pub vertex_buffer: ManuallyDrop<<back::Backend as Backend>::Buffer>,
    pub vertex_memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
    pub vertex_requirements: gfx_hal::memory::Requirements,

    pub vertex_buffer_indices: ManuallyDrop<<back::Backend as Backend>::Buffer>,
    pub vertex_memory_indices: ManuallyDrop<<back::Backend as Backend>::Memory>,
    pub vertex_requirements_indices: gfx_hal::memory::Requirements,

    pub image_buffer: ManuallyDrop<<back::Backend as Backend>::Image>,
    pub image_memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
    pub image_requirements: gfx_hal::memory::Requirements,

    pub sampler: ManuallyDrop<<back::Backend as Backend>::Sampler>,
    pub image_view: ManuallyDrop<<back::Backend as Backend>::ImageView>,
    pub descriptor_pool: ManuallyDrop<<back::Backend as Backend>::DescriptorPool>,

    pub descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pub pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
    pub descriptor_set: ManuallyDrop<<back::Backend as Backend>::DescriptorSet>,
}

/// Contains a single texture and associated sprites
pub struct SingleTexture {
    pub count: u32,

    pub fixed_perspective: Option<Matrix4<f32>>,
    pub mockbuffer: Vec<u8>,
    pub removed: Vec<usize>,

    pub texture_vertex_buffer: ManuallyDrop<<back::Backend as Backend>::Buffer>,
    pub texture_vertex_memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
    pub texture_vertex_requirements: gfx_hal::memory::Requirements,

    pub texture_vertex_buffer_indices: ManuallyDrop<<back::Backend as Backend>::Buffer>,
    pub texture_vertex_memory_indices: ManuallyDrop<<back::Backend as Backend>::Memory>,
    pub texture_vertex_requirements_indices: gfx_hal::memory::Requirements,

    pub texture_image_buffer: ManuallyDrop<<back::Backend as Backend>::Image>,
    pub texture_image_memory: ManuallyDrop<<back::Backend as Backend>::Memory>,

    pub sampler: ManuallyDrop<<back::Backend as Backend>::Sampler>,
    pub image_view: ManuallyDrop<<back::Backend as Backend>::ImageView>,
    pub descriptor_pool: ManuallyDrop<<back::Backend as Backend>::DescriptorPool>,

    pub descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pub pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
    pub descriptor_set: ManuallyDrop<<back::Backend as Backend>::DescriptorSet>,
}

pub struct ColoredTriangleList {
    pub capacity: u64,
    pub triangles_count: usize,
    pub triangles_buffer: <back::Backend as Backend>::Buffer,
    pub triangles_memory: <back::Backend as Backend>::Memory,
    pub memory_requirements: gfx_hal::memory::Requirements,

    pub descriptor_set: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pub pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
}

pub struct ColoredQuadList {
    pub capacity: u64,
    pub count: usize,

    pub quads_buffer: <back::Backend as Backend>::Buffer,
    pub quads_memory: <back::Backend as Backend>::Memory,
    pub memory_requirements: gfx_hal::memory::Requirements,

    pub quads_buffer_indices: <back::Backend as Backend>::Buffer,
    pub quads_memory_indices: <back::Backend as Backend>::Memory,
    pub quads_requirements_indices: gfx_hal::memory::Requirements,

    pub descriptor_set: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pub pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
}

pub enum DrawType {
    StreamingTexture { id: usize },
    DynamicTexture { id: usize },
}

pub struct Windowing {
    #[cfg(not(feature = "gl"))]
    pub window: winit::Window,

    pub events_loop: winit::EventsLoop,

    pub draw_order: Vec<DrawType>,
    pub strtexs: Vec<StreamingTexture>,
    pub dyntexs: Vec<SingleTexture>,
    pub quads: Option<ColoredQuadList>,
    pub debtris: Option<ColoredTriangleList>,
    //
    pub current_frame: usize,
    pub max_frames_in_flight: usize,

    pub image_count: usize,
    pub render_area: gfx_hal::pso::Rect,

    pub depth_images: Vec<<back::Backend as Backend>::Image>,
    pub depth_image_views: Vec<<back::Backend as Backend>::ImageView>,
    pub depth_image_requirements: Vec<gfx_hal::memory::Requirements>,
    pub depth_image_memories: Vec<<back::Backend as Backend>::Memory>,

    pub frames_in_flight_fences: Vec<<back::Backend as Backend>::Fence>,
    pub acquire_image_semaphore_free: ManuallyDrop<<back::Backend as Backend>::Semaphore>,
    pub acquire_image_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    pub present_wait_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    pub command_pool: ManuallyDrop<gfx_hal::pool::CommandPool<back::Backend, gfx_hal::Graphics>>,
    pub framebuffers: Vec<<back::Backend as Backend>::Framebuffer>,
    pub command_buffers: Vec<
        gfx_hal::command::CommandBuffer<
            back::Backend,
            gfx_hal::Graphics,
            gfx_hal::command::MultiShot,
            gfx_hal::command::Primary,
        >,
    >,
    pub images: Vec<<back::Backend as Backend>::Image>,
    pub image_views: Vec<<back::Backend as Backend>::ImageView>,
    pub render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
    pub queue_group: ManuallyDrop<gfx_hal::QueueGroup<back::Backend, gfx_hal::Graphics>>,
    pub swapchain: ManuallyDrop<<back::Backend as Backend>::Swapchain>,
    pub swapconfig: gfx_hal::window::SwapchainConfig,
    pub device_limits: gfx_hal::Limits,
    pub device: ManuallyDrop<back::Device>,
    pub adapter: Adapter<back::Backend>,
    pub surf: <back::Backend as Backend>::Surface,
    pub format: gfx_hal::format::Format,

    #[cfg(not(feature = "gl"))]
    pub vk_inst: ManuallyDrop<back::Instance>,
    pub log: Logpass,
}

// ---

impl Drop for Windowing {
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
            if let Some(mut debtris) = self.debtris.take() {
                self.device.destroy_buffer(debtris.triangles_buffer);
                self.device.free_memory(debtris.triangles_memory);
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

            if let Some(mut quads) = self.quads.take() {
                self.device.destroy_buffer(quads.quads_buffer);
                self.device.free_memory(quads.quads_memory);
                self.device.destroy_buffer(quads.quads_buffer_indices);
                self.device.free_memory(quads.quads_memory_indices);
                for dsl in quads.descriptor_set.drain(..) {
                    self.device.destroy_descriptor_set_layout(dsl);
                }
                self.device
                    .destroy_graphics_pipeline(ManuallyDrop::into_inner(read(&quads.pipeline)));
                self.device
                    .destroy_pipeline_layout(ManuallyDrop::into_inner(read(
                        &quads.pipeline_layout,
                    )));
                self.device
                    .destroy_render_pass(ManuallyDrop::into_inner(read(&quads.render_pass)));
            }

            self.device.destroy_command_pool(
                ManuallyDrop::into_inner(read(&self.command_pool)).into_raw(),
            );
            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(read(&self.render_pass)));
            self.device
                .destroy_swapchain(ManuallyDrop::into_inner(read(&self.swapchain)));

            for mut simple_tex in self.dyntexs.drain(..) {
                self.device.destroy_buffer(ManuallyDrop::into_inner(read(
                    &simple_tex.texture_vertex_buffer_indices,
                )));
                self.device.free_memory(ManuallyDrop::into_inner(read(
                    &simple_tex.texture_vertex_memory_indices,
                )));
                self.device.destroy_buffer(ManuallyDrop::into_inner(read(
                    &simple_tex.texture_vertex_buffer,
                )));
                self.device.free_memory(ManuallyDrop::into_inner(read(
                    &simple_tex.texture_vertex_memory,
                )));
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

            ManuallyDrop::drop(&mut self.queue_group);
            ManuallyDrop::drop(&mut self.device);
            #[cfg(not(feature = "gl"))]
            ManuallyDrop::drop(&mut self.vk_inst);
        }
    }
}
