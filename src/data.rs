use cgmath::Matrix4;
use core::ptr::read;
use fast_logger::Logpass;
#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
use gfx_backend_gl as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use gfx_hal::{command::ClearColor, device::Device, Adapter, Backend};
use std::mem::ManuallyDrop;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SData {
    pub uv_begin: (f32, f32),
    pub uv_end: (f32, f32),
    pub topleft: (i32, i32),
    pub bottomright: (i32, i32),
}

#[derive(Debug)]
pub(crate) struct Text {
    pub(crate) hidden: bool,
    pub(crate) removed: Vec<std::ops::Range<usize>>,
    pub(crate) glyph_brush: glyph_brush::GlyphBrush<'static, SData>,

    pub(crate) texts: Vec<String>,
    pub(crate) font_sizes: Vec<(f32, f32)>,
    pub(crate) origin: Vec<(f32, f32)>,

    pub(crate) width: Vec<i32>,
    pub(crate) height: Vec<i32>,

    pub(crate) fixed_perspective: Option<Matrix4<f32>>,

    pub(crate) posbuf_touch: u32,
    pub(crate) opacbuf_touch: u32,
    pub(crate) uvbuf_touch: u32,
    pub(crate) tranbuf_touch: u32,
    pub(crate) rotbuf_touch: u32,
    pub(crate) scalebuf_touch: u32,

    pub(crate) posbuffer: Vec<[f32; 8]>,   // 8 per quad
    pub(crate) opacbuffer: Vec<[u8; 4]>,   // 4per quad
    pub(crate) uvbuffer: Vec<[f32; 8]>,    // 8 per quad
    pub(crate) tranbuffer: Vec<[f32; 8]>,  // 8 per quad
    pub(crate) rotbuffer: Vec<[f32; 4]>,   // 4 per quad
    pub(crate) scalebuffer: Vec<[f32; 4]>, // 4 per quad

    pub(crate) posbuf: Vec<super::utils::ResizBuf>,
    pub(crate) opacbuf: Vec<super::utils::ResizBuf>,
    pub(crate) uvbuf: Vec<super::utils::ResizBuf>,
    pub(crate) tranbuf: Vec<super::utils::ResizBuf>,
    pub(crate) rotbuf: Vec<super::utils::ResizBuf>,
    pub(crate) scalebuf: Vec<super::utils::ResizBuf>,

    pub(crate) indices: Vec<super::utils::ResizBufIdx4>,

    pub(crate) image_buffer: ManuallyDrop<<back::Backend as Backend>::Image>,
    pub(crate) image_memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
    pub(crate) image_view: ManuallyDrop<<back::Backend as Backend>::ImageView>,
    pub(crate) image_requirements: gfx_hal::memory::Requirements,

    pub(crate) sampler: ManuallyDrop<<back::Backend as Backend>::Sampler>,
    pub(crate) descriptor_pool: ManuallyDrop<<back::Backend as Backend>::DescriptorPool>,

    pub(crate) descriptor_set: ManuallyDrop<<back::Backend as Backend>::DescriptorSet>,
    pub(crate) descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub(crate) pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pub(crate) pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub(crate) render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
}

impl Text {
    pub(crate) fn destroy(mut self, device: &back::Device) {
        for mut indices in self.indices.drain(..) {
            indices.destroy(&device);
        }
        for mut posbuf in self.posbuf.drain(..) {
            posbuf.destroy(&device);
        }
        for mut opacbuf in self.opacbuf.drain(..) {
            opacbuf.destroy(&device);
        }
        for mut uvbuf in self.uvbuf.drain(..) {
            uvbuf.destroy(&device);
        }
        for mut tranbuf in self.tranbuf.drain(..) {
            tranbuf.destroy(&device);
        }
        for mut rotbuf in self.rotbuf.drain(..) {
            rotbuf.destroy(&device);
        }
        for mut scalebuf in self.scalebuf.drain(..) {
            scalebuf.destroy(&device);
        }
        unsafe {
            device.destroy_image(ManuallyDrop::into_inner(read(&self.image_buffer)));
            device.free_memory(ManuallyDrop::into_inner(read(&self.image_memory)));
            device.destroy_render_pass(ManuallyDrop::into_inner(read(&self.render_pass)));
            device.destroy_pipeline_layout(ManuallyDrop::into_inner(read(&self.pipeline_layout)));
            device.destroy_graphics_pipeline(ManuallyDrop::into_inner(read(&self.pipeline)));
            for dsl in self.descriptor_set_layouts.drain(..) {
                device.destroy_descriptor_set_layout(dsl);
            }
            device.destroy_descriptor_pool(ManuallyDrop::into_inner(read(&self.descriptor_pool)));
            device.destroy_sampler(ManuallyDrop::into_inner(read(&self.sampler)));
            device.destroy_image_view(ManuallyDrop::into_inner(read(&self.image_view)));
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum StreamingTextureWrite {
    Single((u32, u32), (u8, u8, u8, u8)),
    Block((u32, u32), (u32, u32), (u8, u8, u8, u8)),
}

/// A texture that host can read/write into directly, functions similarly to a sprite
#[derive(Debug)]
pub(crate) struct StreamingTexture {
    pub(crate) hidden: bool,
    pub(crate) removed: Vec<usize>,

    pub(crate) fixed_perspective: Option<Matrix4<f32>>,

    pub(crate) width: u32,
    pub(crate) height: u32,

    pub(crate) posbuf_touch: u32,
    pub(crate) opacbuf_touch: u32,
    pub(crate) uvbuf_touch: u32,
    pub(crate) tranbuf_touch: u32,
    pub(crate) rotbuf_touch: u32,
    pub(crate) scalebuf_touch: u32,

    pub(crate) posbuffer: Vec<[f32; 8]>,   // 8 per quad
    pub(crate) opacbuffer: Vec<[u8; 4]>,   // 16 per quad
    pub(crate) uvbuffer: Vec<[f32; 8]>,    // 8 per quad
    pub(crate) tranbuffer: Vec<[f32; 8]>,  // 8 per quad
    pub(crate) rotbuffer: Vec<[f32; 4]>,   // 4 per quad
    pub(crate) scalebuffer: Vec<[f32; 4]>, // 4 per quad

    pub(crate) posbuf: Vec<super::utils::ResizBuf>,
    pub(crate) opacbuf: Vec<super::utils::ResizBuf>,
    pub(crate) uvbuf: Vec<super::utils::ResizBuf>,
    pub(crate) tranbuf: Vec<super::utils::ResizBuf>,
    pub(crate) rotbuf: Vec<super::utils::ResizBuf>,
    pub(crate) scalebuf: Vec<super::utils::ResizBuf>,

    pub(crate) indices: Vec<super::utils::ResizBufIdx4>,

    pub(crate) image_buffer: Vec<<back::Backend as Backend>::Image>,
    pub(crate) image_memory: Vec<<back::Backend as Backend>::Memory>,
    pub(crate) image_requirements: Vec<gfx_hal::memory::Requirements>,
    pub(crate) image_view: Vec<<back::Backend as Backend>::ImageView>,
    pub(crate) descriptor_sets: Vec<<back::Backend as Backend>::DescriptorSet>,

    pub(crate) circular_writes: Vec<Vec<StreamingTextureWrite>>,

    pub(crate) sampler: ManuallyDrop<<back::Backend as Backend>::Sampler>,
    pub(crate) descriptor_pool: ManuallyDrop<<back::Backend as Backend>::DescriptorPool>,

    pub(crate) descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub(crate) pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
    pub(crate) pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub(crate) render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
}

/// Contains a single texture and associated sprites
#[derive(Debug)]
pub(crate) struct DynamicTexture {
    pub(crate) hidden: bool,
    pub(crate) removed: Vec<usize>,

    pub(crate) fixed_perspective: Option<Matrix4<f32>>,

    pub(crate) posbuf_touch: u32,
    pub(crate) opacbuf_touch: u32,
    pub(crate) uvbuf_touch: u32,
    pub(crate) tranbuf_touch: u32,
    pub(crate) rotbuf_touch: u32,
    pub(crate) scalebuf_touch: u32,

    pub(crate) posbuffer: Vec<[f32; 8]>,   // 8 per quad
    pub(crate) opacbuffer: Vec<[u8; 4]>,   // 4per quad
    pub(crate) uvbuffer: Vec<[f32; 8]>,    // 8 per quad
    pub(crate) tranbuffer: Vec<[f32; 8]>,  // 8 per quad
    pub(crate) rotbuffer: Vec<[f32; 4]>,   // 4 per quad
    pub(crate) scalebuffer: Vec<[f32; 4]>, // 4 per quad

    pub(crate) posbuf: Vec<super::utils::ResizBuf>,
    pub(crate) opacbuf: Vec<super::utils::ResizBuf>,
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

    pub(crate) holes: Vec<usize>,

    pub(crate) posbuf_touch: u32,
    pub(crate) colbuf_touch: u32,
    pub(crate) tranbuf_touch: u32,
    pub(crate) rotbuf_touch: u32,
    pub(crate) scalebuf_touch: u32,

    pub(crate) posbuffer: Vec<[f32; 6]>,   // 6 per triangle
    pub(crate) colbuffer: Vec<[u8; 12]>,   // 12 per triangle
    pub(crate) tranbuffer: Vec<[f32; 6]>,  // 6 per triangle
    pub(crate) rotbuffer: Vec<[f32; 3]>,   // 3 per triangle
    pub(crate) scalebuffer: Vec<[f32; 3]>, // 3 per triangle

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

    pub(crate) fixed_perspective: Option<Matrix4<f32>>,
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

#[derive(Clone, Debug)]
pub(crate) enum DrawType {
    StreamingTexture { id: usize },
    DynamicTexture { id: usize },
    Quad { id: usize },
    Text { id: usize },
}

pub(crate) struct LayerHoles {
    layer_holes: Vec<Vec<DrawType>>,
}

impl LayerHoles {
    pub fn new(image_count: usize) -> Self {
        Self {
            layer_holes: vec![vec![]; image_count + 1],
        }
    }

    pub fn push(&mut self, layer: DrawType) {
        self.layer_holes.last_mut().unwrap().push(layer);
    }

    pub fn advance_state(&mut self) {
        for idx in 0..self.layer_holes.len() - 1 {
            let mut tmp = vec![];
            std::mem::swap(&mut self.layer_holes[idx + 1], &mut tmp);
            self.layer_holes[idx].append(&mut tmp);
        }
    }

    pub fn find_available(&mut self, filter: impl Fn(&DrawType) -> bool) -> Option<DrawType> {
        let idx = self.layer_holes[0].iter().position(filter);

        if let Some(idx) = idx {
            Some(self.layer_holes[0].swap_remove(idx))
        } else {
            None
        }
    }
}

/// Main structure that holds all vulkan draw states
///
/// [VxDraw] is the entry-point of the library, from here all resources are spawned and managed.
///
/// This structure can safely be dropped and all associated resources will be cleaned up correctly.
pub struct VxDraw {
    pub(crate) perspective: Matrix4<f32>,

    pub(crate) draw_order: Vec<DrawType>,
    pub(crate) layer_holes: LayerHoles,

    pub(crate) texts: Vec<Text>,
    pub(crate) strtexs: Vec<StreamingTexture>,
    pub(crate) dyntexs: Vec<DynamicTexture>,
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
    pub(crate) clear_color: ClearColor,

    ////////////////////////////////////////////////////////////
    // WARNING: ORDER SENSITIVE CODE
    //
    // Re-ordering may break dependencies and cause _rare_ segmentation faults, aborts, or illegal
    // instructions.
    ////////////////////////////////////////////////////////////
    // pub(crate) device_limits: gfx_hal::Limits,
    pub(crate) device: back::Device,
    pub(crate) adapter: Adapter<back::Backend>,

    pub(crate) surf: <back::Backend as Backend>::Surface,
    #[allow(dead_code)]
    #[cfg(not(feature = "gl"))]
    pub(crate) vk_inst: back::Instance,
    #[cfg(not(feature = "gl"))]
    pub(crate) window: winit::Window,

    pub(crate) events_loop: Option<winit::EventsLoop>,
}

// ---

impl Drop for VxDraw {
    #[allow(clippy::cognitive_complexity)]
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
                for mut opacbuf in simple_tex.opacbuf.drain(..) {
                    opacbuf.destroy(&self.device);
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
                for mut indices in strtex.indices.drain(..) {
                    indices.destroy(&self.device);
                }
                for mut posbuf in strtex.posbuf.drain(..) {
                    posbuf.destroy(&self.device);
                }
                for mut opacbuf in strtex.opacbuf.drain(..) {
                    opacbuf.destroy(&self.device);
                }
                for mut uvbuf in strtex.uvbuf.drain(..) {
                    uvbuf.destroy(&self.device);
                }
                for mut tranbuf in strtex.tranbuf.drain(..) {
                    tranbuf.destroy(&self.device);
                }
                for mut rotbuf in strtex.rotbuf.drain(..) {
                    rotbuf.destroy(&self.device);
                }
                for mut scalebuf in strtex.scalebuf.drain(..) {
                    scalebuf.destroy(&self.device);
                }
                for image_buffer in strtex.image_buffer.drain(..) {
                    self.device.destroy_image(image_buffer);
                }
                for image_memory in strtex.image_memory.drain(..) {
                    self.device.free_memory(image_memory);
                }
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
                for image_view in strtex.image_view.drain(..) {
                    self.device.destroy_image_view(image_view);
                }
            }

            for text in self.texts.drain(..) {
                text.destroy(&self.device);
            }
        }
    }
}
