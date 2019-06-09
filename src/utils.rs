//! Various utilities and helpers for vxdraw
use crate::data::VxDraw;
use cgmath::Matrix4;
use fast_logger::error;
#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
use gfx_backend_gl as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use gfx_hal::{
    adapter::{MemoryTypeId, PhysicalDevice},
    command,
    device::Device,
    format, image, memory,
    memory::Properties,
    pso, Adapter, Backend,
};
use std::f32::consts::PI;
use std::iter::once;
use std::mem::ManuallyDrop;

// ---

/// Trait for describing layers and their ordering
pub trait Layerable {
    /// Get the layer number from this layer
    fn get_layer(&self, vx: &VxDraw) -> usize;
}

/// Find the memory type id that satisfies the requirements and the memory properties for the given
/// adapter
pub(crate) fn find_memory_type_id<B: gfx_hal::Backend>(
    adap: &Adapter<B>,
    reqs: memory::Requirements,
    prop: memory::Properties,
) -> MemoryTypeId {
    adap.physical_device
        .memory_properties()
        .memory_types
        .iter()
        .enumerate()
        .find(|&(id, memory_type)| {
            reqs.type_mask & (1 << id) != 0 && memory_type.properties.contains(prop)
        })
        .map(|(id, _)| MemoryTypeId(id))
        .expect("Unable to find memory type id")
}

pub(crate) fn make_vertex_buffer_with_data(
    s: &mut VxDraw,
    data: &[f32],
) -> (
    <back::Backend as Backend>::Buffer,
    <back::Backend as Backend>::Memory,
    memory::Requirements,
) {
    let device = &s.device;
    let (buffer, memory, requirements) = unsafe {
        let buffer_size: u64 = (std::mem::size_of::<f32>() * data.len()) as u64;
        let mut buffer = device
            .create_buffer(buffer_size, gfx_hal::buffer::Usage::VERTEX)
            .expect("cant make bf");
        let requirements = device.get_buffer_requirements(&buffer);
        let memory_type_id = find_memory_type_id(
            &s.adapter,
            requirements,
            Properties::CPU_VISIBLE | Properties::COHERENT,
        );
        let memory = device
            .allocate_memory(memory_type_id, requirements.size)
            .expect("Couldn't allocate vertex buffer memory");
        device
            .bind_buffer_memory(&memory, 0, &mut buffer)
            .expect("Couldn't bind the buffer memory!");
        (buffer, memory, requirements)
    };
    unsafe {
        let mut data_target = device
            .acquire_mapping_writer(&memory, 0..requirements.size)
            .expect("Failed to acquire a memory writer!");
        data_target[..data.len()].copy_from_slice(data);
        device
            .release_mapping_writer(data_target)
            .expect("Couldn't release the mapping writer!");
    }
    (buffer, memory, requirements)
}

/// A more opinionated resizable buffer
#[derive(Debug)]
pub(crate) struct ResizBufIdx4 {
    buffer: ManuallyDrop<<back::Backend as Backend>::Buffer>,
    memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
    capacity: usize,
}

impl ResizBufIdx4 {
    pub(crate) fn new(device: &back::Device, adapter: &Adapter<back::Backend>) -> Self {
        Self::with_capacity(device, adapter, 1)
    }

    pub(crate) fn with_capacity(
        device: &back::Device,
        adapter: &Adapter<back::Backend>,
        capacity: usize,
    ) -> Self {
        let (buffer, memory, requirements) = unsafe {
            let buffer_size: u64 = (capacity * 6 * std::mem::size_of::<u32>()) as u64;
            let mut buffer = device
                .create_buffer(buffer_size, gfx_hal::buffer::Usage::INDEX)
                .expect("cant make bf");
            let requirements = device.get_buffer_requirements(&buffer);
            let memory_type_id = find_memory_type_id(
                adapter,
                requirements,
                Properties::COHERENT | Properties::CPU_VISIBLE | Properties::DEVICE_LOCAL,
            );
            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .expect("Couldn't allocate vertex buffer memory");
            device
                .bind_buffer_memory(&memory, 0, &mut buffer)
                .expect("Couldn't bind the buffer memory!");
            (buffer, memory, requirements)
        };
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer(&memory, 0..requirements.size)
                .expect("Failed to acquire a memory writer!");
            for index in 0..capacity {
                let ver = (index * 6) as u32;
                let ind = (index * 4) as u32;
                data_target[ver as usize..(ver + 6) as usize].copy_from_slice(&[
                    ind,
                    ind + 1,
                    ind + 2,
                    ind + 2,
                    ind + 3,
                    ind,
                ]);
            }
            device
                .release_mapping_writer(data_target)
                .expect("Couldn't release the mapping writer!");
        }
        Self {
            buffer: ManuallyDrop::new(buffer),
            memory: ManuallyDrop::new(memory),
            capacity,
        }
    }

    pub(crate) fn buffer(&self) -> &<back::Backend as Backend>::Buffer {
        &self.buffer
    }

    fn resize(&mut self, device: &back::Device, adapter: &Adapter<back::Backend>, capacity: usize) {
        let mut new_resizbuf = Self::with_capacity(device, adapter, capacity);
        std::mem::swap(&mut self.buffer, &mut new_resizbuf.buffer);
        std::mem::swap(&mut self.memory, &mut new_resizbuf.memory);
        std::mem::swap(&mut self.capacity, &mut new_resizbuf.capacity);
        new_resizbuf.destroy(device);
    }

    pub(crate) fn ensure_capacity(
        &mut self,
        device: &back::Device,
        adapter: &Adapter<back::Backend>,
        capacity: usize,
    ) {
        static SHRINK_TRESHOLD: usize = 2;

        let capacity = capacity.max(1);

        if self.capacity >= capacity * SHRINK_TRESHOLD {
            self.resize(device, adapter, (self.capacity / 2).max(capacity));
        } else if self.capacity >= capacity {
        } else {
            self.resize(device, adapter, (self.capacity * 2).max(capacity));
        }
    }

    pub(crate) fn destroy(&mut self, device: &back::Device) {
        use core::ptr::read;
        unsafe {
            device.destroy_buffer(ManuallyDrop::into_inner(read(&self.buffer)));
            device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
        }
    }
}

#[derive(Debug)]
pub(crate) struct ResizBuf {
    buffer: ManuallyDrop<<back::Backend as Backend>::Buffer>,
    memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
    requirements: memory::Requirements,
    capacity_in_bytes: usize,
}

impl ResizBuf {
    pub(crate) fn new(device: &back::Device, adapter: &Adapter<back::Backend>) -> Self {
        Self::with_capacity(device, adapter, 1)
    }

    pub(crate) fn with_capacity(
        device: &back::Device,
        adapter: &Adapter<back::Backend>,
        capacity_in_bytes: usize,
    ) -> Self {
        let (buffer, memory, requirements) = unsafe {
            let buffer_size: u64 = capacity_in_bytes as u64;
            let mut buffer = device
                .create_buffer(buffer_size, gfx_hal::buffer::Usage::VERTEX)
                .expect("cant make bf");
            let requirements = device.get_buffer_requirements(&buffer);
            let memory_type_id = find_memory_type_id(
                adapter,
                requirements,
                Properties::COHERENT | Properties::CPU_VISIBLE | Properties::DEVICE_LOCAL,
            );
            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .expect("Couldn't allocate vertex buffer memory");
            device
                .bind_buffer_memory(&memory, 0, &mut buffer)
                .expect("Couldn't bind the buffer memory!");
            (buffer, memory, requirements)
        };
        Self {
            buffer: ManuallyDrop::new(buffer),
            memory: ManuallyDrop::new(memory),
            requirements,
            capacity_in_bytes,
        }
    }

    pub(crate) fn buffer(&self) -> &<back::Backend as Backend>::Buffer {
        &self.buffer
    }

    fn resize(
        &mut self,
        device: &back::Device,
        adapter: &Adapter<back::Backend>,
        capacity_in_bytes: usize,
    ) {
        let mut new_resizbuf = Self::with_capacity(device, adapter, capacity_in_bytes);
        std::mem::swap(self, &mut new_resizbuf);
        new_resizbuf.destroy(device);
    }

    pub(crate) fn copy_from_slice_and_maybe_resize<T: Copy>(
        &mut self,
        device: &back::Device,
        adapter: &Adapter<back::Backend>,
        slice: &[T],
    ) {
        static SHRINK_TRESHOLD: u64 = 2;

        let bytes_in_slice = (slice.len() * std::mem::size_of::<T>()).max(1) as u64;

        if self.capacity_in_bytes as u64 >= bytes_in_slice * SHRINK_TRESHOLD {
            self.resize(
                device,
                adapter,
                (self.capacity_in_bytes as usize / 2).max(bytes_in_slice as usize),
            );
            self.copy_from_slice_and_maybe_resize(device, adapter, slice);
        } else if self.capacity_in_bytes as u64 >= bytes_in_slice {
            unsafe {
                let mut data_target = device
                    .acquire_mapping_writer(&self.memory, 0..self.requirements.size)
                    .expect("Failed to acquire a memory writer!");
                data_target[..slice.len()].copy_from_slice(slice);
                device
                    .release_mapping_writer(data_target)
                    .expect("Couldn't release the mapping writer!");
            }
        } else {
            self.resize(
                device,
                adapter,
                (self.capacity_in_bytes as usize * 2).max(bytes_in_slice as usize),
            );
            self.copy_from_slice_and_maybe_resize(device, adapter, slice);
        }
    }

    pub(crate) fn destroy(&mut self, device: &back::Device) {
        use core::ptr::read;
        unsafe {
            device.destroy_buffer(ManuallyDrop::into_inner(read(&self.buffer)));
            device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
        }
    }
}

pub(crate) fn make_transfer_buffer_of_size(
    s: &mut VxDraw,
    size: u64,
) -> (
    <back::Backend as Backend>::Buffer,
    <back::Backend as Backend>::Memory,
    memory::Requirements,
) {
    let device = &s.device;
    let (buffer, memory, requirements) = unsafe {
        let mut buffer = device
            .create_buffer(size, gfx_hal::buffer::Usage::TRANSFER_DST)
            .expect("cant make bf");
        let requirements = device.get_buffer_requirements(&buffer);
        let memory_type_id = find_memory_type_id(
            &s.adapter,
            requirements,
            Properties::CPU_VISIBLE | Properties::COHERENT,
        );
        let memory = device
            .allocate_memory(memory_type_id, requirements.size)
            .expect("Couldn't allocate vertex buffer memory");
        device
            .bind_buffer_memory(&memory, 0, &mut buffer)
            .expect("Couldn't bind the buffer memory!");
        (buffer, memory, requirements)
    };
    (buffer, memory, requirements)
}

pub(crate) fn make_transfer_img_of_size(
    s: &mut VxDraw,
    w: u32,
    h: u32,
) -> (
    <back::Backend as Backend>::Image,
    <back::Backend as Backend>::Memory,
    memory::Requirements,
) {
    let device = &s.device;
    let (buffer, memory, requirements) = unsafe {
        if s.adapter
            .physical_device
            .image_format_properties(
                format::Format::Rgba8Unorm,
                2,
                image::Tiling::Linear,
                image::Usage::TRANSFER_SRC | image::Usage::TRANSFER_DST,
                image::ViewCapabilities::empty(),
            )
            .is_none()
        {
            const MSG: &str = "Device does not support VK_FORMAT_R8G8B8A8_UNORM transfer image";
            error![s.log, "vxdraw", "{}", MSG];
            panic![MSG];
        }
        if !s
            .adapter
            .physical_device
            .format_properties(Some(format::Format::Rgba8Unorm))
            .linear_tiling
            .contains(format::ImageFeature::BLIT_DST)
        {
            const MSG: &str =
                "Device does not support VK_FORMAT_R8G8B8A8_UNORM as blit destination";
            error![s.log, "vxdraw", "{}", MSG];
            panic![MSG];
        }
        let mut buffer = device
            .create_image(
                image::Kind::D2(w, h, 1, 1),
                1,
                format::Format::Rgba8Unorm,
                image::Tiling::Linear,
                image::Usage::TRANSFER_DST | image::Usage::TRANSFER_SRC,
                image::ViewCapabilities::empty(),
            )
            .expect("cant make bf");
        let requirements = device.get_image_requirements(&buffer);
        let memory_type_id = find_memory_type_id(
            &s.adapter,
            requirements,
            Properties::CPU_VISIBLE | Properties::COHERENT,
        );
        let memory = device
            .allocate_memory(memory_type_id, requirements.size)
            .expect("Couldn't allocate image buffer memory");
        device
            .bind_image_memory(&memory, 0, &mut buffer)
            .expect("Couldn't bind the buffer memory!");
        (buffer, memory, requirements)
    };
    (buffer, memory, requirements)
}

#[cfg(test)]
pub(crate) fn make_vertex_buffer_with_data_on_gpu(
    s: &mut VxDraw,
    data: &[f32],
) -> (
    <back::Backend as Backend>::Buffer,
    <back::Backend as Backend>::Memory,
    memory::Requirements,
) {
    let device = &s.device;
    let (buffer, memory, requirements) = unsafe {
        let buffer_size: u64 = (std::mem::size_of::<f32>() * data.len()) as u64;
        let mut buffer = device
            .create_buffer(buffer_size, gfx_hal::buffer::Usage::TRANSFER_SRC)
            .expect("cant make bf");
        let requirements = device.get_buffer_requirements(&buffer);
        let memory_type_id = find_memory_type_id(
            &s.adapter,
            requirements,
            Properties::CPU_VISIBLE | Properties::COHERENT,
        );
        let memory = device
            .allocate_memory(memory_type_id, requirements.size)
            .expect("Couldn't allocate vertex buffer memory");
        device
            .bind_buffer_memory(&memory, 0, &mut buffer)
            .expect("Couldn't bind the buffer memory!");
        (buffer, memory, requirements)
    };
    unsafe {
        let mut data_target = device
            .acquire_mapping_writer(&memory, 0..requirements.size)
            .expect("Failed to acquire a memory writer!");
        data_target[..data.len()].copy_from_slice(data);
        device
            .release_mapping_writer(data_target)
            .expect("Couldn't release the mapping writer!");
    }

    let (buffer_gpu, memory_gpu, memory_gpu_requirements) = unsafe {
        let buffer_size: u64 = (std::mem::size_of::<f32>() * data.len()) as u64;
        let mut buffer = device
            .create_buffer(
                buffer_size,
                gfx_hal::buffer::Usage::TRANSFER_DST | gfx_hal::buffer::Usage::VERTEX,
            )
            .expect("cant make bf");
        let requirements = device.get_buffer_requirements(&buffer);
        let memory_type_id =
            find_memory_type_id(&s.adapter, requirements, Properties::DEVICE_LOCAL);
        let memory = device
            .allocate_memory(memory_type_id, requirements.size)
            .expect("Couldn't allocate vertex buffer memory");
        device
            .bind_buffer_memory(&memory, 0, &mut buffer)
            .expect("Couldn't bind the buffer memory!");
        (buffer, memory, requirements)
    };
    let buffer_size: u64 = (std::mem::size_of::<f32>() * data.len()) as u64;
    let mut cmd_buffer = s
        .command_pool
        .acquire_command_buffer::<gfx_hal::command::OneShot>();
    unsafe {
        cmd_buffer.begin();
        let buffer_barrier = gfx_hal::memory::Barrier::Buffer {
            families: None,
            range: None..None,
            states: gfx_hal::buffer::Access::empty()..gfx_hal::buffer::Access::TRANSFER_WRITE,
            target: &buffer_gpu,
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::TRANSFER,
            gfx_hal::memory::Dependencies::empty(),
            &[buffer_barrier],
        );
        let copy = once(command::BufferCopy {
            src: 0,
            dst: 0,
            size: buffer_size,
        });
        cmd_buffer.copy_buffer(&buffer, &buffer_gpu, copy);
        let buffer_barrier = gfx_hal::memory::Barrier::Buffer {
            families: None,
            range: None..None,
            states: gfx_hal::buffer::Access::TRANSFER_WRITE..gfx_hal::buffer::Access::SHADER_READ,
            target: &buffer_gpu,
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::TRANSFER..pso::PipelineStage::FRAGMENT_SHADER,
            gfx_hal::memory::Dependencies::empty(),
            &[buffer_barrier],
        );
        cmd_buffer.finish();
        let upload_fence = device
            .create_fence(false)
            .expect("Couldn't create an upload fence!");
        s.queue_group.queues[0].submit_nosemaphores(Some(&cmd_buffer), Some(&upload_fence));
        device
            .wait_for_fence(&upload_fence, core::u64::MAX)
            .expect("Couldn't wait for the fence!");
        device.destroy_fence(upload_fence);
        device.destroy_buffer(buffer);
        device.free_memory(memory);
    }
    (buffer_gpu, memory_gpu, memory_gpu_requirements)
}

pub(crate) fn make_centered_equilateral_triangle() -> [f32; 6] {
    let mut tri = [0.0f32; 6];
    tri[2] = 1.0f32 * (60.0f32 / 180.0f32 * PI).cos();
    tri[3] = -1.0f32 * (60.0f32 / 180.0f32 * PI).sin();
    tri[4] = 1.0f32;
    let avg_x = (tri[0] + tri[2] + tri[4]) / 3.0f32;
    let avg_y = (tri[1] + tri[3] + tri[5]) / 3.0f32;
    tri[0] -= avg_x;
    tri[1] -= avg_y;
    tri[2] -= avg_x;
    tri[3] -= avg_y;
    tri[4] -= avg_x;
    tri[5] -= avg_y;
    tri
}

/// Generate a perspective that scales the view according to the window
///
/// This means that a window wider than tall will show a little more on the left and right edges
/// instead of stretching the image to fill the window.
pub fn gen_perspective(s: &VxDraw) -> Matrix4<f32> {
    let size = s.swapconfig.extent;
    let w_over_h = size.width as f32 / size.height as f32;
    let h_over_w = size.height as f32 / size.width as f32;
    if w_over_h >= 1.0 {
        Matrix4::from_nonuniform_scale(1.0 / w_over_h, 1.0, 1.0)
    } else {
        Matrix4::from_nonuniform_scale(1.0, 1.0 / h_over_w, 1.0)
    }
}

pub(crate) fn copy_image_to_rgb(
    s: &mut VxDraw,
    image_index: gfx_hal::window::SwapImageIndex,
) -> Vec<u8> {
    let width = s.swapconfig.extent.width;
    let height = s.swapconfig.extent.height;

    let (buffer, memory, requirements) =
        make_transfer_buffer_of_size(s, u64::from(width * height * 4));
    let (imgbuf, imgmem, _imgreq) = make_transfer_img_of_size(s, width, height);
    let images = &s.images;
    unsafe {
        s.device
            .wait_for_fence(
                &s.frames_in_flight_fences[s.current_frame],
                u64::max_value(),
            )
            .expect("Unable to wait for fence");
    }
    unsafe {
        let mut cmd_buffer = s
            .command_pool
            .acquire_command_buffer::<gfx_hal::command::OneShot>();
        cmd_buffer.begin();
        let image_barrier = gfx_hal::memory::Barrier::Image {
            states: (gfx_hal::image::Access::empty(), image::Layout::Present)
                ..(
                    gfx_hal::image::Access::TRANSFER_READ,
                    image::Layout::TransferSrcOptimal,
                ),
            target: &images[image_index as usize],
            families: None,
            range: image::SubresourceRange {
                aspects: format::Aspects::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
        };
        let dstbarrier = gfx_hal::memory::Barrier::Image {
            states: (gfx_hal::image::Access::empty(), image::Layout::Undefined)
                ..(
                    gfx_hal::image::Access::TRANSFER_WRITE,
                    image::Layout::General,
                ),
            target: &imgbuf,
            families: None,
            range: image::SubresourceRange {
                aspects: format::Aspects::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::TRANSFER,
            gfx_hal::memory::Dependencies::empty(),
            &[image_barrier, dstbarrier],
        );
        cmd_buffer.blit_image(
            &images[image_index as usize],
            image::Layout::TransferSrcOptimal,
            &imgbuf,
            image::Layout::General,
            image::Filter::Nearest,
            once(command::ImageBlit {
                src_subresource: image::SubresourceLayers {
                    aspects: format::Aspects::COLOR,
                    level: 0,
                    layers: 0..1,
                },
                src_bounds: image::Offset { x: 0, y: 0, z: 0 }..image::Offset {
                    x: width as i32,
                    y: height as i32,
                    z: 1,
                },
                dst_subresource: image::SubresourceLayers {
                    aspects: format::Aspects::COLOR,
                    level: 0,
                    layers: 0..1,
                },
                dst_bounds: image::Offset { x: 0, y: 0, z: 0 }..image::Offset {
                    x: width as i32,
                    y: height as i32,
                    z: 1,
                },
            }),
        );
        let image_barrier = gfx_hal::memory::Barrier::Image {
            states: (
                gfx_hal::image::Access::TRANSFER_READ,
                image::Layout::TransferSrcOptimal,
            )..(gfx_hal::image::Access::empty(), image::Layout::Present),
            target: &images[image_index as usize],
            families: None,
            range: image::SubresourceRange {
                aspects: format::Aspects::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::TRANSFER..pso::PipelineStage::BOTTOM_OF_PIPE,
            gfx_hal::memory::Dependencies::empty(),
            &[image_barrier],
        );
        cmd_buffer.finish();
        let the_command_queue = &mut s.queue_group.queues[0];
        let fence = s
            .device
            .create_fence(false)
            .expect("Unable to create fence");
        the_command_queue.submit_nosemaphores(once(&cmd_buffer), Some(&fence));
        s.device
            .wait_for_fence(&fence, u64::max_value())
            .expect("unable to wait for fence");
        s.device.destroy_fence(fence);
    }
    unsafe {
        let mut cmd_buffer = s
            .command_pool
            .acquire_command_buffer::<gfx_hal::command::OneShot>();
        cmd_buffer.begin();
        let image_barrier = gfx_hal::memory::Barrier::Image {
            states: (gfx_hal::image::Access::empty(), image::Layout::Undefined)
                ..(
                    gfx_hal::image::Access::TRANSFER_READ,
                    image::Layout::TransferSrcOptimal,
                ),
            target: &imgbuf,
            families: None,
            range: image::SubresourceRange {
                aspects: format::Aspects::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::TRANSFER,
            gfx_hal::memory::Dependencies::empty(),
            &[image_barrier],
        );
        cmd_buffer.copy_image_to_buffer(
            &imgbuf,
            image::Layout::TransferSrcOptimal,
            &buffer,
            once(command::BufferImageCopy {
                buffer_offset: 0,
                buffer_width: width,
                buffer_height: height,
                image_layers: image::SubresourceLayers {
                    aspects: format::Aspects::COLOR,
                    level: 0,
                    layers: 0..1,
                },
                image_offset: image::Offset { x: 0, y: 0, z: 0 },
                image_extent: image::Extent {
                    width,
                    height,
                    depth: 1,
                },
            }),
        );
        cmd_buffer.finish();
        let the_command_queue = &mut s.queue_group.queues[0];
        let fence = s
            .device
            .create_fence(false)
            .expect("Unable to create fence");
        the_command_queue.submit_nosemaphores(once(&cmd_buffer), Some(&fence));
        s.device
            .wait_for_fence(&fence, u64::max_value())
            .expect("unable to wait for fence");
        s.device.destroy_fence(fence);
        s.command_pool.free(once(cmd_buffer));
    }
    unsafe {
        let reader = s
            .device
            .acquire_mapping_reader::<u8>(&memory, 0..requirements.size as u64)
            .expect("Unable to open reader");
        assert![u64::from(4 * width * height) <= requirements.size];
        let result = reader
            .iter()
            .take((4 * width * height) as usize)
            .cloned()
            .collect::<Vec<_>>();
        s.device.release_mapping_reader(reader);
        s.device.destroy_buffer(buffer);
        s.device.free_memory(memory);
        s.device.destroy_image(imgbuf);
        s.device.free_memory(imgmem);
        result
    }
}

#[cfg(test)]
pub(crate) fn assert_swapchain_eq(vx: &mut VxDraw, name: &str, rgb: Vec<u8>) {
    use ::image as load_image;
    use load_image::ImageDecoder;
    use std::io::Read;

    let rgb = {
        let mut tmp = vec![];
        for rgba in rgb.chunks_exact(4) {
            tmp.extend(&rgba[0..3]);
        }
        tmp
    };

    std::fs::create_dir_all("_build/vxdraw_results").expect("Unable to create directories");

    let genname = String::from("_build/vxdraw_results/") + name + ".png";
    let correctname = String::from("tests/vxdraw/") + name + ".png";
    let diffname = String::from("_build/vxdraw_results/") + name + "#diff.png";
    let appendname = String::from("_build/vxdraw_results/") + name + "#sum.png";

    let store_generated_image = || {
        let file = std::fs::File::create(&genname).expect("Unable to create file");

        let enc = load_image::png::PNGEncoder::new(file);
        enc.encode(
            &rgb,
            vx.swapconfig.extent.width,
            vx.swapconfig.extent.height,
            load_image::ColorType::RGB(8),
        )
        .expect("Unable to encode PNG file");
    };

    let correct = match std::fs::File::open(&correctname) {
        Ok(x) => x,
        Err(err) => {
            store_generated_image();
            std::process::Command::new("feh")
                .args(&[genname])
                .output()
                .expect("Failed to execute process");
            panic!["Unable to open reference file: {}", err]
        }
    };

    let dec = load_image::png::PNGDecoder::new(correct)
        .expect("Unable to read PNG file (does it exist?)");

    if (
        u64::from(vx.swapconfig.extent.width),
        u64::from(vx.swapconfig.extent.height),
    ) != dec.dimensions()
    {
        store_generated_image();
        assert_eq![
            (
                u64::from(vx.swapconfig.extent.width),
                u64::from(vx.swapconfig.extent.height),
            ),
            dec.dimensions(),
            "The swapchain image and the preset correct image MUST be of the exact same size"
        ];
    }
    assert_eq![
        load_image::ColorType::RGB(8),
        dec.colortype(),
        "Both images MUST have the RGB(8) format"
    ];

    let correct_bytes = dec
        .into_reader()
        .expect("Unable to read file")
        .bytes()
        .map(|x| x.expect("Unable to read byte"))
        .collect::<Vec<u8>>();

    fn absdiff(lhs: u8, rhs: u8) -> u8 {
        if let Some(newbyte) = lhs.checked_sub(rhs) {
            newbyte
        } else {
            rhs - lhs
        }
    }

    if correct_bytes != rgb {
        // width*height*(r + g + b)
        // ________________________   < 1.0
        //  255 * width * height * 3

        #[cfg(not(feature = "exact"))]
        {
            let mut sum_diff: u64 = 0;
            for (idx, byte) in correct_bytes.iter().enumerate() {
                sum_diff += absdiff(*byte, rgb[idx]) as u64;
            }

            let width = vx.swapconfig.extent.width as f64;
            let height = vx.swapconfig.extent.height as f64;
            let diff_coeff = sum_diff as f64 / 255.0 / width / height / 3.0;

            const DIFF_COEFF: f64 = 0.003;
            dbg![diff_coeff];
            if diff_coeff > DIFF_COEFF {
                // Continue
            } else {
                dbg!["WARNING: Images were NOT exact, but difference is below treshold"];
                return;
            }
        }

        store_generated_image();

        let mut diff = Vec::with_capacity(correct_bytes.len());
        for (idx, byte) in correct_bytes.iter().enumerate() {
            diff.push(absdiff(*byte, rgb[idx]));
        }
        let file = std::fs::File::create(&diffname).expect("Unable to create file");
        let enc = load_image::png::PNGEncoder::new(file);
        enc.encode(
            &diff,
            vx.swapconfig.extent.width,
            vx.swapconfig.extent.height,
            load_image::ColorType::RGB(8),
        )
        .expect("Unable to encode PNG file");
        std::process::Command::new("convert")
            .args(&[
                "-bordercolor".into(),
                "black".into(),
                "-border".into(),
                "20".into(),
                correctname,
                genname,
                diffname,
                "+append".into(),
                appendname.clone(),
            ])
            .output()
            .expect("Failed to execute process");
        std::process::Command::new("feh")
            .args(&[appendname])
            .output()
            .expect("Failed to execute process");
        panic!["Images were NOT the same!"];
    }
}

#[cfg(test)]
pub(crate) fn add_windmills(vx: &mut VxDraw, rand_rotat: bool) -> Vec<super::debtri::Handle> {
    use cgmath::Rad;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg as random;
    let mut rng = random::new(0);
    let mut debtris = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let tri = super::debtri::DebugTriangle::new();
        let (dx, dy) = (
            rng.gen_range(-1.0f32, 1.0f32),
            rng.gen_range(-1.0f32, 1.0f32),
        );
        let scale = rng.gen_range(0.03f32, 0.1f32);
        let tri = if rand_rotat {
            tri.rotation(Rad(rng.gen_range(-PI, PI)))
        } else {
            tri
        };
        let tri = tri.scale(scale).translation((dx, dy));
        debtris.push(vx.debtri().add(tri));
    }
    debtris
}

#[cfg(test)]
pub(crate) fn remove_windmills(vx: &mut VxDraw) {
    vx.debtri().pop_many(1000);
}

#[cfg(test)]
pub(crate) fn add_4_screencorners(vx: &mut VxDraw) {
    vx.debtri().add(super::debtri::DebugTriangle::from([
        -1.0f32, -1.0, 0.0, -1.0, -1.0, 0.0,
    ]));
    vx.debtri().add(super::debtri::DebugTriangle::from([
        -1.0f32, 1.0, 0.0, 1.0, -1.0, 0.0,
    ]));
    vx.debtri().add(super::debtri::DebugTriangle::from([
        1.0f32, -1.0, 0.0, -1.0, 1.0, 0.0,
    ]));
    vx.debtri().add(super::debtri::DebugTriangle::from([
        1.0f32, 1.0, 0.0, 1.0, 1.0, 0.0,
    ]));
}
