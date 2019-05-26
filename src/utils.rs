use crate::data::Windowing;
use cgmath::Matrix4;
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
    command::{self, BufferCopy},
    device::Device,
    format, image, memory,
    memory::Properties,
    pso, Adapter, Backend,
};
use std::f32::consts::PI;
use std::iter::once;

// ---

/// Find the memory type id that satisfies the requirements and the memory properties for the given
/// adapter
pub fn find_memory_type_id<B: gfx_hal::Backend>(
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

pub fn make_vertex_buffer_with_data(
    s: &mut Windowing,
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
        let memory_type_id = find_memory_type_id(&s.adapter, requirements, Properties::CPU_VISIBLE);
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

pub fn make_index_buffer_with_data(
    s: &mut Windowing,
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
            .create_buffer(buffer_size, gfx_hal::buffer::Usage::INDEX)
            .expect("cant make bf");
        let requirements = device.get_buffer_requirements(&buffer);
        let memory_type_id = find_memory_type_id(&s.adapter, requirements, Properties::CPU_VISIBLE);
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

pub fn make_transfer_buffer_of_size(
    s: &mut Windowing,
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
        let memory_type_id = find_memory_type_id(&s.adapter, requirements, Properties::CPU_VISIBLE);
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

pub fn make_transfer_img_of_size(
    s: &mut Windowing,
    w: u32,
    h: u32,
) -> (
    <back::Backend as Backend>::Image,
    <back::Backend as Backend>::Memory,
    memory::Requirements,
) {
    let device = &s.device;
    let (buffer, memory, requirements) = unsafe {
        let mut buffer = device
            .create_image(
                image::Kind::D2(w, h, 1, 1),
                1,
                format::Format::Rgb8Unorm,
                image::Tiling::Linear,
                image::Usage::TRANSFER_SRC | image::Usage::TRANSFER_DST,
                image::ViewCapabilities::empty(),
            )
            .expect("cant make bf");
        let requirements = device.get_image_requirements(&buffer);
        let memory_type_id = find_memory_type_id(&s.adapter, requirements, Properties::CPU_VISIBLE);
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

pub fn make_vertex_buffer_with_data_on_gpu(
    s: &mut Windowing,
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
        let memory_type_id = find_memory_type_id(&s.adapter, requirements, Properties::CPU_VISIBLE);
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
        let copy = once(BufferCopy {
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

pub fn make_centered_equilateral_triangle() -> [f32; 6] {
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

pub fn gen_perspective(s: &Windowing) -> Matrix4<f32> {
    let size = s.swapconfig.extent;
    let w_over_h = size.width as f32 / size.height as f32;
    let h_over_w = size.height as f32 / size.width as f32;
    if w_over_h >= 1.0 {
        Matrix4::from_nonuniform_scale(1.0 / w_over_h, 1.0, 1.0)
    } else {
        Matrix4::from_nonuniform_scale(1.0, 1.0 / h_over_w, 1.0)
    }
}

pub fn copy_image_to_rgb(
    s: &mut Windowing,
    image_index: gfx_hal::window::SwapImageIndex,
) -> Vec<u8> {
    let width = s.swapconfig.extent.width;
    let height = s.swapconfig.extent.height;

    let (buffer, memory, requirements) =
        make_transfer_buffer_of_size(s, u64::from(width * height * 3));
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
        assert![u64::from(3 * width * height) <= requirements.size];
        let result = reader
            .iter()
            .take((3 * width * height) as usize)
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

pub struct Align {
    pub access_offset: u64,
    pub how_many_bytes_you_need: u64,
    pub non_coherent_atom_size: u64,
    pub memory_size: u64,
}
pub struct AlignResult {
    pub begin: u64,
    pub end: u64,
    pub index_offset: usize,
}
pub fn perfect_mapping_alignment(align: Align) -> AlignResult {
    struct Alignment(pub u64);
    fn align_top(alignment: Alignment, value: u64) -> u64 {
        if value % alignment.0 != 0 {
            let alig = value + (alignment.0 - value % alignment.0);
            assert![alig % alignment.0 == 0];
            alig
        } else {
            value
        }
    }
    let begin = align.access_offset - align.access_offset % align.non_coherent_atom_size;
    let end = align_top(
        Alignment(align.non_coherent_atom_size),
        align.access_offset + align.how_many_bytes_you_need,
    );
    let index_offset = (align.access_offset - begin) as usize;
    AlignResult {
        begin,
        end,
        index_offset,
    }
}

#[cfg(test)]
pub fn assert_swapchain_eq(windowing: &mut Windowing, name: &str, rgb: Vec<u8>) {
    use ::image as load_image;
    use load_image::ImageDecoder;
    use std::io::Read;

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
            windowing.swapconfig.extent.width,
            windowing.swapconfig.extent.height,
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

    assert_eq![
        (
            u64::from(windowing.swapconfig.extent.width),
            u64::from(windowing.swapconfig.extent.height),
        ),
        dec.dimensions(),
        "The swapchain image and the preset correct image MUST be of the exact same size"
    ];
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
        store_generated_image();

        let mut diff = Vec::with_capacity(correct_bytes.len());
        for (idx, byte) in correct_bytes.iter().enumerate() {
            diff.push(absdiff(*byte, rgb[idx]));
        }
        let file = std::fs::File::create(&diffname).expect("Unable to create file");
        let enc = load_image::png::PNGEncoder::new(file);
        enc.encode(
            &diff,
            windowing.swapconfig.extent.width,
            windowing.swapconfig.extent.height,
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
pub fn add_windmills(
    windowing: &mut Windowing,
    rand_rotat: bool,
) -> Vec<super::debtri::DebugTriangleHandle> {
    use rand::Rng;
    use rand_pcg::Pcg64Mcg as random;
    let mut rng = random::new(0);
    let mut debtris = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let mut tri = super::debtri::DebugTriangle::default();
        let (dx, dy) = (
            rng.gen_range(-1.0f32, 1.0f32),
            rng.gen_range(-1.0f32, 1.0f32),
        );
        let scale = rng.gen_range(0.03f32, 0.1f32);
        if rand_rotat {
            tri.rotation = rng.gen_range(-PI, PI);
        }
        tri.scale = scale;
        tri.translation = (dx, dy);
        debtris.push(super::debtri::push(windowing, tri));
    }
    debtris
}

pub fn remove_windmills(windowing: &mut Windowing) {
    super::debtri::pop_many(windowing, 1000);
}

pub fn add_4_screencorners(windowing: &mut Windowing) {
    super::debtri::push(
        windowing,
        super::debtri::DebugTriangle::from([-1.0f32, -1.0, 0.0, -1.0, -1.0, 0.0]),
    );
    super::debtri::push(
        windowing,
        super::debtri::DebugTriangle::from([-1.0f32, 1.0, 0.0, 1.0, -1.0, 0.0]),
    );
    super::debtri::push(
        windowing,
        super::debtri::DebugTriangle::from([1.0f32, -1.0, 0.0, -1.0, 1.0, 0.0]),
    );
    super::debtri::push(
        windowing,
        super::debtri::DebugTriangle::from([1.0f32, 1.0, 0.0, 1.0, 1.0, 0.0]),
    );
}
