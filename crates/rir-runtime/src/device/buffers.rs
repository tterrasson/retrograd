//! Host-visible buffers.

use ash::vk;

use crate::RuntimeError;

use super::*;

/// A host-visible storage buffer. The runtime does not seek device-local memory:
/// it validates results, not throughput.
pub(crate) struct Buffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) size: usize,
    pub(crate) device: ash::Device,
}

impl Buffer {
    pub(crate) fn new(gpu: &Gpu, size: usize) -> Result<Buffer, RuntimeError> {
        let dev = &gpu.device;
        // A Vulkan buffer has non-zero size; an empty argument remains legal for
        // the kernel (no invocation will read it).
        let size = size.max(4);
        let info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { dev.create_buffer(&info, None) }.map_err(vk_err("vkCreateBuffer"))?;
        let req = unsafe { dev.get_buffer_memory_requirements(buffer) };
        let props = unsafe {
            gpu.instance
                .get_physical_device_memory_properties(gpu.physical)
        };
        let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let Some(index) = (0..props.memory_type_count).find(|&i| {
            req.memory_type_bits & (1 << i) != 0
                && props.memory_types[i as usize].property_flags.contains(want)
        }) else {
            unsafe { dev.destroy_buffer(buffer, None) };
            return Err(RuntimeError::MissingFeature {
                feature: "host_visible_memory".into(),
                detail: "no coherent host-visible memory type".into(),
            });
        };
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(index);
        let memory = match unsafe { dev.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(code) => {
                unsafe { dev.destroy_buffer(buffer, None) };
                return Err(vk_err("vkAllocateMemory")(code));
            }
        };
        // From here both handles exist: `Buffer` owns them, and its `Drop`
        // releases them even if binding fails.
        let b = Buffer {
            buffer,
            memory,
            size,
            device: dev.clone(),
        };
        unsafe { dev.bind_buffer_memory(buffer, memory, 0) }
            .map_err(vk_err("vkBindBufferMemory"))?;
        Ok(b)
    }

    pub(crate) fn write(&self, data: &[u8]) -> Result<(), RuntimeError> {
        let ptr = unsafe {
            self.device.map_memory(
                self.memory,
                0,
                self.size as u64,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(vk_err("vkMapMemory"))?;
        // The mapped range is `self.size` bytes and `data` is no larger - the
        // buffer was created at that size. This is an assertion rather than a
        // comment because `copy_nonoverlapping` safety depends on it, and a
        // future caller could otherwise break it invisibly.
        debug_assert!(data.len() <= self.size, "write outside the mapping");
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                ptr.cast::<u8>(),
                data.len().min(self.size),
            );
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    pub(crate) fn read(&self, data: &mut [u8]) -> Result<(), RuntimeError> {
        // Final safeguard beneath `read_outputs` checks: the mapping is
        // `self.size` bytes and the copy is `data.len()`. Leaving this bound
        // implicit would make safety depend on a caller - and this API is safe.
        if data.len() > self.size {
            return Err(RuntimeError::ArgLenMismatch {
                binding: "device buffer".to_string(),
                expected: self.size,
                got: data.len(),
            });
        }
        let ptr = unsafe {
            self.device.map_memory(
                self.memory,
                0,
                self.size as u64,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(vk_err("vkMapMemory"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), data.as_mut_ptr(), data.len());
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }
}

/// A buffer releases itself: `run` creates one per argument and may fail at any
/// one, including during upload. The only case where they are not destroyed is
/// an in-flight dispatch, which is explicit
/// (`std::mem::forget`).
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
