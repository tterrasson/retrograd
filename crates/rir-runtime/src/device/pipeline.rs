//! What belongs to one kernel: SPIR-V module, descriptor layout, pipeline, and
//! the one-call `run` that uploads, dispatches and reads back.

use std::cell::Cell;

use ash::vk;

use crate::manifest::{Manifest, ManifestDispatch};
use crate::{Arg, RuntimeError};

use super::*;

/// Objects created during `build`, destroyed if construction fails along the
/// way. `vkDestroy*` on a null handle is a no-op, so no test is needed: objects
/// not yet created remain null.
pub(crate) struct PartialPipeline<'d> {
    pub(crate) dev: &'d ash::Device,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) module: vk::ShaderModule,
}

impl PartialPipeline<'_> {
    /// Disarms the guard and returns handles to their final owner.
    pub(crate) fn into_parts(
        self,
    ) -> (
        vk::DescriptorSetLayout,
        vk::PipelineLayout,
        vk::ShaderModule,
    ) {
        let parts = (self.set_layout, self.layout, self.module);
        std::mem::forget(self);
        parts
    }
}

impl Drop for PartialPipeline<'_> {
    fn drop(&mut self) {
        unsafe {
            self.dev.destroy_shader_module(self.module, None);
            self.dev.destroy_pipeline_layout(self.layout, None);
            self.dev
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// A kernel pipeline, reusable across `run` calls.
pub struct Pipeline<'g> {
    pub(crate) gpu: &'g Gpu,
    pub(crate) manifest: Manifest,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) module: vk::ShaderModule,
    pub(crate) pipeline: vk::Pipeline,
}

impl Pipeline<'_> {
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// A complete dispatch: upload inputs, execute, and read outputs back into
    /// bound slices. Synchronous - waits for completion.
    ///
    /// `args` follows manifest binding order; `values` supplies its declared
    /// push constants, from which the number of
    /// workgroups.
    pub fn run(&self, args: &mut [Arg], values: &crate::Values) -> Result<(), RuntimeError> {
        let session = self.prepare(args, values)?;
        session.dispatch()?;
        session.read_outputs(args)
    }

    /// The first half of `run`: checks, buffer allocation, input upload, and
    /// resolution of grid and push constants. Its result can be dispatched any
    /// number of times without touching host memory - making `Session::time`
    /// measurable.
    ///
    /// The session borrows the pipeline: buffers bound to the dispatch remain
    /// valid for the session's lifetime.
    pub fn prepare(
        &self,
        args: &[Arg],
        values: &crate::Values,
    ) -> Result<Session<'_>, RuntimeError> {
        // The checks and the encoding are the manifest's, not the device's, and
        // they are shared with the CUDA half.
        let (push, groups) = self.manifest.encode_dispatch(args, values)?;

        let mut buffers = Vec::with_capacity(args.len());
        for a in args.iter() {
            buffers.push(Buffer::new(self.gpu, a.len())?);
        }
        // All memory is host-visible: inputs are uploaded before dispatch and
        // outputs read back afterward.
        for (b, a) in buffers.iter().zip(args.iter()) {
            if let Arg::In(data) = a {
                b.write(data)?;
            }
        }

        let bound = args
            .iter()
            .map(|a| BoundArg {
                write: matches!(a, Arg::Out(_)),
                len: a.len(),
            })
            .collect();
        Ok(Session {
            pipeline: self,
            buffers,
            bound,
            push,
            groups,
            in_flight: Cell::new(false),
        })
    }

    /// Records and submits `repeats` identical dispatches in **one** command
    /// buffer and waits for completion.
    ///
    /// `repeats > 1` exists only for timing: on macOS, submission and fence wait
    /// cost a fixed several-millisecond quantum that overwhelms shader time and
    /// makes measurements irreproducible between runs. Amortizing this cost over
    /// N executions reveals the kernel.
    ///
    /// `separated` says what the N executions are. With a barrier between them
    /// they follow rather than overlap, so N cost N times one - a **latency**,
    /// and the number every caller of `Session::time` has always read. Without
    /// it the device may overlap them, and what is measured is a
    /// **throughput**: N independent executions of a kernel that reads and
    /// writes distinct buffers, which is a legitimate thing to measure and a
    /// different one (`Session::time_stream`). The distinction is not academic
    /// on this box - a Vulkan memory barrier on MoltenVK ends the Metal encoder,
    /// which costs more than any kernel this repo generates, so a latency here
    /// measures the barrier and nothing else.
    pub(crate) fn dispatch(
        &self,
        buffers: &[Buffer],
        push: &[u8],
        groups: [u32; 3],
        repeats: u32,
        separated: bool,
    ) -> Result<(), DispatchError> {
        // Queue and command pool are shared by all calls: submission is
        // serialized (see `Gpu::submit`). A poisoned mutex invalidates nothing
        // in Vulkan - the protected state is the device itself.
        let _guard = self.gpu.submit.lock().unwrap_or_else(|e| e.into_inner());

        let dev = &self.gpu.device;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(buffers.len() as u32)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&sizes)
            .max_sets(1);
        let pool = unsafe { dev.create_descriptor_pool(&pool_info, None) }
            .map_err(vk_err("vkCreateDescriptorPool"))?;

        let run = || -> Result<(), DispatchError> {
            let layouts = [self.set_layout];
            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            let set = unsafe { dev.allocate_descriptor_sets(&alloc) }
                .map_err(vk_err("vkAllocateDescriptorSets"))?[0];

            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .zip(self.manifest.bindings.iter())
                .map(|(info, b)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(b.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            unsafe { dev.update_descriptor_sets(&writes, &[]) };

            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.gpu.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe { dev.allocate_command_buffers(&alloc) }
                .map_err(vk_err("vkAllocateCommandBuffers"))?[0];

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                dev.begin_command_buffer(cmd, &begin)
                    .map_err(vk_err("vkBeginCommandBuffer"))?;
                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                dev.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[set],
                    &[],
                );
                if !push.is_empty() {
                    dev.cmd_push_constants(
                        cmd,
                        self.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push,
                    );
                }
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                for i in 0..repeats.max(1) {
                    if i > 0 && separated {
                        dev.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &barrier,
                            &[],
                            &[],
                        );
                    }
                    dev.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
                }
                dev.end_command_buffer(cmd)
                    .map_err(vk_err("vkEndCommandBuffer"))?;
            }

            let cmds = [cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
            let fence_info = vk::FenceCreateInfo::default();
            let fence =
                unsafe { dev.create_fence(&fence_info, None) }.map_err(vk_err("vkCreateFence"))?;

            // A `vkQueueSubmit` failure puts nothing in flight; a
            // `vkWaitForFences` timeout does. The dispatch may still execute,
            // and its fence, command buffer, descriptors, and buffers are still
            // owned by it and must not be destroyed. Give the device a chance
            // to recover; if `vkDeviceWaitIdle` also fails, nothing is freeable.
            let (result, in_flight) =
                match unsafe { dev.queue_submit(self.gpu.queue, &submit, fence) } {
                    Err(code) => (Err(vk_err("vkQueueSubmit")(code)), false),
                    Ok(()) => {
                        match unsafe { dev.wait_for_fences(&[fence], true, DISPATCH_TIMEOUT_NS) } {
                            Ok(()) => (Ok(()), false),
                            Err(code) => (
                                Err(vk_err("vkWaitForFences")(code)),
                                unsafe { dev.device_wait_idle() }.is_err(),
                            ),
                        }
                    }
                };

            if !in_flight {
                unsafe {
                    dev.destroy_fence(fence, None);
                    dev.free_command_buffers(self.gpu.command_pool, &cmds);
                }
            }
            result.map_err(|error| DispatchError { error, in_flight })
        };

        let out = run();
        // Same reason: a descriptor set still referenced by an in-flight
        // dispatch must not be destroyed.
        if !matches!(
            out,
            Err(DispatchError {
                in_flight: true,
                ..
            })
        ) {
            unsafe { dev.destroy_descriptor_pool(pool, None) };
        }
        out
    }
}

impl Drop for Pipeline<'_> {
    fn drop(&mut self) {
        let dev = &self.gpu.device;
        unsafe {
            let _ = dev.device_wait_idle();
            dev.destroy_pipeline(self.pipeline, None);
            dev.destroy_shader_module(self.module, None);
            dev.destroy_pipeline_layout(self.layout, None);
            dev.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
