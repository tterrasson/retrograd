//! Executing a [`DispatchPlan`]: several dispatches, one scratch, barriers
//! between them.
//!
//! What this adds to `Pipeline` is not a loop over dispatches - `dispatch`
//! already records several - it is everything a *sequence* needs and a
//! repetition does not: buffers shared between passes, a different descriptor
//! set per pass, a different constant block per pass, and a grid per pass
//! derived from the plan's own extent expressions rather than from one
//! manifest.
//!
//! **Nothing is recomputed here either.** The plan says which buffer each
//! binding takes and through which strides; each pass's own manifest says where
//! its push constants sit and how its grid is derived from its axes. This file
//! evaluates the plan's expressions on the node's shape, hands the results to
//! the manifest reader every single dispatch already goes through, and records
//! the result. A number this file invented would be a number the shader does not
//! read.

use ash::vk;
use rir_core::Access;
use rir_core::plan::{Barrier, DispatchPlan, Extent, ExtentError, PlanError, Source};

use crate::manifest::{ManifestReader, Values};
use crate::{Arg, Manifest, RuntimeError};

use super::*;

/// An op-level argument, by the name the plan gives it.
pub struct PlanArg<'a> {
    pub name: &'a str,
    pub arg: Arg<'a>,
}

impl<'a> PlanArg<'a> {
    pub fn input<T: crate::Pod>(name: &'a str, data: &'a [T]) -> Self {
        PlanArg {
            name,
            arg: Arg::input(data),
        }
    }

    pub fn output<T: crate::Pod>(name: &'a str, data: &'a mut [T]) -> Self {
        PlanArg {
            name,
            arg: Arg::output(data),
        }
    }
}

/// A built plan: one pipeline per pass, in pass order.
pub struct Plan<'g> {
    gpu: &'g Gpu,
    plan: DispatchPlan,
    passes: Vec<Pipeline<'g>>,
}

impl<'g> Plan<'g> {
    /// Builds every pass and checks the plan against them.
    ///
    /// The check is structural (`DispatchPlan::check`) **and** relational: a
    /// pass whose plan bindings are not the bindings its manifest declares, in
    /// order, is refused here rather than dispatched with a descriptor set that
    /// happens to have the right length.
    pub fn build(
        gpu: &'g Gpu,
        plan: DispatchPlan,
        passes: &[(&Manifest, &[u32])],
    ) -> Result<Plan<'g>, RuntimeError> {
        plan.check().map_err(plan_error)?;
        if passes.len() != plan.passes.len() {
            return Err(RuntimeError::ArgCountMismatch {
                expected: plan.passes.len(),
                got: passes.len(),
            });
        }
        let mut built = Vec::with_capacity(passes.len());
        for (i, (manifest, spirv)) in passes.iter().enumerate() {
            if manifest.backend != plan.backend {
                return Err(RuntimeError::BadManifest(format!(
                    "plan '{}' is for {}, but pass {i} is for {}",
                    plan.name,
                    plan.backend.name(),
                    manifest.backend.name()
                )));
            }
            if manifest.name != plan.passes[i].artifact {
                return Err(RuntimeError::BadManifest(format!(
                    "plan '{}' pass {i} names artifact '{}', but the manifest is '{}'",
                    plan.name, plan.passes[i].artifact, manifest.name
                )));
            }
            let declared = &plan.passes[i].bindings;
            if declared.len() != manifest.bindings.len() {
                return Err(RuntimeError::ArgCountMismatch {
                    expected: manifest.bindings.len(),
                    got: declared.len(),
                });
            }
            for (d, m) in declared.iter().zip(manifest.bindings.iter()) {
                if d.name != m.name {
                    return Err(RuntimeError::BadManifest(format!(
                        "plan '{}' pass {i}: binds '{}' where the manifest declares '{}'",
                        plan.name, d.name, m.name
                    )));
                }
                if let Source::Scratch(name) = &d.source {
                    let scratch = plan
                        .scratch
                        .iter()
                        .find(|scratch| scratch.name == *name)
                        .expect("DispatchPlan::check resolved every scratch source");
                    if scratch.dtype != m.dtype {
                        return Err(RuntimeError::BadManifest(format!(
                            "plan '{}' pass {i} binds scratch '{}' as {}, but the manifest expects {}",
                            plan.name,
                            name,
                            scratch.dtype.name(),
                            m.dtype.name()
                        )));
                    }
                    match m.access {
                        Access::Write if i != scratch.produced_by => {
                            return Err(RuntimeError::BadManifest(format!(
                                "plan '{}' pass {i} writes scratch '{}', whose producer is pass {}",
                                plan.name, name, scratch.produced_by
                            )));
                        }
                        Access::Read if i == scratch.produced_by => {
                            return Err(RuntimeError::BadManifest(format!(
                                "plan '{}' pass {i} reads scratch '{}' where the plan declares its producer",
                                plan.name, name
                            )));
                        }
                        Access::Read if i > scratch.last_read_by => {
                            return Err(RuntimeError::BadManifest(format!(
                                "plan '{}' pass {i} reads scratch '{}' after its declared last reader {}",
                                plan.name, name, scratch.last_read_by
                            )));
                        }
                        _ => {}
                    }
                }
            }
            for scratch in plan
                .scratch
                .iter()
                .filter(|scratch| scratch.produced_by == i)
            {
                let produced = declared.iter().zip(manifest.bindings.iter()).any(
                    |(binding, manifest_binding)| {
                        binding.source == Source::Scratch(scratch.name.clone())
                            && manifest_binding.access == Access::Write
                    },
                );
                if !produced {
                    return Err(RuntimeError::BadManifest(format!(
                        "plan '{}' pass {i} does not write declared scratch producer '{}'",
                        plan.name, scratch.name
                    )));
                }
            }
            built.push(gpu.build(manifest, spirv)?);
        }
        Ok(Plan {
            gpu,
            plan,
            passes: built,
        })
    }

    pub fn plan(&self) -> &DispatchPlan {
        &self.plan
    }

    /// Allocates the scratch, uploads the inputs, and resolves every pass's
    /// grid and constant block for a node whose axes `axes` resolves.
    pub fn prepare<'p>(
        &'p self,
        args: &[PlanArg<'_>],
        axes: &dyn Fn(&str) -> Option<u64>,
    ) -> Result<PlanSession<'p>, RuntimeError> {
        // Buffers: the op's arguments first, in the order the caller passed
        // them, then one per scratch. The index of each is what a pass's
        // descriptor set is built from.
        let mut buffers = Vec::with_capacity(args.len() + self.plan.scratch.len());
        let mut names: Vec<String> = Vec::new();
        let mut sizes: Vec<usize> = Vec::new();
        for a in args {
            if names.iter().any(|name| name == a.name)
                || self
                    .plan
                    .scratch
                    .iter()
                    .any(|scratch| scratch.name == a.name)
            {
                return Err(RuntimeError::BadManifest(format!(
                    "plan '{}': duplicate or reserved argument name '{}'",
                    self.plan.name, a.name
                )));
            }
            let buffer = Buffer::new(self.gpu, a.arg.len())?;
            if let Arg::In(data) = &a.arg {
                buffer.write(data)?;
            }
            names.push(a.name.to_string());
            sizes.push(a.arg.len());
            buffers.push(buffer);
        }
        let mut scratch_index = Vec::with_capacity(self.plan.scratch.len());
        for s in &self.plan.scratch {
            let elements = eval_extent(&self.plan.name, &s.elements, axes)?;
            let bytes = elements
                .checked_mul(s.dtype.size_bytes() as u64)
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or_else(|| {
                    plan_error(PlanError::BudgetOverflow {
                        plan: self.plan.name.clone(),
                    })
                })?;
            scratch_index.push(buffers.len());
            names.push(s.name.clone());
            sizes.push(bytes);
            buffers.push(Buffer::new(self.gpu, bytes)?);
        }

        let mut recorded = Vec::with_capacity(self.plan.passes.len());
        for (i, pass) in self.plan.passes.iter().enumerate() {
            let manifest = self.passes[i].manifest();
            let mut values = Values::new();
            for (axis, extent) in &pass.axes {
                let n = eval_extent(&self.plan.name, extent, axes)?;
                values.u32(
                    &format!("n_{axis}"),
                    u32::try_from(n).map_err(|_| {
                        RuntimeError::BadManifest(format!("axis '{axis}' beyond the u32 range"))
                    })?,
                );
            }
            let mut indices = Vec::with_capacity(pass.bindings.len());
            for (b, m) in pass.bindings.iter().zip(manifest.bindings.iter()) {
                let index = match &b.source {
                    Source::Arg(name) => names
                        .iter()
                        .position(|n| n == name)
                        .ok_or_else(|| RuntimeError::MissingValue { name: name.clone() })?,
                    Source::Scratch(name) => {
                        let s = self
                            .plan
                            .scratch
                            .iter()
                            .position(|s| s.name == *name)
                            .ok_or_else(|| RuntimeError::MissingValue { name: name.clone() })?;
                        scratch_index[s]
                    }
                };
                if matches!(&b.source, Source::Arg(_)) {
                    let matches_access = matches!(
                        (m.access, &args[index].arg),
                        (Access::Read, Arg::In(_)) | (Access::Write, Arg::Out(_))
                    );
                    if !matches_access {
                        return Err(RuntimeError::AccessMismatch {
                            binding: format!("{}::{}", pass.artifact, b.name),
                            expected: match m.access {
                                Access::Read => "an input",
                                Access::Write => "an output",
                            },
                        });
                    }
                }
                // Strides are the plan's, in elements; the manifest's dtype
                // turns them into bytes. This is where a tiled view becomes a
                // stride list and nothing is copied.
                let unit = m.dtype.size_bytes() as u64;
                for (d, stride) in b.strides.iter().enumerate() {
                    let e = eval_extent(&self.plan.name, stride, axes)?;
                    let bytes = e.checked_mul(unit).ok_or_else(|| {
                        RuntimeError::BadManifest(format!(
                            "stride of '{}' beyond the u32 range",
                            b.name
                        ))
                    })?;
                    values.u32(
                        &format!("{}_nb{d}", b.name),
                        u32::try_from(bytes).map_err(|_| {
                            RuntimeError::BadManifest(format!(
                                "stride of '{}' beyond the u32 range",
                                b.name
                            ))
                        })?,
                    );
                }
                indices.push(index);
            }

            let push = manifest.push_bytes(&values)?;
            let groups = manifest.groups(&values)?;
            // The same bound every single dispatch checks, on buffers this
            // function sized rather than a caller: a pass whose view addresses
            // past its buffer is refused before the device sees it.
            for ((b, m), index) in pass
                .bindings
                .iter()
                .zip(manifest.bindings.iter())
                .zip(indices.iter().copied())
            {
                if let Some(need) = manifest.required_bytes(m, &values)?
                    && sizes[index] < need
                {
                    return Err(RuntimeError::BufferTooSmall {
                        binding: format!("{}::{}", pass.artifact, b.name),
                        need,
                        got: sizes[index],
                    });
                }
            }
            recorded.push(Recorded {
                push,
                groups,
                buffers: indices,
                barrier: pass.barrier,
            });
        }

        let bound = args
            .iter()
            .map(|a| BoundArg {
                write: matches!(a.arg, Arg::Out(_)),
                len: a.arg.len(),
            })
            .collect();
        Ok(PlanSession {
            plan: self,
            buffers,
            bound,
            recorded,
            in_flight: std::cell::Cell::new(false),
        })
    }
}

/// One pass, resolved for one node: its constant block, its grid, the buffers
/// its descriptor set binds, and what must precede it.
struct Recorded {
    push: Vec<u8>,
    groups: [u32; 3],
    buffers: Vec<usize>,
    barrier: Barrier,
}

/// A prepared plan: scratch allocated, inputs uploaded, every pass resolved.
pub struct PlanSession<'p> {
    plan: &'p Plan<'p>,
    buffers: Vec<Buffer>,
    bound: Vec<BoundArg>,
    recorded: Vec<Recorded>,
    in_flight: std::cell::Cell<bool>,
}

impl PlanSession<'_> {
    /// The grid of each pass, in order - what a timing has to be read against.
    pub fn groups(&self) -> Vec<[u32; 3]> {
        self.recorded.iter().map(|r| r.groups).collect()
    }

    /// Runs the whole plan once.
    pub fn dispatch(&self) -> Result<(), RuntimeError> {
        self.dispatch_n(1)
    }

    /// `n` complete runs of the plan in one submission, always separated: a
    /// repetition writes the same scratch the previous one read, so overlapping
    /// two of them would be a race. A plan has no unbarriered timing for that
    /// reason, and `Session::time_stream`'s trick does not transfer here.
    fn dispatch_n(&self, n: u32) -> Result<(), RuntimeError> {
        if self.in_flight.get() {
            return Err(RuntimeError::InFlight);
        }
        self.record(n)
            .map_err(|DispatchError { error, in_flight }| {
                self.in_flight.set(in_flight);
                error
            })
    }

    /// Times `iters` complete runs after `warmup` uncounted ones, in one
    /// submission, and returns the mean per run - the same shape of number
    /// `Session::time` returns, for the same reason.
    pub fn time(&self, warmup: u32, iters: u32) -> Result<std::time::Duration, RuntimeError> {
        if warmup > 0 {
            self.dispatch_n(warmup)?;
        }
        let iters = iters.max(1);
        let start = std::time::Instant::now();
        self.dispatch_n(iters)?;
        Ok(start.elapsed() / iters)
    }

    /// Copies the op's outputs back. Scratch is never read back: it does not
    /// exist outside the plan.
    pub fn read_outputs(&self, args: &mut [PlanArg<'_>]) -> Result<(), RuntimeError> {
        if args.len() != self.bound.len() {
            return Err(RuntimeError::ArgCountMismatch {
                expected: self.bound.len(),
                got: args.len(),
            });
        }
        for (i, a) in args.iter_mut().enumerate() {
            if matches!(a.arg, Arg::Out(_)) != self.bound[i].write {
                return Err(RuntimeError::AccessMismatch {
                    binding: a.name.to_string(),
                    expected: if self.bound[i].write {
                        "the output bound during preparation"
                    } else {
                        "the input bound during preparation"
                    },
                });
            }
            if let Arg::Out(data) = &mut a.arg {
                if data.len() != self.bound[i].len {
                    return Err(RuntimeError::ArgLenMismatch {
                        binding: a.name.to_string(),
                        expected: self.bound[i].len,
                        got: data.len(),
                    });
                }
                self.buffers[i].read(data)?;
            }
        }
        Ok(())
    }

    /// One command buffer: every pass of every run, with a full barrier wherever
    /// the plan declares one and between runs.
    fn record(&self, runs: u32) -> Result<(), DispatchError> {
        let gpu = self.plan.gpu;
        let _guard = gpu.submit.lock().unwrap_or_else(|e| e.into_inner());
        let dev = &gpu.device;

        let sets = self.recorded.len() as u32;
        let descriptors: u32 = self.recorded.iter().map(|r| r.buffers.len() as u32).sum();
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(descriptors.max(1))];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&sizes)
            .max_sets(sets.max(1));
        let pool = unsafe { dev.create_descriptor_pool(&pool_info, None) }
            .map_err(vk_err("vkCreateDescriptorPool"))?;

        let run = || -> Result<(), DispatchError> {
            // One descriptor set per pass, written once and reused by every run:
            // the buffers do not change between runs, only the order of the
            // dispatches does.
            let mut pass_sets = Vec::with_capacity(self.recorded.len());
            for (i, r) in self.recorded.iter().enumerate() {
                let layouts = [self.plan.passes[i].set_layout];
                let alloc = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts);
                let set = unsafe { dev.allocate_descriptor_sets(&alloc) }
                    .map_err(vk_err("vkAllocateDescriptorSets"))?[0];
                let infos: Vec<[vk::DescriptorBufferInfo; 1]> = r
                    .buffers
                    .iter()
                    .map(|b| {
                        [vk::DescriptorBufferInfo::default()
                            .buffer(self.buffers[*b].buffer)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]
                    })
                    .collect();
                let writes: Vec<vk::WriteDescriptorSet> = infos
                    .iter()
                    .zip(self.plan.passes[i].manifest().bindings.iter())
                    .map(|(info, b)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(b.binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(info)
                    })
                    .collect();
                unsafe { dev.update_descriptor_sets(&writes, &[]) };
                pass_sets.push(set);
            }

            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(gpu.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe { dev.allocate_command_buffers(&alloc) }
                .map_err(vk_err("vkAllocateCommandBuffers"))?[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            unsafe {
                dev.begin_command_buffer(cmd, &begin)
                    .map_err(vk_err("vkBeginCommandBuffer"))?;
                for run in 0..runs.max(1) {
                    for (i, r) in self.recorded.iter().enumerate() {
                        // The plan's barrier, plus one between runs: the first
                        // pass of run n+1 rewrites what the last pass of run n
                        // read.
                        if r.barrier == Barrier::Full || (i == 0 && run > 0) {
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
                        let pipeline = &self.plan.passes[i];
                        dev.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline.pipeline,
                        );
                        dev.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline.layout,
                            0,
                            &[pass_sets[i]],
                            &[],
                        );
                        if !r.push.is_empty() {
                            dev.cmd_push_constants(
                                cmd,
                                pipeline.layout,
                                vk::ShaderStageFlags::COMPUTE,
                                0,
                                &r.push,
                            );
                        }
                        dev.cmd_dispatch(cmd, r.groups[0], r.groups[1], r.groups[2]);
                    }
                }
                dev.end_command_buffer(cmd)
                    .map_err(vk_err("vkEndCommandBuffer"))?;
            }

            let cmds = [cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
            let fence = unsafe { dev.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(vk_err("vkCreateFence"))?;
            let (result, in_flight) = match unsafe { dev.queue_submit(gpu.queue, &submit, fence) } {
                Err(code) => (Err(vk_err("vkQueueSubmit")(code)), false),
                Ok(()) => match unsafe { dev.wait_for_fences(&[fence], true, DISPATCH_TIMEOUT_NS) }
                {
                    Ok(()) => (Ok(()), false),
                    Err(code) => (
                        Err(vk_err("vkWaitForFences")(code)),
                        unsafe { dev.device_wait_idle() }.is_err(),
                    ),
                },
            };
            if !in_flight {
                unsafe {
                    dev.destroy_fence(fence, None);
                    dev.free_command_buffers(gpu.command_pool, &cmds);
                }
            }
            result.map_err(|error| DispatchError { error, in_flight })
        };

        let out = run();
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

impl Drop for PlanSession<'_> {
    fn drop(&mut self) {
        if self.in_flight.get() {
            // Same rule as `Session`: an incomplete dispatch may still read
            // these buffers, and the scratch is among them.
            for b in self.buffers.drain(..) {
                std::mem::forget(b);
            }
        }
    }
}

/// A plan refused for what it says about itself, in the runtime's vocabulary.
/// `BadManifest` and not a variant of its own: it is the same class of failure,
/// a generated artifact this runtime cannot execute - and the message carries
/// the plan's own words.
fn plan_error(e: PlanError) -> RuntimeError {
    RuntimeError::BadManifest(e.to_string())
}

fn eval_extent(
    plan: &str,
    extent: &Extent,
    axes: &dyn Fn(&str) -> Option<u64>,
) -> Result<u64, RuntimeError> {
    match extent.eval(axes) {
        Ok(value) => Ok(value),
        Err(ExtentError::UnknownAxis(axis)) => Err(plan_error(PlanError::UnknownAxis {
            plan: plan.to_string(),
            axis,
        })),
        Err(ExtentError::Overflow) => Err(plan_error(PlanError::BudgetOverflow {
            plan: plan.to_string(),
        })),
    }
}
