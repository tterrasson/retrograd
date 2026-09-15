//! Loader, instance, physical-device selection, capability probe, and the
//! logical device: everything shared between kernels.

use std::ffi::{CStr, CString};

use ash::vk;

use crate::RuntimeError;
use crate::manifest::{Manifest, ManifestReader};

use super::*;

/// Converts a `VkResult` into an error named after the failed call.
pub(crate) fn vk_err(op: &'static str) -> impl Fn(vk::Result) -> RuntimeError {
    move |code| RuntimeError::Vulkan {
        op,
        code: code.as_raw(),
    }
}

pub(crate) const EXT_8BIT: &CStr = c"VK_KHR_8bit_storage";
pub(crate) const EXT_16BIT: &CStr = c"VK_KHR_16bit_storage";
pub(crate) const EXT_F16_I8: &CStr = c"VK_KHR_shader_float16_int8";
pub(crate) const EXT_PORTABILITY: &CStr = c"VK_KHR_portability_subset";

/// Device capabilities in manifest vocabulary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeviceCaps {
    pub(crate) subgroup_size: u32,
    pub(crate) subgroup_arithmetic: bool,
    pub(crate) api_version: u32,
    pub(crate) storage_8bit: bool,
    pub(crate) storage_16bit: bool,
    pub(crate) shader_int8: bool,
    pub(crate) shader_float16: bool,
    pub(crate) max_push_size: u32,
    /// `maxComputeSharedMemorySize`. Not a feature the manifest negotiates,
    /// a kernel that overran it would fail to create its pipeline - but the one
    /// budget Vulkan publishes that a schedule spends, so the sweep reads it
    /// here rather than hard-coding 32 KiB.
    pub(crate) max_shared_bytes: u32,
}

pub struct Gpu {
    // Declaration order = destruction order: device before instance, instance
    // before loader.
    pub(crate) device: ash::Device,
    pub(crate) physical: vk::PhysicalDevice,
    pub(crate) caps: DeviceCaps,
    pub(crate) queue: vk::Queue,
    pub(crate) command_pool: vk::CommandPool,
    /// The queue and command pool require **external** synchronization:
    /// `vkQueueSubmit`, `vkAllocateCommandBuffers`, and `vkFreeCommandBuffers`
    /// are not thread-safe on the same object. `Pipeline::run` takes `&self` and
    /// the types are `Send + Sync`, so safe Rust may call them concurrently:
    /// this mutex makes that use legal. A validation runtime gains nothing from
    /// parallel dispatches.
    pub(crate) submit: std::sync::Mutex<()>,
    pub(crate) instance: ash::Instance,
    pub(crate) _entry: ash::Entry,
}

impl Gpu {
    /// Opens the first available compute device. `Err(NoLoader)` if there is no
    /// `libvulkan`, `Err(NoDevice)` if there is no GPU: these are distinct
    /// because they have different remedies.
    pub fn open() -> Result<Gpu, RuntimeError> {
        let entry = load_entry()?;

        let app = vk::ApplicationInfo::default()
            .application_name(c"rir-runtime")
            .api_version(vk::make_api_version(0, 1, 1, 0));

        // MoltenVK and similar implementations expose themselves only when
        // enumerated explicitly; elsewhere the extension is absent and left
        // untouched.
        let portability = unsafe { entry.enumerate_instance_extension_properties(None) }
            .unwrap_or_default()
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(c"VK_KHR_portability_enumeration"));
        let inst_exts: Vec<*const i8> = if portability {
            vec![c"VK_KHR_portability_enumeration".as_ptr()]
        } else {
            vec![]
        };
        let flags = if portability {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        let info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .flags(flags)
            .enabled_extension_names(&inst_exts);
        let instance = match unsafe { entry.create_instance(&info, None) } {
            Ok(instance) => instance,
            // A loader can be present while none of its installed ICDs can
            // drive this process (wrong architecture/API, stale MoltenVK
            // registration…). No shader has been compiled or submitted yet:
            // this is the same test precondition as an empty device list, not
            // evidence that generated code is wrong.
            Err(vk::Result::ERROR_INCOMPATIBLE_DRIVER) => return Err(RuntimeError::NoDevice),
            Err(code) => return Err(vk_err("vkCreateInstance")(code)),
        };

        let devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(d) => d,
            Err(code) => {
                unsafe { instance.destroy_instance(None) };
                return Err(vk_err("vkEnumeratePhysicalDevices")(code));
            }
        };
        let picked = devices
            .into_iter()
            .find_map(|pd| compute_queue_family(&instance, pd).map(|q| (pd, q)));
        let Some((physical, family)) = picked else {
            unsafe { instance.destroy_instance(None) };
            return Err(RuntimeError::NoDevice);
        };

        let caps = probe_caps(&instance, physical);
        let device = match create_device(&instance, physical, family, caps) {
            Ok(d) => d,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };
        let queue = unsafe { device.get_device_queue(family, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(code) => {
                // The device is already created: losing it here would keep the
                // GPU occupied until process exit.
                unsafe {
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(vk_err("vkCreateCommandPool")(code));
            }
        };

        Ok(Gpu {
            _entry: entry,
            instance,
            physical,
            caps,
            device,
            queue,
            command_pool,
            submit: std::sync::Mutex::new(()),
        })
    }

    /// Name of the selected device - for test messages.
    pub fn name(&self) -> String {
        let props = unsafe { self.instance.get_physical_device_properties(self.physical) };
        props
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "?".to_string())
    }

    /// Shared storage one workgroup may declare on this device, in bytes.
    /// Published by Vulkan as a limit; a schedule spends it, so a sweep divides
    /// by it rather than by a constant of its own (`Footprint`).
    pub fn max_shared_memory(&self) -> u32 {
        self.caps.max_shared_bytes
    }

    /// Checks manifest features **before** creating anything: an insufficient
    /// device must produce `MissingFeature`, not a shader that
    /// compiles and reads arbitrary data.
    pub(crate) fn check_features(&self, m: &Manifest) -> Result<(), RuntimeError> {
        let c = self.caps;
        for f in &m.features {
            let (ok, detail) = match f.as_str() {
                "vulkan>=1.0" => (true, String::new()),
                "vulkan>=1.1" => (
                    c.api_version >= vk::make_api_version(0, 1, 1, 0),
                    format!(
                        "device at {}.{}",
                        vk::api_version_major(c.api_version),
                        vk::api_version_minor(c.api_version)
                    ),
                ),
                "subgroup_arithmetic" => (c.subgroup_arithmetic, "not exposed".into()),
                "storage_buffer_8bit" => (c.storage_8bit, "VK_KHR_8bit_storage absent".into()),
                "storage_buffer_16bit" => (c.storage_16bit, "VK_KHR_16bit_storage absent".into()),
                "shader_int8" => (c.shader_int8, "shaderInt8 absent".into()),
                "shader_float16" => (c.shader_float16, "shaderFloat16 absent".into()),
                other => match other.strip_prefix("subgroup_size>=") {
                    Some(n) => {
                        let need: u32 = n.parse().unwrap_or(u32::MAX);
                        (
                            c.subgroup_size >= need,
                            format!("subgroupSize = {}", c.subgroup_size),
                        )
                    }
                    // An unknown feature is a rejection, not a shrug: the
                    // manifest is newer than this runtime.
                    None => (false, "feature unknown to this runtime".into()),
                },
            };
            if !ok {
                return Err(RuntimeError::MissingFeature {
                    feature: f.clone(),
                    detail,
                });
            }
        }
        Ok(())
    }

    /// Builds a kernel pipeline: one storage-buffer binding per manifest
    /// argument, in order, and a push-constant block of the declared size.
    pub fn build(&self, m: &Manifest, spirv: &[u32]) -> Result<Pipeline<'_>, RuntimeError> {
        self.check_features(m)?;

        // The manifest push-constant block must fit the device limit: beyond it,
        // `vkCreatePipelineLayout` fails, and `MissingFeature` says *why* this
        // kernel cannot run here.
        let push_size = m.push_size();
        if push_size > self.caps.max_push_size {
            return Err(RuntimeError::MissingFeature {
                feature: format!("push_constants>={push_size}"),
                detail: format!("maxPushConstantsSize = {}", self.caps.max_push_size),
            });
        }

        // Everything created before the pipeline is entrusted to the guard: a
        // failure along the way destroys it instead of leaking Vulkan handles
        // on every attempt with invalid SPIR-V.
        let mut partial = PartialPipeline {
            dev: &self.device,
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            module: vk::ShaderModule::null(),
        };

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = m
            .bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        partial.set_layout = unsafe { self.device.create_descriptor_set_layout(&set_info, None) }
            .map_err(vk_err("vkCreateDescriptorSetLayout"))?;

        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_size)];
        let layouts = [partial.set_layout];
        let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        if push_size > 0 {
            layout_info = layout_info.push_constant_ranges(&ranges);
        }
        partial.layout = unsafe { self.device.create_pipeline_layout(&layout_info, None) }
            .map_err(vk_err("vkCreatePipelineLayout"))?;

        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        partial.module = unsafe { self.device.create_shader_module(&module_info, None) }
            .map_err(vk_err("vkCreateShaderModule"))?;

        let entry = CString::new("main").expect("literal entry point holds no interior NUL");
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(partial.module)
            .name(&entry);
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(partial.layout)];
        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
        }
        .map_err(|(_, code)| vk_err("vkCreateComputePipelines")(code))?[0];

        // The pipeline exists: the guard transfers ownership to `Pipeline`,
        // whose `Drop` now destroys these three objects.
        let (set_layout, layout, module) = partial.into_parts();
        Ok(Pipeline {
            gpu: self,
            manifest: m.clone(),
            set_layout,
            layout,
            module,
            pipeline,
        })
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Vulkan loader locations searched when name-based `dlopen` fails.
///
/// `Entry::load()` requests `libvulkan.dylib`/`.so.1` from the dynamic loader,
/// which searches only its default paths. This is enough on Linux and with the
/// LunarG SDK, but **not** with Homebrew on Apple Silicon: `/opt/homebrew/lib`
/// is not in `dlopen`'s path, so a correctly installed MoltenVK appeared absent.
/// Known locations are therefore tried before giving up.
pub(crate) const LOADER_PATHS: &[&str] = &[
    "/opt/homebrew/lib/libvulkan.dylib", // Homebrew, Apple Silicon
    "/opt/homebrew/lib/libvulkan.1.dylib",
    "/usr/local/lib/libvulkan.dylib", // Homebrew, Intel; manually installed SDK
    "/usr/local/lib/libvulkan.1.dylib",
];

/// Escape hatch for an unusual installation: the exact loader path, which takes
/// precedence over everything else.
pub(crate) const LOADER_ENV: &str = "RIR_VULKAN_LIB";

pub(crate) fn load_entry() -> Result<ash::Entry, RuntimeError> {
    // Safe: `load`/`load_from` only open a shared library and read its symbols;
    // failure is returned rather than propagated as a panic.
    if let Some(path) = std::env::var_os(LOADER_ENV) {
        return unsafe { ash::Entry::load_from(&path) }
            .map_err(|e| RuntimeError::NoLoader(format!("{LOADER_ENV}={path:?} : {e}")));
    }
    match unsafe { ash::Entry::load() } {
        Ok(e) => Ok(e),
        Err(first) => {
            for p in LOADER_PATHS {
                if let Ok(e) = unsafe { ash::Entry::load_from(p) } {
                    return Ok(e);
                }
            }
            Err(RuntimeError::NoLoader(format!(
                "{first} - not in {} either; set {LOADER_ENV} if the loader is elsewhere",
                LOADER_PATHS.join(", ")
            )))
        }
    }
}

pub(crate) fn compute_queue_family(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
) -> Option<u32> {
    unsafe { instance.get_physical_device_queue_family_properties(pd) }
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|i| i as u32)
}

pub(crate) fn probe_caps(instance: &ash::Instance, pd: vk::PhysicalDevice) -> DeviceCaps {
    let props = unsafe { instance.get_physical_device_properties(pd) };

    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };

    let mut f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut s8 = vk::PhysicalDevice8BitStorageFeatures::default();
    let mut s16 = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut feats2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f16i8)
        .push_next(&mut s8)
        .push_next(&mut s16);
    unsafe { instance.get_physical_device_features2(pd, &mut feats2) };

    let exts = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
    let has = |name: &CStr| exts.iter().any(|e| e.extension_name_as_c_str() == Ok(name));

    // A capability is usable only when both the extension **and** feature are
    // present: either without the other yields a shader that refuses to load.
    let core12 = props.api_version >= vk::make_api_version(0, 1, 2, 0);
    DeviceCaps {
        subgroup_size: subgroup.subgroup_size,
        subgroup_arithmetic: subgroup
            .supported_operations
            .contains(vk::SubgroupFeatureFlags::ARITHMETIC)
            && subgroup
                .supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE),
        api_version: props.api_version,
        storage_8bit: (core12 || has(EXT_8BIT)) && s8.storage_buffer8_bit_access != 0,
        storage_16bit: (core12 || has(EXT_16BIT)) && s16.storage_buffer16_bit_access != 0,
        shader_int8: (core12 || has(EXT_F16_I8)) && f16i8.shader_int8 != 0,
        shader_float16: (core12 || has(EXT_F16_I8)) && f16i8.shader_float16 != 0,
        max_push_size: props.limits.max_push_constants_size,
        max_shared_bytes: props.limits.max_compute_shared_memory_size,
    }
}

pub(crate) fn create_device(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    family: u32,
    caps: DeviceCaps,
) -> Result<ash::Device, RuntimeError> {
    let priorities = [1.0f32];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priorities)];

    let exts = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
    let has = |name: &CStr| exts.iter().any(|e| e.extension_name_as_c_str() == Ok(name));

    // Enable every device capability generated kernels may require;
    // `check_features` has already rejected anything missing.
    let mut names: Vec<*const i8> = Vec::new();
    if caps.storage_8bit && has(EXT_8BIT) {
        names.push(EXT_8BIT.as_ptr());
    }
    if caps.storage_16bit && has(EXT_16BIT) {
        names.push(EXT_16BIT.as_ptr());
    }
    if (caps.shader_int8 || caps.shader_float16) && has(EXT_F16_I8) {
        names.push(EXT_F16_I8.as_ptr());
    }
    if has(EXT_PORTABILITY) {
        // Required whenever exposed (MoltenVK).
        names.push(EXT_PORTABILITY.as_ptr());
    }

    let mut f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
        .shader_int8(caps.shader_int8)
        .shader_float16(caps.shader_float16);
    let mut s8 = vk::PhysicalDevice8BitStorageFeatures::default()
        .storage_buffer8_bit_access(caps.storage_8bit);
    let mut s16 = vk::PhysicalDevice16BitStorageFeatures::default()
        .storage_buffer16_bit_access(caps.storage_16bit);
    let mut feats = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f16i8)
        .push_next(&mut s8)
        .push_next(&mut s16);

    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&names)
        .push_next(&mut feats);
    unsafe { instance.create_device(pd, &info, None) }.map_err(vk_err("vkCreateDevice"))
}
