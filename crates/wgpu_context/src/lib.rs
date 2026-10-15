// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple helpers for managing wgpu state and surfaces.

use wgpu::{
    Adapter, Backends, Device, Features, Instance, Limits, MemoryHints, Queue, Surface,
    SurfaceTarget,
};

mod alpha_blitter;
mod buffer_renderer;
mod error;
mod surface_renderer;
mod util;

pub use alpha_blitter::{AlphaConversion, AlphaConvertBlitter};
pub use buffer_renderer::{BufferRenderer, BufferRendererConfig};
pub use error::WgpuContextError;
pub use surface_renderer::{SurfaceRenderer, SurfaceRendererConfiguration, TextureConfiguration};
pub use util::block_on_wgpu;

/// What a [`WGPUContext`] is created with. Build it with [`Default`] and the builder methods;
/// [`WGPUContext::with_features_and_limits`] stays for callers that need only those two.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct WGPUContextOptions {
    /// Features to request in addition to what the renderer needs.
    pub extra_features: Option<Features>,
    /// Limits to request in place of the defaults.
    pub override_limits: Option<Limits>,
    /// The backends the instance probes. `None` leaves wgpu's default (every backend the build
    /// supports). Narrowing skips the backends that cannot work on the platform, which saves
    /// their start-up cost. The `WGPU_BACKEND` environment variable wins over this when set, and
    /// when the narrowed set yields no adapter a renderer may fall back once to the default set
    /// (see [`WGPUContext::is_narrowed`]).
    pub backends: Option<Backends>,
}

impl WGPUContextOptions {
    pub fn extra_features(self, extra_features: Features) -> Self {
        Self {
            extra_features: Some(extra_features),
            ..self
        }
    }

    pub fn override_limits(self, override_limits: Limits) -> Self {
        Self {
            override_limits: Some(override_limits),
            ..self
        }
    }

    pub fn backends(self, backends: Backends) -> Self {
        Self {
            backends: Some(backends),
            ..self
        }
    }
}

/// A wgpu `Device`, it's associated `Queue`, and the `Adapter` and `Instance` used to create them
#[derive(Clone, Debug)]
pub struct DeviceHandle {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl DeviceHandle {
    /// Creates a `DeviceHandle` with `Device` that's compatible with the specified `Surface`
    pub async fn new_from_compatible_surface(
        instance: Instance,
        compatible_surface: Option<&Surface<'_>>,
        extra_features: Option<Features>,
        override_limits: Option<Limits>,
    ) -> Result<Self, WgpuContextError> {
        let adapter =
            wgpu::util::initialize_adapter_from_env_or_default(&instance, compatible_surface)
                .await?;

        // Determine features to request
        // The user may request additional features
        let requested_features = extra_features.unwrap_or(Features::empty());
        let available_features = adapter.features();
        let required_features = requested_features & available_features;

        // Determine limits to request
        // The user may override the limits
        let required_limits = override_limits.clone().unwrap_or_else(|| Limits {
            // Fix iOS simulator
            max_inter_stage_shader_variables: 15,
            ..Limits::default()
        });

        // Create the device and the queue
        let descripter = wgpu::DeviceDescriptor {
            label: None,
            required_features,
            required_limits,
            memory_hints: MemoryHints::MemoryUsage,
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
        };
        let (device, queue) = adapter.request_device(&descripter).await?;

        // Create the device handle and store in the pool
        Ok(DeviceHandle {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Creates a new surface for the specified window and dimensions.
    pub async fn create_surface<'w>(
        &mut self,
        window: impl Into<SurfaceTarget<'w>>,
        surface_config: SurfaceRendererConfiguration,
        intermediate_texture_config: Option<TextureConfiguration>,
    ) -> Result<SurfaceRenderer<'w>, WgpuContextError> {
        // Create a surface from the window handle
        let surface = self.instance.create_surface(window.into())?;
        SurfaceRenderer::new(
            surface,
            surface_config,
            intermediate_texture_config,
            self.clone(),
        )
    }
}

/// Simple render context that maintains wgpu state for rendering the pipeline.
pub struct WGPUContext {
    /// A WGPU `Instance`. This only needs to be created once per application.
    pub instance: Instance,
    /// A pool of already-created devices so that we can avoid recreating devices
    /// when we already have a suitable one available
    pub device_pool: Vec<DeviceHandle>,

    // Config
    extra_features: Option<Features>,
    override_limits: Option<Limits>,
    /// True when `instance` was limited to caller-chosen backends, so a default one is a
    /// possible fallback.
    narrowed: bool,
}

impl Default for WGPUContext {
    fn default() -> Self {
        Self::new()
    }
}

impl WGPUContext {
    pub fn new() -> Self {
        Self::with_features_and_limits(None, None)
    }

    pub fn with_features_and_limits(
        extra_features: Option<Features>,
        override_limits: Option<Limits>,
    ) -> Self {
        Self::with_options(WGPUContextOptions {
            extra_features,
            override_limits,
            backends: None,
        })
    }

    /// Creates a context from `options`. `WGPU_BACKEND`, when set, overrides
    /// `options.backends`.
    pub fn with_options(options: WGPUContextOptions) -> Self {
        let from_env = Backends::from_env();
        let narrowed = from_env.is_none() && options.backends.is_some();
        let backends = from_env.or(options.backends).unwrap_or_default();
        Self {
            instance: Self::new_instance(backends),
            device_pool: Vec::new(),
            extra_features: options.extra_features,
            override_limits: options.override_limits,
            narrowed,
        }
    }

    fn new_instance(backends: Backends) -> Instance {
        Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::from_build_config().with_env(),
            backend_options: wgpu::BackendOptions::from_env_or_default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),

            // TODO: support passing display handle
            // Needed for opengl/webgl
            display: None,
        })
    }

    /// An instance over the default backends (`WGPU_BACKEND` still applies). This is the
    /// fallback when a narrowed instance finds no adapter: pass it to
    /// [`replace_instance`](Self::replace_instance).
    pub fn default_instance() -> Instance {
        Self::new_instance(Backends::from_env().unwrap_or_default())
    }

    /// Whether the instance was limited to backends the caller chose, so that
    /// [`default_instance`](Self::default_instance) could find adapters it cannot.
    pub fn is_narrowed(&self) -> bool {
        self.narrowed
    }

    /// Swaps in `instance` and drops the pooled devices, which belong to the old one. The
    /// context is no longer narrowed afterwards.
    pub fn replace_instance(&mut self, instance: Instance) {
        self.instance = instance;
        self.device_pool.clear();
        self.narrowed = false;
    }

    pub fn extra_features(&self) -> Option<Features> {
        self.extra_features
    }

    pub fn override_limits(&self) -> Option<Limits> {
        self.override_limits.clone()
    }

    /// Creates a new surface for the specified window and dimensions.
    pub fn create_surface<'w>(
        &self,
        window: impl Into<SurfaceTarget<'w>>,
    ) -> Result<Surface<'w>, WgpuContextError> {
        Ok(self.instance.create_surface(window.into())?)
    }

    /// Creates a new surface for the specified window and dimensions.
    pub async fn create_surface_renderer<'w>(
        &mut self,
        window: impl Into<SurfaceTarget<'w>>,
        surface_config: SurfaceRendererConfiguration,
        intermediate_texture_config: Option<TextureConfiguration>,
    ) -> Result<SurfaceRenderer<'w>, WgpuContextError> {
        // A window handle cannot be cloned here, so a narrowed instance is checked for any
        // adapter before the surface is made, and replaced by the default one if it has none.
        if self.narrowed {
            let probe = self
                .instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await;
            if probe.is_err() {
                self.replace_instance(Self::default_instance());
            }
        }

        // Create a surface from the window handle
        let surface = self.create_surface(window.into())?;

        // Find or create a suitable device for rendering to the surface
        let dev_id = self
            .find_or_create_device(Some(&surface))
            .await
            .or(Err(WgpuContextError::NoCompatibleDevice))?;
        let device_handle = self.device_pool[dev_id].clone();

        SurfaceRenderer::new(
            surface,
            surface_config,
            intermediate_texture_config,
            device_handle,
        )
    }

    /// Creates a new `BufferRenderer` for the specified dimensions.
    pub async fn create_buffer_renderer(
        &mut self,
        config: BufferRendererConfig,
    ) -> Result<BufferRenderer, WgpuContextError> {
        // Find or create a suitable device for rendering to the surface
        let dev_id = self
            .find_or_create_device(None)
            .await
            .or(Err(WgpuContextError::NoCompatibleDevice))?;
        let device_handle = self.device_pool[dev_id].clone();

        Ok(BufferRenderer::new(config, device_handle, dev_id))
    }

    /// Finds or creates a compatible device handle id.
    pub async fn find_or_create_device(
        &mut self,
        compatible_surface: Option<&Surface<'_>>,
    ) -> Result<usize, WgpuContextError> {
        match self.find_existing_device_idx(compatible_surface) {
            Some(device_id) => Ok(device_id),
            None => self.create_device(compatible_surface).await,
        }
    }

    /// Finds or creates a compatible device handle id.
    pub fn find_compatible_device_handle(
        &mut self,
        compatible_surface: Option<&Surface<'_>>,
    ) -> Option<DeviceHandle> {
        self.find_existing_device_idx(compatible_surface)
            .map(|idx| self.device_pool[idx].clone())
    }

    /// Finds  a compatible device handle id.
    fn find_existing_device_idx(&self, compatible_surface: Option<&Surface<'_>>) -> Option<usize> {
        match compatible_surface {
            Some(s) => self
                .device_pool
                .iter()
                .enumerate()
                .find(|(_, d)| d.adapter.is_surface_supported(s))
                .map(|(i, _)| i),
            None => (!self.device_pool.is_empty()).then_some(0),
        }
    }

    pub fn create_device_handle(
        &self,
        compatible_surface: Option<&Surface<'_>>,
    ) -> impl Future<Output = Result<DeviceHandle, WgpuContextError>> {
        DeviceHandle::new_from_compatible_surface(
            self.instance.clone(),
            compatible_surface,
            self.extra_features,
            self.override_limits.clone(),
        )
    }

    /// Creates a compatible device handle id.
    async fn create_device(
        &mut self,
        compatible_surface: Option<&Surface<'_>>,
    ) -> Result<usize, WgpuContextError> {
        let device_handle = self.create_device_handle(compatible_surface).await?;
        self.device_pool.push(device_handle);
        Ok(self.device_pool.len() - 1)
    }
}
