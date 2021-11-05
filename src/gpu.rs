use std::borrow::Cow;
use std::convert::TryInto;

use wgpu::util::DeviceExt;

#[derive(Clone, Debug)]
pub enum Error {
    WGSLProgramSyntax(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Error::WGSLProgramSyntax(src) => write!(f, "WGSL Program Syntax Error: {}", src),
        }
    }
}

impl std::error::Error for Error {}

/// Construct a gpu program taking self as the input.
pub trait IntoGpu {
    type Gpu;
    fn into_gpu(self) -> Self::Gpu;
    fn with_gpu(self, gpu: Gpu) -> Self::Gpu;
}

impl<'d, T: bytemuck::Pod + Primitive> IntoGpu for &'d [T] {
    type Gpu = GpuData<Slice<T>>;

    fn with_gpu(self, gpu: Gpu) -> Self::Gpu {
        // Allocate space on the GPU

        // Allocate the input buffer used to update slice values.
        let input_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Buffer"),
                contents: bytemuck::cast_slice(self),
                usage: wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
            });

        // Allocate the buffer used for GPU processing.
        let storage_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size::<T>(self.len()),
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_SRC
                | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        GpuData {
            storage: Slice {
                input_buffer,
                storage_buffer,
                len: self.len(),
                phantom: std::marker::PhantomData,
            },
            gpu,
        }
    }

    fn into_gpu(self) -> Self::Gpu {
        self.with_gpu(Gpu::new())
    }
}

/// A wrapper for a `wgpu` GPU device and command queue.
pub struct Gpu {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    /// Request device and initialize the command queue.
    fn new() -> Self {
        futures::executor::block_on(Gpu::new_async())
    }

    /// Request device and initialize the command queue asynchronously.
    async fn new_async() -> Self {
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();
        Gpu {
            adapter,
            device,
            queue,
        }
    }
}

/// Data stored on the GPU.
pub struct GpuData<S> {
    storage: S,
    gpu: Gpu,
}

impl<T: Primitive> GpuData<Slice<T>> {
    #[inline]
    pub fn map<P: AsRef<str>>(self, program: P) -> Program<Slice<T>> {
        Program::new(self).map(program)
    }
}

//// TODO: In addition to this way of compiling shaders add a derive macro to compile shaders at compile time.
//fn compile_shader(source: &str) -> shaderc::CompilationArtifact {
//    let mut compiler = shaderc::Compiler::new().unwrap();
//    let binary_result = compiler
//        .compile_into_spirv(
//            source,
//            shaderc::ShaderKind::Compute,
//            "flatk.comp",
//            "main",
//            None,
//        )
//        .unwrap();
//
//    wgpu::util::WordAligned(binary_result).0
//}

/// Scalar types supported by GPU buffers.
pub trait Primitive: Send + Sync + Copy {
    /// Corresponding WGSL type name.
    const WGSL: &'static str;

    /// Size of the type in bytes on GPU.
    const SIZE: usize;

    /// Construct an instance of this type from a set of bytes.
    fn from_ne_byte_slice(bytes: &[u8]) -> Self;
    /// Cast the given mutable byte slice to a mutable reference to `Self`.
    ///
    /// # Safety
    ///
    /// The given byte slice must be a valid representation of `Self`.
    unsafe fn cast_from_ne_byte_slice_mut(bytes: &mut [u8]) -> &mut Self;
}

impl Primitive for bool {
    const WGSL: &'static str = "bool";
    const SIZE: usize = 1;
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), Self::SIZE);
        debug_assert!(bytes[0] == 0 || bytes[0] == 1);
        bytes[0] != 0
    }
    #[inline]
    unsafe fn cast_from_ne_byte_slice_mut(bytes: &mut [u8]) -> &mut Self {
        debug_assert_eq!(bytes.len(), Self::SIZE);
        debug_assert!(bytes[0] == 0 || bytes[0] == 1);
        &mut *(bytes as *mut [u8] as *mut Self)
    }
}
impl Primitive for i32 {
    const WGSL: &'static str = "i32";
    const SIZE: usize = 4;
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), Self::SIZE);
        Self::from_ne_bytes(bytes.try_into().unwrap())
    }
    #[inline]
    unsafe fn cast_from_ne_byte_slice_mut(bytes: &mut [u8]) -> &mut Self {
        debug_assert_eq!(bytes.len(), Self::SIZE);
        &mut *(bytes as *mut [u8] as *mut Self)
    }
}
impl Primitive for u32 {
    const WGSL: &'static str = "u32";
    const SIZE: usize = 4;
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), Self::SIZE);
        Self::from_ne_bytes(bytes.try_into().unwrap())
    }
    #[inline]
    unsafe fn cast_from_ne_byte_slice_mut(bytes: &mut [u8]) -> &mut Self {
        debug_assert_eq!(bytes.len(), Self::SIZE);
        &mut *(bytes as *mut [u8] as *mut Self)
    }
}
impl Primitive for f32 {
    const WGSL: &'static str = "f32";
    const SIZE: usize = 4;
    #[inline]
    fn from_ne_byte_slice(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), Self::SIZE);
        Self::from_ne_bytes(bytes.try_into().unwrap())
    }
    #[inline]
    unsafe fn cast_from_ne_byte_slice_mut(bytes: &mut [u8]) -> &mut Self {
        debug_assert_eq!(bytes.len(), Self::SIZE);
        &mut *(bytes as *mut [u8] as *mut Self)
    }
}

fn buffer_size<T: Primitive>(len: usize) -> wgpu::BufferAddress {
    (len * T::SIZE) as wgpu::BufferAddress
}

pub trait GpuStorage {
    type Host: ?Sized;
    /// Update the underlying values with the values from the given host storage.
    fn update(&self, host: &Self::Host, gpu: &Gpu);
    /// Parallel version of `update`.
    #[cfg(feature = "rayon")]
    fn update_par(&self, host: &Self::Host, gpu: &Gpu);
    fn len(&self) -> usize;
    /// Determines the structural and type equality between two GPU data structures.
    fn is_same(&self, other: &Self) -> bool;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A slice represented on the GPU.
pub struct Slice<T> {
    /// Buffer used to store and process data.
    storage_buffer: wgpu::Buffer,
    input_buffer: wgpu::Buffer,
    /// Number of primitives contained in the buffer.
    len: usize,
    phantom: std::marker::PhantomData<T>,
}

impl<T: Primitive> Slice<T> {
    async fn update_async(&self, host: &<Self as GpuStorage>::Host, gpu: &Gpu) {
        let buffer_slice = self.input_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Write);

        gpu.device.poll(wgpu::Maintain::Wait);

        if let Ok(()) = buffer_future.await {
            let mut data = buffer_slice.get_mapped_range_mut();
            data.chunks_exact_mut(T::SIZE)
                .map(|b| unsafe { T::cast_from_ne_byte_slice_mut(b) })
                .zip(host.iter())
                .for_each(|(buf_val, &host_val)| *buf_val = host_val);

            drop(data);
            self.input_buffer.unmap();
        } else {
            panic!("Failed to update storage buffer");
        }
    }
    #[cfg(feature = "rayon")]
    async fn update_par_async(&self, host: &<Self as GpuStorage>::Host, gpu: &Gpu) {
        use rayon::prelude::*;
        let buffer_slice = self.input_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Write);

        gpu.device.poll(wgpu::Maintain::Wait);

        if let Ok(()) = buffer_future.await {
            let mut data = buffer_slice.get_mapped_range_mut();
            data.par_chunks_exact_mut(T::SIZE)
                .map(|b| unsafe { T::cast_from_ne_byte_slice_mut(b) })
                .zip(host.par_iter())
                .for_each(|(buf_val, &host_val)| *buf_val = host_val);

            drop(data);
            self.input_buffer.unmap();
        } else {
            panic!("Failed to update storage buffer");
        }
    }
}

impl<T: Primitive> GpuStorage for Slice<T> {
    type Host = [T];
    fn update(&self, host: &Self::Host, gpu: &Gpu) {
        futures::executor::block_on(self.update_async(host, gpu));
    }

    #[cfg(feature = "rayon")]
    fn update_par(&self, host: &Self::Host, gpu: &Gpu) {
        futures::executor::block_on(self.update_par_async(host, gpu));
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn is_same(&self, other: &Slice<T>) -> bool {
        self.len == other.len
    }
}

/// Output GPU buffer representing a result.
pub struct OutputBuffer<T> {
    /// Buffer used to store and process data.
    buf: wgpu::Buffer,
    /// Number of elements of type `T` contained in the buffer.
    len: usize,
    phantom: std::marker::PhantomData<T>,
}

impl<T: Primitive> OutputBuffer<T> {
    fn new(len: usize, gpu: &Gpu) -> Self {
        // Create the output buffer on the GPU.
        OutputBuffer {
            buf: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: buffer_size::<T>(len),
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            }),
            len,
            phantom: std::marker::PhantomData,
        }
    }
}

/// A compute shader program in WGSL with an associated data structure.
pub struct Program<S> {
    data: GpuData<S>,
    workgroup_size: u32,
    program: String,
    main: String,
    /// The status of this program
    ///
    /// This is used for early exists when it is determined that the program is not well formed.
    status: Result<(), Error>,
}

impl<S> Program<S> {
    pub fn new(data: GpuData<S>) -> Self {
        Program {
            data,
            workgroup_size: 64,
            program: String::new(),
            main: String::new(),
            status: Ok(()),
        }
    }
}

impl<T: Primitive> Program<Slice<T>> {
    pub(crate) fn shader_header() -> String {
        format!(
            //"
            //#version 450
            //layout(local_size_x = {}) in;

            //layout(set = 0, binding = 0) buffer Data {{
            //    uint[] slice;
            //}};", workgroup_size
            "
            [[block]]
            struct Data {{
                data: [[stride(4)]] array<{}>;
            }};
            [[group(0), binding(0)]]
            var<storage> v_indices: [[access(read_write)]] Data;
            ",
            T::WGSL
        )
    }

    pub(crate) fn map_program(name: &str) -> String {
        format!(
            //"{{
            //    uint index = gl_GlobalInvocationID.x;
            //    slice[index] = {name}(slice[index]);
            //}}",
            "{{
                v_indices.data[global_id.x] = {name}(v_indices.data[global_id.x]);
            }}",
            name = name
        )
    }

    fn assemble(program: &str, main: &str, workgroup_size: u32) -> String {
        format!(
            "
            {header}
            {program}
            [[stage(compute), workgroup_size({wg_size})]]
            fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
                 {body}
            }}
            ",
            header = Self::shader_header(),
            program = program,
            wg_size = workgroup_size,
            body = main
        )
    }

    pub fn with_workgroup_size(mut self, workgroup_size: u32) -> Self {
        self.workgroup_size = workgroup_size;
        self
    }

    pub fn compile(self) -> Result<CompiledProgram<Slice<T>, OutputBuffer<T>>, Error> {
        let Program {
            data: GpuData { storage, gpu },
            workgroup_size,
            program,
            main,
            status,
        } = self;

        // Exit pre-emptively if the program is already known to be ill-formed.
        if let Some(err) = status.err() {
            return Err(err);
        }

        // Assemble the complete shader program.
        let shader_program = Self::assemble(program.as_str(), main.as_str(), workgroup_size);

        //std::eprintln!("{}", &shader_program);

        let mut flags = wgpu::ShaderFlags::VALIDATION;
        match gpu.adapter.get_info().backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Gl => {
                flags |= wgpu::ShaderFlags::EXPERIMENTAL_TRANSLATION;
            }
            _ => {}
        }

        let shader_module = gpu
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_program)),
                flags,
            });

        // Instantiate the pipeline.
        let compute_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader_module,
                    entry_point: "main",
                });

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage.storage_buffer.as_entire_binding(),
            }],
        });

        //let pipeline_layout = gpu
        //    .device
        //    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //        bind_group_layouts: &[&bind_group_layout],
        //        push_constant_ranges: &[],
        //    });

        let output_buffer = OutputBuffer::new(storage.len, &gpu);

        Ok(CompiledProgram {
            input: storage,
            output: output_buffer,
            gpu,
            workgroup_size,
            compute_pipeline,
            bind_group,
        })
    }

    /// Append a map pass.
    pub fn map<P: AsRef<str>>(mut self, program: P) -> Program<Slice<T>> {
        let program = program.as_ref();
        if !program.starts_with("fn ") {
            return Program {
                status: Err(Error::WGSLProgramSyntax(
                    "WGSL Program must always start with the string \"fn \"".to_string(),
                )),
                ..self
            };
        }

        // Parse the function signature to extract the name.

        let (_, rest) = program.split_at(3); // Skip "fn " (above check makes sure this does not panic)
        if let Some(program_name) = rest.split("(").next().map(|name| name.trim()) {
            self.program.push_str(program);
            self.main.push_str(&Self::map_program(program_name));
            self
        } else {
            Program {
                status: Err(Error::WGSLProgramSyntax(
                    "Failed to find the function parameters surrounded by parentheses".to_string(),
                )),
                ..self
            }
        }
    }
}

/// A slice on the gpu along with a compiled shader program parameterized by the input data type
/// `I` and output data `O`.
pub struct CompiledProgram<I, O> {
    input: I,
    output: O,
    gpu: Gpu,
    workgroup_size: u32,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl<I: GpuStorage, O> CompiledProgram<I, O> {
    /// Specify alternative data to use instead of the previously associated data.
    /// This function returns the original data stored in this `CompiledProgram`.
    ///
    /// # Panics
    ///
    /// This function will panic if the given other data structure is not equivalent to the one
    /// already associated with the compiled program.
    pub fn replace_storage(&mut self, other: I) -> I {
        assert!(self.input.is_same(&other));
        std::mem::replace(&mut self.input, other)
    }

    /// Update the values stored in the underlying GPU storage.
    pub fn update_data(&self, other: &I::Host) {
        self.input.update(other, &self.gpu);
    }
}

impl<T: Primitive> CompiledProgram<Slice<T>, OutputBuffer<T>> {
    /// Execute the program.
    pub fn run(&self) -> &Self {
        let CompiledProgram {
            input:
                Slice {
                    input_buffer,
                    storage_buffer,
                    len,
                    ..
                },
            gpu: Gpu { device, queue, .. },
            workgroup_size,
            compute_pipeline,
            bind_group,
            ..
        } = self;

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&input_buffer, 0, &storage_buffer, 0, buffer_size::<T>(*len));
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(compute_pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch(
                ((*len as f32) / (*workgroup_size as f32)).ceil() as u32,
                1,
                1,
            );
        }

        queue.submit(Some(encoder.finish()));
        &self
    }

    /// Collect the computed data
    pub fn collect(&self) -> Vec<T> {
        futures::executor::block_on(self.collect_async())
    }

    pub async fn collect_async(&self) -> Vec<T> {
        let CompiledProgram {
            input:
                Slice {
                    storage_buffer,
                    len,
                    ..
                },
            output,
            gpu,
            ..
        } = self;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert_eq!(output.len, *len);
        encoder.copy_buffer_to_buffer(&storage_buffer, 0, &output.buf, 0, buffer_size::<T>(*len));

        gpu.queue.submit(Some(encoder.finish()));

        let buffer_slice = output.buf.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

        gpu.device.poll(wgpu::Maintain::Wait);

        if let Ok(()) = buffer_future.await {
            let data = buffer_slice.get_mapped_range();
            let result = data
                .chunks_exact(T::SIZE)
                .map(|b| T::from_ne_byte_slice(b))
                .collect();

            drop(data);
            output.buf.unmap();

            result
        } else {
            panic!("failed to run compute on gpu");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add1() {
        let add1: &str = stringify! {
            fn program(x: i32) -> i32 {
                return x + 1i;
            }
        };

        let numbers = (1..10).collect::<Vec<_>>();

        // Prepare the slice on the gpu
        let gpu_slice = numbers.as_slice().into_gpu();

        // Compile the map program to the gpu
        let map_gpu_slice = gpu_slice.map(add1).compile().unwrap();

        // Execute the gpu program and collect the results.
        let result = map_gpu_slice.run().collect();
        assert_eq!(vec![2, 3, 4, 5, 6, 7, 8, 9, 10], result);

        // Run again to make sure that the input hasn't changed
        let result = map_gpu_slice.run().collect();
        assert_eq!(vec![2, 3, 4, 5, 6, 7, 8, 9, 10], result);

        // Update the input and run again
        let rev_numbers: Vec<_> = numbers.iter().cloned().rev().collect();
        map_gpu_slice.update_data(rev_numbers.as_slice());
        let result = map_gpu_slice.run().collect();
        assert_eq!(vec![10, 9, 8, 7, 6, 5, 4, 3, 2], result);
    }
    #[test]
    fn collatz() {
        let collatz: &str = stringify! {
            fn program(n_base: u32) -> u32 {
                var n: u32 = n_base;
                var i: u32 = 0u;
                loop {
                    if (n <= 1u) {
                        break;
                    }
                    if (n % 2u == 0u) {
                        n = n / 2u;
                    } else {
                        // Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
                        if (n >= 1431655765u) {   // 0x55555555u
                            return 4294967295u;   // 0xffffffffu
                        }

                        n = 3u * n + 1u;
                    }
                    i = i + 1u;
                }
                return i;
            }
        };

        let numbers = (1u32..10).collect::<Vec<_>>();

        // Prepare the slice on the gpu
        let gpu_slice = numbers.as_slice().into_gpu();

        // Compile the map program to the gpu
        let map_gpu_slice = gpu_slice.map(collatz).compile().unwrap();

        // Execute the gpu program and collect the results.
        let result = map_gpu_slice.run().collect();
        assert_eq!(vec![0, 1, 7, 2, 5, 8, 16, 3, 19], result);

        // Run again to make sure that the input hasn't changed
        let result = map_gpu_slice.run().collect();
        assert_eq!(vec![0, 1, 7, 2, 5, 8, 16, 3, 19], result);

        // Update the input and run again
        let rev_numbers: Vec<_> = numbers.iter().cloned().rev().collect();
        map_gpu_slice.update_data(rev_numbers.as_slice());
        let result = map_gpu_slice.run().collect();
        assert_eq!(vec![19, 3, 16, 8, 5, 2, 7, 1, 0], result);
    }
}
