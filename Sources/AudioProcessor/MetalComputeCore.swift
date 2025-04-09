import Foundation
import Metal
import MetalKit
import Accelerate
import Logging
import Combine
import AudioBloomCore

/// Errors that can occur in Metal compute operations
@available(macOS 15.0, *)
public enum MetalComputeError: Error {
    /// Metal device could not be initialized
    case deviceInitFailed
    /// Command queue creation failed
    case commandQueueCreationFailed
    /// Default library creation failed
    case libraryCreationFailed(Error)
    /// Function not found in library
    case functionNotFound(String)
    /// Pipeline state creation failed
    case pipelineCreationFailed(Error)
    /// Buffer allocation failed
    case bufferAllocationFailed
    /// Invalid kernel parameters
    case invalidKernelParameters
    /// Thread contention
    case threadContention
    /// Execution failure
    case executionFailed(Error)
    /// Resource not found
    case resourceNotFound(String)
}

/// Types of compute kernels available
@available(macOS 15.0, *)
public enum ComputeKernelType: String, CaseIterable {
    /// Fast Fourier Transform
    case fft = "fft_forward"
    /// Inverse Fast Fourier Transform
    case ifft = "fft_inverse"
    /// Spectrum analysis
    case spectrumAnalysis = "spectrum_analysis"
    /// Time domain filtering
    case timeFilter = "time_domain_filter"
    /// Frequency domain filtering
    case frequencyFilter = "frequency_domain_filter"
    /// Audio normalization
    case normalize = "normalize_audio"
    /// Custom kernel (requires name)
    case custom
}

/// Completion handler for asynchronous computations
@available(macOS 15.0, *)
public typealias ComputeCompletionHandler = (Result<Void, Error>) -> Void

/// Manages Metal compute operations for audio processing
@available(macOS 15.0, *)
public class MetalComputeCore {
    // MARK: - Properties
    
    /// Logger instance
    private let logger = Logger(label: "com.audiobloom.metal-compute")
    
    /// Metal device for GPU acceleration
    private let device: MTLDevice
    
    /// Metal command queue for submitting compute commands
    private let commandQueue: MTLCommandQueue
    
    /// Default compute library
    private let defaultLibrary: MTLLibrary
    
    /// Cache of compute pipeline states by kernel name
    private var pipelineStateCache: [String: MTLComputePipelineState] = [:]
    
    /// Buffer management for recycling GPU buffers
    private var bufferPool: [Int: [MTLBuffer]] = [:]
    
    /// Counter for generating unique buffer IDs
    private var nextBufferID: UInt64 = 1
    
    /// Dictionary mapping buffer IDs to Metal buffers
    private var activeBuffers: [UInt64: MTLBuffer] = [:]
    
    /// Serial queue for buffer operations
    private let bufferQueue = DispatchQueue(label: "com.audiobloom.metal-buffer", qos: .userInteractive)
    
    /// Concurrent queue for compute operations
    private let computeQueue = DispatchQueue(label: "com.audiobloom.metal-compute", qos: .userInteractive, attributes: .concurrent)
    
    /// Semaphore for limiting concurrent operations
    private let concurrencySemaphore: DispatchSemaphore
    
    /// Performance monitor
    private let performanceMonitor = PerformanceMonitor()
    
    /// Maximum concurrent operations
    private let maxConcurrentOperations: Int
    
    // MARK: - Initialization
    
    /// Initialize the Metal compute core with the default device
    /// - Parameter maxConcurrentOperations: Maximum number of concurrent operations (default: 3)
    public init(maxConcurrentOperations: Int = 3) throws {
        self.maxConcurrentOperations = maxConcurrentOperations
        self.concurrencySemaphore = DispatchSemaphore(value: maxConcurrentOperations)
        
        // Initialize Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            logger.error("Failed to create Metal device")
            throw MetalComputeError.deviceInitFailed
        }
        self.device = device
        logger.info("Using Metal device: \(device.name)")
        
        // Create command queue
        guard let commandQueue = device.makeCommandQueue() else {
            logger.error("Failed to create Metal command queue")
            throw MetalComputeError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
        
        // Load default library
        do {
            self.defaultLibrary = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            logger.error("Failed to create default Metal library: \(error)")
            throw MetalComputeError.libraryCreationFailed(error)
        }
        
        // Pre-compile common kernels
        try precompileCommonKernels()
        
        logger.info("Metal compute core initialized successfully")
    }
    
    /// Pre-compile commonly used kernels for better performance
    private func precompileCommonKernels() throws {
        // Precompile built-in kernels for faster launch
        for kernelType in ComputeKernelType.allCases where kernelType != .custom {
            do {
                _ = try getPipelineState(for: kernelType.rawValue)
                logger.debug("Pre-compiled kernel: \(kernelType.rawValue)")
            } catch {
                logger.warning("Failed to pre-compile kernel \(kernelType.rawValue): \(error)")
                // Continue with other kernels even if one fails
            }
        }
    }
    
    // MARK: - Kernel Management
    
    /// Get or create a compute pipeline state for the given kernel function
    /// - Parameter kernelName: Name of the kernel function
    /// - Returns: Compiled compute pipeline state
    private func getPipelineState(for kernelName: String) throws -> MTLComputePipelineState {
        // Check if pipeline state is already cached
        if let cachedState = pipelineStateCache[kernelName] {
            return cachedState
        }
        
        // Create new pipeline state
        guard let kernelFunction = defaultLibrary.makeFunction(name: kernelName) else {
            logger.error("Failed to find kernel function: \(kernelName)")
            throw MetalComputeError.functionNotFound(kernelName)
        }
        
        do {
            let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
            // Cache the pipeline state
            pipelineStateCache[kernelName] = pipelineState
            return pipelineState
        } catch {
            logger.error("Failed to create pipeline state for \(kernelName): \(error)")
            throw MetalComputeError.pipelineCreationFailed(error)
        }
    }
    
    // MARK: - Buffer Management
    
    /// Allocate a new Metal buffer or reuse one from the pool
    /// - Parameters:
    ///   - length: Length of the buffer in bytes
    ///   - options: Buffer storage options
    /// - Returns: Buffer ID and the allocated buffer
    public func allocateBuffer(length: Int, options: MTLResourceOptions = .storageModeShared) throws -> (id: UInt64, buffer: MTLBuffer) {
        return try bufferQueue.sync {
            // Check if there's a reusable buffer in the pool
            if var availableBuffers = bufferPool[length], !availableBuffers.isEmpty {
                let buffer = availableBuffers.removeLast()
                bufferPool[length] = availableBuffers
                
                // Generate a new ID
                let bufferID = nextBufferID
                nextBufferID += 1
                
                // Register the buffer
                activeBuffers[bufferID] = buffer
                
                return (bufferID, buffer)
            }
            
            // Create a new buffer
            guard let buffer = device.makeBuffer(length: length, options: options) else {
                logger.error("Failed to allocate Metal buffer of size \(length)")
                throw MetalComputeError.bufferAllocationFailed
            }
            
            // Generate a new ID
            let bufferID = nextBufferID
            nextBufferID += 1
            
            // Register the buffer
            activeBuffers[bufferID] = buffer
            
            return (bufferID, buffer)
        }
    }
    
    /// Release a buffer back to the pool
    /// - Parameter id: ID of the buffer to release
    public func releaseBuffer(id: UInt64) {
        bufferQueue.sync {
            guard let buffer = activeBuffers[id] else {
                logger.warning("Attempted to release non-existent buffer: \(id)")
                return
            }
            
            // Remove from active buffers
            activeBuffers.removeValue(forKey: id)
            
            // Add to buffer pool for recycling
            let length = buffer.length
            var lengthBuffers = bufferPool[length] ?? []
            lengthBuffers.append(buffer)
            bufferPool[length] = lengthBuffers
            
            logger.debug("Released buffer \(id) to pool (size: \(length))")
        }
    }
    
    /// Update buffer contents from CPU memory
    /// - Parameters:
    ///   - id: Buffer ID
    ///   - data: Pointer to the source data
    ///   - length: Length of the data in bytes
    ///   - offset: Offset into the buffer
    public func updateBuffer(id: UInt64, from data: UnsafeRawPointer, length: Int, offset: Int = 0) throws {
        guard let buffer = bufferQueue.sync(execute: { activeBuffers[id] }) else {
            logger.error("Attempted to update non-existent buffer: \(id)")
            throw MetalComputeError.resourceNotFound("Buffer \(id)")
        }
        
        guard offset + length <= buffer.length else {
            logger.error("Buffer update exceeds buffer size: \(offset + length) > \(buffer.length)")
            throw MetalComputeError.invalidKernelParameters
        }
        
        // Different approach based on storage mode
        if buffer.storageMode == .shared || buffer.storageMode == .managed {
            // For shared or managed buffers, we can write directly
            let bufferPointer = buffer.contents().advanced(by: offset)
            memcpy(bufferPointer, data, length)
            
            // If managed, we need to synchronize
            if buffer.storageMode == .managed {
                buffer.didModifyRange(offset..<(offset + length))
            }
        } else {
            // For private storage, use a blit encoder
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                logger.error("Failed to create blit command encoder")
                throw MetalComputeError.executionFailed(MetalComputeError.commandQueueCreationFailed)
            }
            
            // Create a temporary buffer with the data
            guard let tempBuffer = device.makeBuffer(bytes: data, length: length, options: .storageModeShared) else {
                logger.error("Failed to create temporary buffer")
                throw MetalComputeError.bufferAllocationFailed
            }
            
            // Copy from temp buffer to destination buffer
            blitEncoder.copy(from: tempBuffer, sourceOffset: 0, to: buffer, destinationOffset: offset, size: length)
            blitEncoder.endEncoding()
            
            // Commit and wait
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }
    
    /// Read data from a Metal buffer into CPU memory
    /// - Parameters:
    ///   - id: Buffer ID
    ///   - destination: Pointer to the destination memory
    ///   - length: Length of the data to read in bytes
    ///   - offset: Offset into the buffer
    public func readBuffer(id: UInt64, into destination: UnsafeMutableRawPointer, length: Int, offset: Int = 0) throws {
        guard let buffer = bufferQueue.sync(execute: { activeBuffers[id] }) else {
            logger.error("Attempted to read non-existent buffer: \(id)")
            throw MetalComputeError.resourceNotFound("Buffer \(id)")
        }
        
        guard offset + length <= buffer.length else {
            logger.error("Buffer read exceeds buffer size: \(offset + length) > \(buffer.length)")
            throw MetalComputeError.invalidKernelParameters
        }
        
        // Different approach based on storage mode
        if buffer.storageMode == .shared || buffer.storageMode == .managed {
            // For shared or managed buffers, we can read directly
            let bufferPointer = buffer.contents().advanced(by: offset)
            memcpy(destination, bufferPointer, length)
        } else {
            // For private storage, use a blit encoder to copy to a temporary buffer
            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                logger.error("Failed to create command buffer")
                throw MetalComputeError.executionFailed(MetalComputeError.commandQueueCreationFailed)
            }
            
            // Create a temporary buffer to hold the data
            guard let tempBuffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                logger.error("Failed to create temporary buffer")
                throw MetalComputeError.bufferAllocationFailed
            }
            
            // Copy from source buffer to temporary buffer
            guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                logger.error("Failed to create blit command encoder")
                throw MetalComputeError.executionFailed(MetalComputeError.commandQueueCreationFailed)
            }
            
            blitEncoder.copy(from: buffer, sourceOffset: offset, to: tempBuffer, destinationOffset: 0, size: length)
            blitEncoder.endEncoding()
            
            // Execute and wait for completion
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // Copy from temporary buffer to destination
            memcpy(destination, tempBuffer.contents(), length)
        }
    }
    
    // MARK: - Compute Operations
    
    /// Execute a compute kernel asynchronously
    /// - Parameters:
    ///   - kernelType: Type of compute kernel to execute
    ///   - customKernelName: Name of custom kernel (if kernelType is .custom)
    ///   - gridSize: Grid size for the compute operation
    ///   - buffers: Array of buffer IDs to use as inputs/outputs
    ///   - parameters: Additional parameters as a byte buffer
    ///   - completion: Completion handler called when operation finishes
    public func executeKernel(
        kernelType: ComputeKernelType,
        customKernelName: String? = nil,
        gridSize: MTLSize,
        buffers: [UInt64],
        parameters: Data? = nil,
        completion: @escaping ComputeCompletionHandler
    ) {
        // Use a kernel name based on type
        let kernelName: String
        if kernelType == .custom, let name = customKernelName {
            kernelName = name
        } else {
            kernelName = kernelType.rawValue
        }
        
        // Submit the task to the compute queue
        computeQueue.async { [weak self] in
            guard let self = self else {
                completion(.failure(MetalComputeError.resourceNotFound("MetalComputeCore instance")))
                return
            }
            
            // Wait for a slot to become available (limits concurrent GPU operations)
            if self.concurrencySemaphore.wait(timeout: .now() + .seconds(5)) == .timedOut {
                self.logger.error("Timed out waiting for compute resources")
                completion(.failure(MetalComputeError.threadContention))
                return
            }
            
            // Use defer to ensure semaphore is always signaled
            defer {
                self.concurrencySemaphore.signal()
            }
            
            // Start performance tracking
            let operationId = self.performanceMonitor.beginOperation(name: kernelName)
            
            do {
                // Get the pipeline state for this kernel
                let pipelineState = try self.getPipelineState(for: kernelName)
                
                // Grab all the Metal buffers from their IDs
                var metalBuffers: [MTLBuffer] = []
                
                // Lock for reading buffers
                self.bufferQueue.sync {
                    for bufferId in buffers {
                        guard let buffer = self.activeBuffers[bufferId] else {
                            // If any buffer is missing, we fail the entire operation
                            self.logger.error("Buffer \(bufferId) not found for kernel \(kernelName)")
                            throw MetalComputeError.resourceNotFound("Buffer \(bufferId)")
                        }
                        metalBuffers.append(buffer)
                    }
                }
                
                // Create a command buffer
                guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
                    throw MetalComputeError.commandQueueCreationFailed
                }
                
                // Create a compute command encoder
                guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw MetalComputeError.executionFailed(MetalComputeError.commandQueueCreationFailed)
                }
                
                // Set the compute pipeline state
                computeEncoder.setComputePipelineState(pipelineState)
                
                // Set the buffers
                for (index, buffer) in metalBuffers.enumerated() {
                    computeEncoder.setBuffer(buffer, offset: 0, index: index)
                }
                
                // Set additional parameters if provided
                if let parameters = parameters {
                    parameters.withUnsafeBytes { rawBufferPointer in
                        guard let baseAddress = rawBufferPointer.baseAddress else { return }
                        computeEncoder.setBytes(baseAddress, length: parameters.count, index: metalBuffers.count)
                    }
                }
                
                // Calculate threadgroup size based on device capabilities
                let threadgroupSize = MTLSize(
                    width: pipelineState.threadExecutionWidth,
                    height: 1,
                    depth: 1
                )
                
                // Calculate number of threadgroups
                let threadgroupCount = MTLSize(
                    width: (gridSize.width + threadgroupSize.width - 1) / threadgroupSize.width,
                    height: (gridSize.height + threadgroupSize.height - 1) / threadgroupSize.height,
                    depth: (gridSize.depth + threadgroupSize.depth - 1) / threadgroupSize.depth
                )
                
                // Dispatch the compute work
                computeEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
                
                // End encoding
                computeEncoder.endEncoding()
                
                // Add completion handler to measure performance and call user callback
                commandBuffer.addCompletedHandler { [weak self] commandBuffer in
                    guard let self = self else { return }
                    
                    // End performance tracking
                    self.performanceMonitor.endOperation(id: operationId)
                    
                    // Check for errors
                    if let error = commandBuffer.error {
                        self.logger.error("Compute operation \(kernelName) failed: \(error)")
                        completion(.failure(MetalComputeError.executionFailed(error)))
                    } else {
                        // Log performance metrics
                        let metrics = self.performanceMonitor.getMetrics(id: operationId)
                        self.logger.debug("""
                            Compute operation \(kernelName) completed:
                              - Duration: \(metrics.duration * 1000) ms
                              - GPU Time: \(commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) ms
                        """)
                        
                        completion(.success(()))
                    }
                }
                
                // Submit the command buffer
                commandBuffer.commit()
                
            } catch {
                // Stop performance tracking on error
                self.performanceMonitor.endOperation(id: operationId, error: error)
                
                // Log error
                self.logger.error("Failed to execute kernel \(kernelName): \(error)")
                
                // Notify caller
                completion(.failure(error))
            }
        }
    }
    
    /// Execute a compute kernel synchronously
    /// - Parameters:
    ///   - kernelType: Type of compute kernel to execute
    ///   - customKernelName: Name of custom kernel (if kernelType is .custom)
    ///   - gridSize: Grid size for the compute operation
    ///   - buffers: Array of buffer IDs to use as inputs/outputs
    ///   - parameters: Additional parameters as a byte buffer
    public func executeKernelSync(
        kernelType: ComputeKernelType,
        customKernelName: String? = nil,
        gridSize: MTLSize,
        buffers: [UInt64],
        parameters: Data? = nil
    ) throws {
        // Create a semaphore to wait for completion
        let semaphore = DispatchSemaphore(value: 0)
        var operationError: Error?
        
        // Execute asynchronously but wait for completion
        executeKernel(
            kernelType: kernelType,
            customKernelName: customKernelName,
            gridSize: gridSize,
            buffers: buffers,
            parameters: parameters
        ) { result in
            switch result {
            case .success:
                break
            case .failure(let error):
                operationError = error
            }
            semaphore.signal()
        }
        
        // Wait for completion
        _ = semaphore.wait(timeout: .distantFuture)
        
        // Propagate any error
        if let error = operationError {
            throw error
        }
    }
    
    // MARK: - Helper Methods for Audio Processing
    
    /// Perform Fast Fourier Transform on audio data
    /// - Parameters:
    ///   - inputBufferId: Input buffer containing audio samples
    ///   - outputBufferId: Output buffer to receive FFT results
    ///   - sampleCount: Number of audio samples to process
    ///   - inverse: Whether to perform inverse FFT
    ///   - completion: Completion handler
    public func performFFT(
        inputBufferId: UInt64,
        outputBufferId: UInt64,
        sampleCount: Int,
        inverse: Bool = false,
        completion: @escaping ComputeCompletionHandler
    ) {
        // Ensure sample count is a power of 2
        let actualSampleCount = isPowerOfTwo(sampleCount) ? sampleCount : nextPowerOfTwo(sampleCount)
        
        // Create parameters buffer
        var params = FFTParameters(
            sampleCount: UInt32(actualSampleCount),
            inputOffset: 0,
            outputOffset: 0,
            inverse: inverse ? 1 : 0
        )
        
        let paramsData = Data(bytes: &params, count: MemoryLayout<FFTParameters>.size)
        
        // Execute appropriate kernel
        let kernelType: ComputeKernelType = inverse ? .ifft : .fft
        
        executeKernel(
            kernelType: kernelType,
            gridSize: MTLSize(width: actualSampleCount / 2, height: 1, depth: 1),
            buffers: [inputBufferId, outputBufferId],
            parameters: paramsData,
            completion: completion
        )
    }
    
    /// Apply frequency domain filter to audio data
    /// - Parameters:
    ///   - inputBufferId: Input buffer containing frequency domain data
    ///   - outputBufferId: Output buffer to receive filtered results
    ///   - filterBufferId: Buffer containing filter coefficients
    ///   - sampleCount: Number of samples to process
    ///   - completion: Completion handler
    public func applyFrequencyFilter(
        inputBufferId: UInt64,
        outputBufferId: UInt64,
        filterBufferId: UInt64,
        sampleCount: Int,
        completion: @escaping ComputeCompletionHandler
    ) {
        // Create parameters buffer
        var params = FilterParameters(
            sampleCount: UInt32(sampleCount),
            inputOffset: 0,
            outputOffset: 0,
            filterOffset: 0
        )
        
        let paramsData = Data(bytes: &params, count: MemoryLayout<FilterParameters>.size)
        
        // Execute filtering kernel
        executeKernel(
            kernelType: .frequencyFilter,
            gridSize: MTLSize(width: sampleCount, height: 1, depth: 1),
            buffers: [inputBufferId, outputBufferId, filterBufferId],
            parameters: paramsData,
            completion: completion
        )
    }
    
    /// Apply time domain filter to audio data
    /// - Parameters:
    ///   - inputBufferId: Input buffer containing audio samples
    ///   - outputBufferId: Output buffer to receive filtered results
    ///   - filterBufferId: Buffer containing filter coefficients
    ///   - sampleCount: Number of samples to process
    ///   - filterLength: Length of the filter
    ///   - completion: Completion handler
    public func applyTimeFilter(
        inputBufferId: UInt64,
        outputBufferId: UInt64,
        filterBufferId: UInt64,
        sampleCount: Int,
        filterLength: Int,
        completion: @escaping ComputeCompletionHandler
    ) {
        // Create parameters buffer
        var params = TimeFilterParameters(
            sampleCount: UInt32(sampleCount),
            filterLength: UInt32(filterLength),
            inputOffset: 0,
            outputOffset: 0,
            filterOffset: 0
        )
        
        let paramsData = Data(bytes: &params, count: MemoryLayout<TimeFilterParameters>.size)
        
        // Execute filtering kernel
        executeKernel(
            kernelType: .timeFilter,
            gridSize: MTLSize(width: sampleCount, height: 1, depth: 1),
            buffers: [inputBufferId, outputBufferId, filterBufferId],
            parameters: paramsData,
            completion: completion
        )
    }
    
    /// Analyze audio spectrum to extract frequency band information
    /// - Parameters:
    ///   - spectrumBufferId: Buffer containing FFT spectrum data
    ///   - outputBufferId: Buffer to receive analysis results
    ///   - sampleCount: Number of spectrum samples
    ///   - sampleRate: Audio sample rate in Hz
    ///   - completion: Completion handler
    public func analyzeSpectrum(
        spectrumBufferId: UInt64,
        outputBufferId: UInt64,
        sampleCount: Int,
        sampleRate: Float,
        completion: @escaping ComputeCompletionHandler
    ) {
        // Create parameters buffer
        var params = SpectrumAnalysisParameters(
            sampleCount: UInt32(sampleCount),
            sampleRate: sampleRate,
            bassMinFreq: 20.0,
            bassMaxFreq: 250.0,
            midMinFreq: 250.0,
            midMaxFreq: 4000.0,
            trebleMinFreq: 4000.0,
            trebleMaxFreq: 20000.0
        )
        
        let paramsData = Data(bytes: &params, count: MemoryLayout<SpectrumAnalysisParameters>.size)
        
        // Execute spectrum analysis kernel
        executeKernel(
            kernelType: .spectrumAnalysis,
            gridSize: MTLSize(width: sampleCount, height: 1, depth: 1),
            buffers: [spectrumBufferId, outputBufferId],
            parameters: paramsData,
            completion: completion
        )
    }
    /// Normalize audio data to a target level
    /// - Parameters:
    ///   - inputBufferId: Input buffer containing audio samples
    ///   - outputBufferId: Output buffer to receive normalized results
    ///   - sampleCount: Number of audio samples to process
    ///   - targetLevel: Target normalization level (0.0-1.0)
    ///   - completion: Completion handler
    public func normalizeAudio(
        inputBufferId: UInt64,
        outputBufferId: UInt64,
        sampleCount: Int,
        targetLevel: Float = 0.9,
        completion: @escaping ComputeCompletionHandler
    ) {
        // Create parameters buffer
        var params = NormalizeParameters(
            sampleCount: UInt32(sampleCount),
            targetLevel: targetLevel,
            inputOffset: 0,
            outputOffset: 0
        )
        
        let paramsData = Data(bytes: &params, count: MemoryLayout<NormalizeParameters>.size)
        
        // Execute normalization kernel
        executeKernel(
            kernelType: .normalize,
            gridSize: MTLSize(width: sampleCount, height: 1, depth: 1),
            buffers: [inputBufferId, outputBufferId],
            parameters: paramsData,
            completion: completion
        )
    }
    
    // MARK: - Utility Methods
    
    /// Check if a number is a power of 2
    /// - Parameter number: Number to check
    /// - Returns: Whether the number is a power of 2
    private func isPowerOfTwo(_ number: Int) -> Bool {
        return number > 0 && (number & (number - 1)) == 0
    }
    
    /// Get the next power of 2 greater than or equal to the given number
    /// - Parameter number: Input number
    /// - Returns: Next power of 2
    private func nextPowerOfTwo(_ number: Int) -> Int {
        var n = number
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n += 1
        return n
    }
    
    /// Get buffer information for reporting
    /// - Returns: Buffer statistics
    public func getBufferStats() -> (activeCount: Int, pooledCount: Int, totalBytes: Int) {
        return bufferQueue.sync {
            let activeCount = activeBuffers.count
            
            var pooledCount = 0
            var totalBytes = 0
            
            // Count active buffers
            for buffer in activeBuffers.values {
                totalBytes += buffer.length
            }
            
            // Count pooled buffers
            for (size, buffers) in bufferPool {
                pooledCount += buffers.count
                totalBytes += size * buffers.count
            }
            
            return (activeCount, pooledCount, totalBytes)
        }
    }
    
    /// Clean up unused buffers from the buffer pool
    public func cleanupUnusedBuffers() {
        bufferQueue.sync {
            // Remove all pooled buffers
            let count = bufferPool.reduce(0) { $0 + $1.value.count }
            bufferPool.removeAll(keepingCapacity: true)
            logger.info("Cleaned up \(count) unused buffers from pool")
        }
    }
    
    /// Clean up resources when deinitializing
    deinit {
        bufferQueue.sync {
            // Clean up all buffers
            activeBuffers.removeAll()
            bufferPool.removeAll()
            
            // Clear pipeline state cache
            pipelineStateCache.removeAll()
            
            logger.info("MetalComputeCore deinitialized")
        }
    }
}

// MARK: - Parameter Structures

/// Parameters for FFT operations
@available(macOS 15.0, *)
struct FFTParameters {
    /// Number of samples to process
    let sampleCount: UInt32
    /// Offset in input buffer
    let inputOffset: UInt32
    /// Offset in output buffer
    let outputOffset: UInt32
    /// Whether to perform inverse FFT (1) or forward FFT (0)
    let inverse: UInt32
}

/// Parameters for filter operations
@available(macOS 15.0, *)
struct FilterParameters {
    /// Number of samples to process
    let sampleCount: UInt32
    /// Offset in input buffer
    let inputOffset: UInt32
    /// Offset in output buffer
    let outputOffset: UInt32
    /// Offset in filter buffer
    let filterOffset: UInt32
}

/// Parameters for time domain filter operations
@available(macOS 15.0, *)
struct TimeFilterParameters {
    /// Number of samples to process
    let sampleCount: UInt32
    /// Length of the filter
    let filterLength: UInt32
    /// Offset in input buffer
    let inputOffset: UInt32
    /// Offset in output buffer
    let outputOffset: UInt32
    /// Offset in filter buffer
    let filterOffset: UInt32
}

/// Parameters for spectrum analysis operations
@available(macOS 15.0, *)
struct SpectrumAnalysisParameters {
    /// Number of spectrum samples
    let sampleCount: UInt32
    /// Sample rate in Hz
    let sampleRate: Float
    /// Minimum bass frequency (Hz)
    let bassMinFreq: Float
    /// Maximum bass frequency (Hz)
    let bassMaxFreq: Float
    /// Minimum mid frequency (Hz)
    let midMinFreq: Float
    /// Maximum mid frequency (Hz)
    let midMaxFreq: Float
    /// Minimum treble frequency (Hz)
    let trebleMinFreq: Float
    /// Maximum treble frequency (Hz)
    let trebleMaxFreq: Float
}

/// Parameters for normalization operations
@available(macOS 15.0, *)
struct NormalizeParameters {
    /// Number of samples to process
    let sampleCount: UInt32
    /// Target normalization level (0.0-1.0)
    let targetLevel: Float
    /// Offset in input buffer
    let inputOffset: UInt32
    /// Offset in output buffer
    let outputOffset: UInt32
}

// MARK: - Performance Monitoring

/// Metrics for a compute operation
@available(macOS 15.0, *)
public struct OperationMetrics {
    /// Operation name
    public let name: String
    /// Duration in seconds
    public let duration: TimeInterval
    /// Whether the operation succeeded
    public let succeeded: Bool
    /// Error that occurred (if any)
    public let error: Error?
    /// Start time
    public let startTime: Date
    /// End time
    public let endTime: Date
}

/// Tracks performance of Metal compute operations
@available(macOS 15.0, *)
class PerformanceMonitor {
    /// Operation tracking information
    private struct OperationInfo {
        /// Operation name
        let name: String
        /// Start time
        let startTime: Date
        /// End time (nil if operation is still in progress)
        var endTime: Date?
        /// Error that occurred (if any)
        var error: Error?
    }
    
    /// Lock for thread-safe access
    private let lock = NSRecursiveLock()
    
    /// Counter for generating unique operation IDs
    private var nextOperationID: UInt64 = 1
    
    /// Currently tracked operations
    private var activeOperations: [UInt64: OperationInfo] = [:]
    
    /// Completed operations (limited to recent history)
    private var completedOperations: [UInt64: OperationInfo] = [:]
    
    /// Maximum number of completed operations to track
    private let maxCompletedOperations = 100
    
    /// Begin tracking a new operation
    /// - Parameter name: Operation name
    /// - Returns: Operation ID
    func beginOperation(name: String) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        
        let operationID = nextOperationID
        nextOperationID += 1
        
        let info = OperationInfo(
            name: name,
            startTime: Date(),
            endTime: nil,
            error: nil
        )
        
        activeOperations[operationID] = info
        return operationID
    }
    
    /// End tracking for an operation
    /// - Parameters:
    ///   - id: Operation ID
    ///   - error: Error that occurred (if any)
    func endOperation(id: UInt64, error: Error? = nil) {
        lock.lock()
        defer { lock.unlock() }
        
        guard var info = activeOperations[id] else {
            return
        }
        
        // Update operation info
        info.endTime = Date()
        info.error = error
        
        // Move from active to completed
        activeOperations.removeValue(forKey: id)
        completedOperations[id] = info
        
        // Limit completed operations history
        if completedOperations.count > maxCompletedOperations {
            let sortedKeys = completedOperations.keys.sorted()
            if let oldestKey = sortedKeys.first {
                completedOperations.removeValue(forKey: oldestKey)
            }
        }
    }
    
    /// Get metrics for a completed operation
    /// - Parameter id: Operation ID
    /// - Returns: Operation metrics
    func getMetrics(id: UInt64) -> OperationMetrics {
        lock.lock()
        defer { lock.unlock() }
        
        // Check for completed operation
        if let info = completedOperations[id], let endTime = info.endTime {
            let duration = endTime.timeIntervalSince(info.startTime)
            return OperationMetrics(
                name: info.name,
                duration: duration,
                succeeded: info.error == nil,
                error: info.error,
                startTime: info.startTime,
                endTime: endTime
            )
        }
        
        // Check for active operation
        if let info = activeOperations[id] {
            let duration = Date().timeIntervalSince(info.startTime)
            return OperationMetrics(
                name: info.name,
                duration: duration,
                succeeded: false,
                error: nil,
                startTime: info.startTime,
                endTime: Date()
            )
        }
        
        // Operation not found
        return OperationMetrics(
            name: "unknown",
            duration: 0,
            succeeded: false,
            error: MetalComputeError.resourceNotFound("Operation \(id)"),
            startTime: Date(),
            endTime: Date()
        )
    }
    
    /// Get metrics for all operations in a given time range
    /// - Parameters:
    ///   - startDate: Start date for the range
    ///   - endDate: End date for the range
    /// - Returns: Array of operation metrics
    func getMetricsInRange(startDate: Date, endDate: Date) -> [OperationMetrics] {
        lock.lock()
        defer { lock.unlock() }
        
        var metrics: [OperationMetrics] = []
        
        // Add completed operations in range
        for (id, info) in completedOperations {
            if let endTime = info.endTime,
               info.startTime >= startDate && endTime <= endDate {
                metrics.append(getMetrics(id: id))
            }
        }
        
        // Sort by start time
        metrics.sort { $0.startTime < $1.startTime }
        return metrics
    }
    
    /// Reset the performance monitor, clearing all tracked operations
    func reset() {
        lock.lock()
        defer { lock.unlock() }
        
        activeOperations.removeAll()
        completedOperations.removeAll()
    }
    
    /// Get statistics about operations
    /// - Returns: Tuple with overall statistics
    func getStatistics() -> (totalCompleted: Int, activeCount: Int, averageDuration: TimeInterval, successRate: Double) {
        lock.lock()
        defer { lock.unlock() }
        
        let totalCompleted = completedOperations.count
        let activeCount = activeOperations.count
        
        // Calculate average duration of completed operations
        var totalDuration: TimeInterval = 0
