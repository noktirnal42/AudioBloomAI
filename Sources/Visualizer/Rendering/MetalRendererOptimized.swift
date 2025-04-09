// MetalRendererOptimized.swift
// Performance-optimized Metal renderer for AudioBloomAI visualizations
//

import Foundation
import Metal
import MetalKit
import QuartzCore
import Combine
import os.log

/// High-performance Metal renderer with advanced optimization features
@available(macOS 15.0, *)
public class MetalRendererOptimized: MetalRenderer {
    // MARK: - Performance Configuration
    
    /// Performance configuration options
    public struct PerformanceConfig {
        /// Target frame rate (0 for unlimited)
        var targetFrameRate: Int = 60
        
        /// Enable dynamic quality scaling
        var enableDynamicQuality: Bool = true
        
        /// Minimum acceptable frame rate before quality reduction
        var minAcceptableFrameRate: Int = 45
        
        /// Maximum buffer pool size
        var maxBufferPoolSize: Int = 32
        
        /// Enable compute preprocessing
        var enableComputePreprocessing: Bool = true
        
        /// Enable detailed performance monitoring
        var enablePerformanceMonitoring: Bool = true
        
        /// Default high-quality configuration
        static var highQuality: PerformanceConfig {
            var config = PerformanceConfig()
            config.targetFrameRate = 120
            config.minAcceptableFrameRate = 100
            return config
        }
        
        /// Default balanced configuration
        static var balanced: PerformanceConfig {
            return PerformanceConfig()
        }
        
        /// Default power-saving configuration
        static var powerSaving: PerformanceConfig {
            var config = PerformanceConfig()
            config.targetFrameRate = 30
            config.minAcceptableFrameRate = 25
            return config
        }
    }
    
    // MARK: - Performance Monitoring
    
    /// Performance metrics for monitoring
    private struct PerformanceMetrics {
        var frameTime: Double = 0
        var cpuTime: Double = 0
        var gpuTime: Double = 0
        var frameCount: Int = 0
        var drawCallCount: Int = 0
        var bufferMemoryUsage: Int = 0
        var textureMemoryUsage: Int = 0
        var frameTimeHistory: [Double] = []
        var qualityLevel: Float = 1.0
        
        /// Reset metrics for a new collection period
        mutating func reset() {
            frameTime = 0
            cpuTime = 0
            gpuTime = 0
            frameCount = 0
            drawCallCount = 0
            frameTimeHistory.removeAll()
        }
    }
    
    // MARK: - Buffer Pool Management
    
    /// Buffer pool entry for recycling Metal buffers
    private struct BufferPoolEntry {
        let buffer: MTLBuffer
        let size: Int
        let lastUsed: CFTimeInterval
        let label: String
        var inUse: Bool = false
    }
    
    // MARK: - Properties
    
    /// Current performance configuration
    private var performanceConfig: PerformanceConfig
    
    /// Performance metrics
    private var metrics = PerformanceMetrics()
    
    /// Buffer pools for recycling common buffer types
    private var vertexBufferPool: [BufferPoolEntry] = []
    private var uniformBufferPool: [BufferPoolEntry] = []
    private var computeBufferPool: [BufferPoolEntry] = []
    
    /// Frame timing control
    private var lastFrameTimestamp: CFTimeInterval = 0
    private var targetFrameInterval: CFTimeInterval = 1.0 / 60.0
    
    /// State caching
    private var lastUniformValues: AudioUniforms?
    private var stateCache: [String: Any] = [:]
    
    /// Compute pipeline state for audio preprocessing
    private var computePipelineState: MTLComputePipelineState?
    
    /// High-precision logger
    private let logger = Logger(subsystem: "com.audiobloom.visualizer", category: "MetalRenderer")
    
    /// Performance monitoring publisher
    private let performanceSubject = PassthroughSubject<[String: Any], Never>()
    
    /// Internal GPU counters
    private var gpuCounterBuffer: MTLBuffer?
    private var gpuCounterSemaphore = DispatchSemaphore(value: 0)
    
    /// Quality management
    private var currentQualityLevel: Float = 1.0
    private var qualityAdjustmentTimestamp: CFTimeInterval = 0
    
    // MARK: - Initialization
    
    /// Initialize the optimized Metal renderer
    /// - Parameter config: Performance configuration
    public init(config: PerformanceConfig = .balanced) throws {
        // Store performance configuration
        self.performanceConfig = config
        
        // Set frame rate target
        if config.targetFrameRate > 0 {
            targetFrameInterval = 1.0 / Double(config.targetFrameRate)
        } else {
            targetFrameInterval = 0 // No limit
        }
        
        // Initialize the base renderer
        try super.init()
        
        // Setup additional optimization components
        setupOptimizations()
        
        logger.debug("Initialized optimized Metal renderer with \(config.targetFrameRate) FPS target")
    }
    
    /// Setup additional optimizations beyond the base renderer
    private func setupOptimizations() {
        // Initialize buffer pools
        initializeBufferPools()
        
        // Setup compute pipeline if enabled
        if performanceConfig.enableComputePreprocessing {
            setupComputePipeline()
        }
        
        // Setup GPU performance counters
        setupPerformanceCounters()
        
        // Initialize quality settings
        currentQualityLevel = 1.0
    }
    
    /// Initialize buffer pools for recycling
    private func initializeBufferPools() {
        // Pre-allocate some common buffer sizes for vertex data
        if let device = super.device {
            // Create a few common vertex buffers
            let vertexBufferSizes = [1024, 4096, 16384]
            for size in vertexBufferSizes {
                if let buffer = device.makeBuffer(length: size, options: .storageModeShared) {
                    buffer.label = "PooledVertexBuffer_\(size)"
                    vertexBufferPool.append(
                        BufferPoolEntry(
                            buffer: buffer,
                            size: size,
                            lastUsed: CACurrentMediaTime(),
                            label: "VertexBuffer_\(size)"
                        )
                    )
                }
            }
            
            // Create uniform buffers for triple-buffering
            for i in 0..<3 {
                let uniformSize = MemoryLayout<AudioUniforms>.stride
                if let buffer = device.makeBuffer(length: uniformSize, options: .storageModeShared) {
                    buffer.label = "PooledUniformBuffer_\(i)"
                    uniformBufferPool.append(
                        BufferPoolEntry(
                            buffer: buffer,
                            size: uniformSize,
                            lastUsed: CACurrentMediaTime(),
                            label: "UniformBuffer_\(i)"
                        )
                    )
                }
            }
        }
        
        logger.debug("Initialized buffer pools with \(vertexBufferPool.count) vertex buffers and \(uniformBufferPool.count) uniform buffers")
    }
    
    /// Setup compute pipeline for audio data preprocessing
    private func setupComputePipeline() {
        guard let device = super.device else { return }
        
        do {
            // Load the default library
            let library = try device.makeDefaultLibrary(bundle: Bundle.module)
            
            // Get the compute function for audio preprocessing
            if let computeFunction = library.makeFunction(name: "preprocess_audio_data") {
                // Create compute pipeline state
                computePipelineState = try device.makeComputePipelineState(function: computeFunction)
                logger.debug("Created compute pipeline for audio preprocessing")
            } else {
                logger.warning("Failed to find preprocess_audio_data compute function")
            }
        } catch {
            logger.error("Failed to create compute pipeline: \(error.localizedDescription)")
        }
    }
    
    /// Setup GPU performance counters
    private func setupPerformanceCounters() {
        guard let device = super.device,
              performanceConfig.enablePerformanceMonitoring else { return }
        
        // Create a buffer for GPU counters
        gpuCounterBuffer = device.makeBuffer(length: 256, options: .storageModeShared)
        gpuCounterBuffer?.label = "GPUPerformanceCounters"
        
        logger.debug("Setup GPU performance counters")
    }
    
    // MARK: - Buffer Management
    
    /// Get a buffer from the pool, or create a new one if needed
    /// - Parameters:
    ///   - size: Buffer size in bytes
    ///   - options: Buffer options
    ///   - label: Descriptive label for the buffer
    /// - Returns: Metal buffer
    private func getBufferFromPool(size: Int, options: MTLResourceOptions, label: String) -> MTLBuffer? {
        guard let device = super.device else { return nil }
        
        // Look for an existing buffer in the pool
        for i in 0..<vertexBufferPool.count {
            let entry = vertexBufferPool[i]
            if !entry.inUse && entry.size >= size {
                // Found a suitable buffer
                vertexBufferPool[i].inUse = true
                vertexBufferPool[i].buffer.label = label
                return entry.buffer
            }
        }
        
        // No suitable buffer found, create a new one if pool isn't full
        if vertexBufferPool.count < performanceConfig.maxBufferPoolSize {
            // Round up to next power of 2 for better reuse
            let paddedSize = nextPowerOfTwo(size)
            if let newBuffer = device.makeBuffer(length: paddedSize, options: options) {
                newBuffer.label = label
                
                // Add to pool
                vertexBufferPool.append(
                    BufferPoolEntry(
                        buffer: newBuffer,
                        size: paddedSize,
                        lastUsed: CACurrentMediaTime(),
                        label: label,
                        inUse: true
                    )
                )
                
                return newBuffer
            }
        }
        
        // Fallback: create a temporary buffer (not pooled)
        let tempBuffer = device.makeBuffer(length: size, options: options)
        tempBuffer?.label = "\(label)_temp"
        return tempBuffer
    }
    
    /// Return a buffer to the pool for reuse
    /// - Parameter buffer: The buffer to return
    private func returnBufferToPool(_ buffer: MTLBuffer) {
        // Find the buffer in the pool and mark as not in use
        for i in 0..<vertexBufferPool.count {
            if vertexBufferPool[i].buffer === buffer {
                vertexBufferPool[i].inUse = false
                return
            }
        }
        // If not found, it was a temporary buffer - no action needed
    }
    
    /// Calculate next power of two for efficient buffer sizing
    private func nextPowerOfTwo(_ value: Int) -> Int {
        var power = 1
        while power < value {
            power *= 2
        }
        return power
    }
    
    /// Clean up unused buffers that haven't been used for a while
    private func cleanupBufferPool() {
        let currentTime = CACurrentMediaTime()
        let timeout: CFTimeInterval = 5.0 // 5 seconds timeout
        
        // Remove buffers that haven't been used recently
        vertexBufferPool.removeAll { entry in
            return !entry.inUse && (currentTime - entry.lastUsed > timeout)
        }
    }
    
    // MARK: - Audio Data Preprocessing
    
    /// Preprocess audio data using compute shader for improved performance
    /// - Parameter audioData: Raw audio data
    /// - Returns: Preprocessed audio data
    private func preprocessAudioData(_ audioData: [Float]) -> [Float] {
        guard let device = super.device,
              let computePipelineState = computePipelineState,
              performanceConfig.enableComputePreprocessing else {
            return audioData
        }
        
        // Create input and output buffers
        let count = audioData.count
        guard let inputBuffer = device.makeBuffer(bytes: audioData, length: count * MemoryLayout<Float>.size, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            return audioData
        }
        
        // Create a command buffer
        guard let commandQueue = super.commandQueue,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return audioData
        }
        
        // Configure the compute encoder
        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        computeEncoder.setBytes([Int32(count)], length: MemoryLayout<Int32>.size, index: 2)
        
        // Calculate optimal thread counts
        let threadsPerThreadgroup = min(computePipelineState.maxTotalThreadsPerThreadgroup, 32)
        let threadgroupCount = (count + threadsPerThreadgroup - 1) / threadsPerThreadgroup
        
        // Dispatch the compute work
        computeEncoder.dispatchThreadgroups(MTLSize(width: threadgroupCount, height: 1, depth: 1),
                                           threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1))
        
        // End encoding and commit
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read back the processed data
        var processedData = [Float](repeating: 0, count: count)
        memcpy(&processedData, outputBuffer.contents(), count * MemoryLayout<Float>.size)
        
        return processedData
    }
    
    // MARK: - State Management
    
    /// Check if uniform updates are needed by comparing with previous values
    /// - Parameter uniforms: New uniform values
    /// - Returns: True if update is needed
    private func needsUniformUpdate(_ uniforms: AudioUniforms) -> Bool {
        // If no previous values, definitely need update
        guard let lastUniforms = lastUniformValues else {
            return true
        }
        
        // Compare key properties to determine if update is needed
        if uniforms.visualizationMode != lastUniforms.visualizationMode ||
           uniforms.transitionProgress != lastUniforms.transitionProgress ||
           uniforms.previousMode != lastUniforms.previousMode ||
           uniforms.time != lastUniforms.time ||
           uniforms.bassLevel != lastUniforms.bassLevel ||
           uniforms.midLevel != lastUniforms.midLevel ||
           uniforms.trebleLevel != lastUniforms.trebleLevel ||
           uniforms.beatDetected != lastUniforms.beatDetected {
            return true
        }
        
        // For audio data, check if there are significant changes
        let checkFrequency = 8 // Check every 8th value
        let threshold: Float = 0.05 // Threshold for change detection
        for i in stride(from: 0, to: min(1024, uniforms.audioData.count), by: checkFrequency) {
            if abs(uniforms.audioData[i] - lastUniforms.audioData[i]) > threshold {
                return true
            }
        }
        
        // No significant changes detected
        return false
    }
    
    /// Update state cache with new values
    /// - Parameter uniforms: New uniform values
    private func updateStateCache(_ uniforms: AudioUniforms) {
        lastUniformValues = uniforms
        
        // Cache other frequently accessed state values
        stateCache["visualizationMode"] = uniforms.visualizationMode
        stateCache["themeIndex"] = uniforms.themeIndex
        stateCache["transitionProgress"] = uniforms.transitionProgress
    }
    
    // MARK: - Performance Monitoring
    
    /// Get performance metrics publisher
    /// - Returns: Publisher that emits performance metrics
    public func getPerformanceMetricsPublisher() -> AnyPublisher<[String: Any], Never> {
        return performanceSubject.eraseToAnyPublisher()
    }
    
    /// Update performance metrics with current frame data
    /// - Parameters:
    ///   - frameTime: Time taken to render the frame
    ///   - gpuTime: GPU time (if available)
    private func updatePerformanceMetrics(frameTime: Double, gpuTime: Double? = nil) {
        guard performanceConfig.enablePerformanceMonitoring else { return }
        
        // Update metrics
        metrics.frameCount += 1
        metrics.frameTime += frameTime
        if let gpuTime = gpuTime {
            metrics.gpuTime += gpuTime
        }
        
        // Add to frame time history (for variance calculation)
        metrics.frameTimeHistory.append(frameTime)
        if metrics.frameTimeHistory.count > 60 {
            metrics.frameTimeHistory.removeFirst()
        }
        
        // Update buffer memory usage
        var totalBufferMemory = 0
        for entry in vertexBufferPool {
            totalBufferMemory += entry.size
        }
        metrics.bufferMemoryUsage = totalBufferMemory
        
        // Publish metrics approximately once per second
        let publishInterval: Double = 1.0
        let currentTime = CACurrentMediaTime()
        if currentTime - lastFrameTimestamp >= publishInterval && metrics.frameCount > 0 {
            // Calculate averages
            let avgFrameTime = metrics.frameTime / Double(metrics.frameCount)
            let avgFrameRate = 1.0 / avgFrameTime
            let avgGPUTime = metrics.gpuTime / Double(metrics.frameCount)
            
            // Calculate frame time variability (jitter)
            var frameTimeVariance = 0.0
            if metrics.frameTimeHistory.count > 1 {
                let meanFrameTime = metrics.frameTimeHistory.reduce(0.0, +) / Double(metrics.frameTimeHistory.count)
                frameTimeVariance = metrics.frameTimeHistory.reduce(0.0) { sum, time in
                    let diff = time - meanFrameTime
                    return sum + diff * diff
                } / Double(metrics.frameTimeHistory.count)
            }
            
            // Create metrics dictionary
            let performanceData: [String: Any] = [
                "frameRate": avgFrameRate,
                "frameTime": avgFrameTime * 1000.0, // Convert to ms
                "gpuTime": avgGPUTime * 1000.0, // Convert to ms
                "frameCount": metrics.frameCount,
                "drawCallCount": metrics.drawCallCount,
                "bufferMemoryUsage": metrics.bufferMemoryUsage,
                "qualityLevel": currentQualityLevel,
                "frameTimeJitter": sqrt(frameTimeVariance) * 1000.0 // Convert to ms
            ]
            
            // Publish metrics
            performanceSubject.send(performanceData)
            
            // Reset metrics for next interval
            metrics.reset()
            lastFrameTimestamp = currentTime
            
            // Log performance snapshot
            logger.debug("FPS: \(String(format: "%.1f", avgFrameRate)), Frame time: \(String(format: "%.2f", avgFrameTime * 1000.0))ms, Quality: \(currentQualityLevel)")
        }
    }
    
    /// Evaluate performance and adjust quality if needed
    private func evaluatePerformance() {
        guard performanceConfig.enableDynamicQuality else { return }
        
        // Only adjust quality periodically
        let currentTime = CACurrentMediaTime()
        let qualityAdjustInterval: Double = 2.0 // Adjust every 2 seconds
        
        if currentTime - qualityAdjustmentTimestamp < qualityAdjustInterval {
            return
        }
        
        // Calculate recent average frame rate
        if metrics.frameTimeHistory.count >= 30 {
            let avgFrameTime = metrics.frameTimeHistory.reduce(0.0, +) / Double(metrics.frameTimeHistory.count)
            let avgFrameRate = 1.0 / avgFrameTime
            
            // Check if frame rate is below target
            if avgFrameRate < Double(performanceConfig.minAcceptableFrameRate) && currentQualityLevel > 0.25 {
                // Reduce quality level
                currentQualityLevel = max(0.25, currentQualityLevel - 0.1)
                logger.debug("Reducing quality level to \(currentQualityLevel) due to low frame rate (\(String(format: "%.1f", avgFrameRate)) FPS)")
            } 
            // If frame rate is well above target, try increasing quality
            else if avgFrameRate > Double(performanceConfig.minAcceptableFrameRate) * 1.2 && currentQualityLevel < 1.0 {
                // Increase quality level gradually
                currentQualityLevel = min(1.0, currentQualityLevel + 0.05)
                logger.debug("Increasing quality level to \(currentQualityLevel) - frame rate is good (\(String(format: "%.1f", avgFrameRate)) FPS)")
            }
        }
        
        qualityAdjustmentTimestamp = currentTime
    }
    
    // MARK: - Frame Rate Control
    
    /// Check if a new frame should be rendered based on target frame rate
    /// - Returns: True if a new frame should be rendered
    private func shouldRenderNewFrame() -> Bool {
        if targetFrameInterval <= 0 {
            return true // No frame rate limit
        }
        
        let currentTime = CACurrentMediaTime()
        let timeSinceLastFrame = currentTime - lastFrameTimestamp
        
        // Skip this frame if we're rendering too fast
        if timeSinceLastFrame < targetFrameInterval {
            return false
        }
        
        return true
    }
    
    // MARK: - Overridden Methods
    
    /// Update audio data with optimized preprocessing
    /// - Parameter data: Raw audio frequency data
    override public func updateAudioData(_ data: [Float]) {
        // Use compute preprocessing if enabled
        if performanceConfig.enableComputePreprocessing {
            let processedData = preprocessAudioData(data)
            super.updateAudioData(processedData)
        } else {
            super.updateAudioData(data)
        }
    }
    
    /// Optimized render implementation with performance monitoring
    /// - Parameters:
    ///   - drawable: Metal drawable to render to
    ///   - renderPassDescriptor: Render pass descriptor
    override public func render(to drawable: CAMetalDrawable, with renderPassDescriptor: MTLRenderPassDescriptor) {
        // Apply frame rate limiting
        if !shouldRenderNewFrame() {
            return
        }
        
        // Start frame timing
        let frameStartTime = CACurrentMediaTime()
        
        // Apply dynamic quality if enabled
        evaluatePerformance()
        
        // Let base class handle the actual rendering
        super.render(to: drawable, with: renderPassDescriptor)
        
        // Update performance metrics
        let frameEndTime = CACurrentMediaTime()
        let frameTime = frameEndTime - frameStartTime
        updatePerformanceMetrics(frameTime: frameTime)
        
        // Periodically clean up resources
        if metrics.frameCount % 300 == 0 {
            cleanupBufferPool()
        }
    }
    
    /// Apply quality settings to uniform updates
    override func updateUniformBuffer() {
        // Apply quality reduction if needed
        if currentQualityLevel < 1.0 {
            // Adjust particle count based on quality level
            let particleScale = max(0.2, currentQualityLevel)
            particleCount = Float(50.0 * particleScale)
            
            // Simplify computations for complex visualization modes
            if currentVisualizationMode == 2 || currentVisualizationMode == 3 {
                // Particle and Neural modes are more complex, adjust them more
                spectrumSmoothing = Float(0.5 * (1.0 - currentQualityLevel) + 0.2)
            }
        }
        
        // Use the optimized update with state caching
        guard needsUniformUpdate(uniforms) else {
            // Skip update if no significant changes
            return
        }
        
        // Call the parent implementation
        super.updateUniformBuffer()
        
        // Update state cache after updating uniforms
        updateStateCache(uniforms)
    }
    
    // MARK: - Quality Control
    
    /// Set the quality level directly
    /// - Parameter level: Quality level (0.25-1.0)
    public func setQualityLevel(_ level: Float) {
        currentQualityLevel = min(1.0, max(0.25, level))
        logger.debug("Quality level manually set to \(currentQualityLevel)")
    }
    
    /// Get the current quality level
    /// - Returns: Current quality level (0.25-1.0)
    public func getQualityLevel() -> Float {
        return currentQualityLevel
    }
    
    /// Update the performance configuration
    /// - Parameter config: New performance configuration
    public func updatePerformanceConfig(_ config: PerformanceConfig) {
        performanceConfig = config
        
        // Update frame rate target
        if config.targetFrameRate > 0 {
            targetFrameInterval = 1.0 / Double(config.targetFrameRate)
        } else {
            targetFrameInterval = 0 // No limit
        }
        
        logger.debug("Updated performance configuration - target FPS: \(config.targetFrameRate)")
    }
    
    // MARK: - Resource Management
    
    /// Perform comprehensive cleanup of resources
    public func cleanupResources() {
        // Release all buffers in the pool
        vertexBufferPool.removeAll()
        uniformBufferPool.removeAll()
        computeBufferPool.removeAll()
        
        // Release GPU counters
        gpuCounterBuffer = nil
        
        // Reset compute pipeline state
        computePipelineState = nil
        
        // Reset state cache
        stateCache.removeAll()
        lastUniformValues = nil
        
        // Reset metrics
        metrics.reset()
        
        logger.debug("Performed comprehensive resource cleanup")
    }
    
    /// Deinitializer for proper cleanup
    deinit {
        cleanupResources()
        logger.debug("MetalRendererOptimized deinitialized")
    }
}

// MARK: - AudioShader Compute Extension

/// This compute shader function would be implemented in the corresponding metal file
/// (shown here as a reference for the compute preprocessing functionality)
/*
kernel void preprocess_audio_data(
    const device float *inputData [[buffer(0)]],
    device float *outputData [[buffer(1)]],
    constant int32
