import Foundation
import MetalKit
import simd

// Visualization modes supported by the renderer
enum VisualizationType: Int {
    case waveform = 0
    case spectrum = 1
    case circular = 2
}

// Wrapper for the uniform buffer data structure defined in Shaders.metal
struct Uniforms {
    var modelMatrix: simd_float4x4
    var projectionMatrix: simd_float4x4
    var time: Float
    var intensity: Float
    var visualizationType: Float
    
    init() {
        modelMatrix = matrix_identity_float4x4
        projectionMatrix = matrix_identity_float4x4
        time = 0.0
        intensity = 0.5
        visualizationType = 0.0
    }
}

// Wrapper for the audio data buffer defined in Shaders.metal
struct AudioDataBufferContent {
    var samples: UnsafeMutablePointer<Float>?
    var fftData: UnsafeMutablePointer<Float>?
    var sampleCount: UInt32
    var fftDataSize: UInt32
    var amplitude: Float
    
    init() {
        samples = nil
        fftData = nil
        sampleCount = 0
        fftDataSize = 0
        amplitude = 1.0
    }
}

// Audio visualization renderer using Metal
class AudioVisualizer: NSObject, MTKViewDelegate {
    // MARK: - Properties
    
    // Metal objects
    private var device: MTLDevice
    private var commandQueue: MTLCommandQueue
    private var library: MTLLibrary
    
    // Pipeline states for different visualization types
    private var waveformPipelineState: MTLRenderPipelineState
    private var spectrumPipelineState: MTLRenderPipelineState
    private var circularPipelineState: MTLRenderPipelineState
    private var defaultPipelineState: MTLRenderPipelineState
    
    // Buffers
    private var vertexBuffer: MTLBuffer?
    private var uniformBuffer: MTLBuffer
    private var audioDataBuffer: MTLBuffer
    
    // Audio data
    private var audioSamples: [Float] = []
    private var fftData: [Float] = []
    private var audioDataBufferContent = AudioDataBufferContent()
    
    // Visualization parameters
    private var uniforms = Uniforms()
    private var visualizationType: VisualizationType = .waveform
    private var startTime: CFTimeInterval
    
    // Configuration
    private let maxSamples = 1024
    private let maxFFTSize = 512
    
    // MARK: - Initialization
    
    init(device: MTLDevice, view: MTKView) throws {
        self.device = device
        self.startTime = CACurrentMediaTime()
        
        // Create command queue
        guard let commandQueue = device.makeCommandQueue() else {
            throw NSError(domain: "AudioVisualizerErrorDomain", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
        }
        self.commandQueue = commandQueue
        
        // Load Metal shader library
        guard let library = device.makeDefaultLibrary() else {
            throw NSError(domain: "AudioVisualizerErrorDomain", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to load Metal library"])
        }
        self.library = library
        
        // Create pipeline states for each visualization type
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        
        // Waveform pipeline
        pipelineDescriptor.vertexFunction = library.makeFunction(name: "waveformVertexShader")
        pipelineDescriptor.fragmentFunction = library.makeFunction(name: "waveformFragmentShader")
        waveformPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        
        // Spectrum pipeline
        pipelineDescriptor.vertexFunction = library.makeFunction(name: "spectrumVertexShader")
        pipelineDescriptor.fragmentFunction = library.makeFunction(name: "spectrumFragmentShader")
        spectrumPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        
        // Circular pipeline
        pipelineDescriptor.vertexFunction = library.makeFunction(name: "circularVertexShader")
        pipelineDescriptor.fragmentFunction = library.makeFunction(name: "circularFragmentShader")
        circularPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        
        // Default fallback pipeline
        pipelineDescriptor.vertexFunction = library.makeFunction(name: "vertexShader")
        pipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentShader")
        defaultPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        
        // Create uniform buffer
        let uniformBufferSize = MemoryLayout<Uniforms>.size
        guard let uniformBuffer = device.makeBuffer(length: uniformBufferSize, options: .storageModeShared) else {
            throw NSError(domain: "AudioVisualizerErrorDomain", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create uniform buffer"])
        }
        self.uniformBuffer = uniformBuffer
        
        // Create audio data buffer
        let audioDataSize = MemoryLayout<AudioDataBufferContent>.size
        guard let audioDataBuffer = device.makeBuffer(length: audioDataSize, options: .storageModeShared) else {
            throw NSError(domain: "AudioVisualizerErrorDomain", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio data buffer"])
        }
        self.audioDataBuffer = audioDataBuffer
        
        // Initialize vertex buffer for basic shape
        createVertexBuffer()
        
        // Initialize audio data
        audioSamples = [Float](repeating: 0, count: maxSamples)
        fftData = [Float](repeating: 0, count: maxFFTSize)
        
        // Initialize default uniform values
        updateProjectionMatrix(viewSize: view.bounds.size)
        
        super.init()
        
        // Set up MTKView delegate
        view.delegate = self
        view.preferredFramesPerSecond = 60
    }
    
    // MARK: - Buffer Management
    
    private func createVertexBuffer() {
        // Simple 2D quad for drawing with point primitives or lines
        let vertexCount = maxSamples
        let vertexSize = MemoryLayout<simd_float3>.size * vertexCount
        
        guard let buffer = device.makeBuffer(length: vertexSize, options: .storageModeShared) else {
            print("Failed to create vertex buffer")
            return
        }
        
        vertexBuffer = buffer
    }
    
    // MARK: - Audio Data Updates
    
    func updateAudioSamples(_ samples: [Float]) {
        // Copy audio samples to our internal buffer with safety bounds check
        let count = min(samples.count, maxSamples)
        for i in 0..<count {
            audioSamples[i] = samples[i]
        }
        
        // Clear remaining samples if any
        if count < maxSamples {
            for i in count..<maxSamples {
                audioSamples[i] = 0
            }
        }
        
        // Update the sample count in our buffer structure
        updateAudioDataBuffer(sampleCount: UInt32(count))
    }
    
    func updateFFTData(_ fftMagnitudes: [Float]) {
        // Copy FFT data to our internal buffer with safety bounds check
        let count = min(fftMagnitudes.count, maxFFTSize)
        for i in 0..<count {
            fftData[i] = fftMagnitudes[i]
        }
        
        // Clear remaining FFT data if any
        if count < maxFFTSize {
            for i in count..<maxFFTSize {
                fftData[i] = 0
            }
        }
        
        // Update the FFT data size in our buffer structure
        updateAudioDataBuffer(fftDataSize: UInt32(count))
    }
    
    private func updateAudioDataBuffer(sampleCount: UInt32? = nil, fftDataSize: UInt32? = nil) {
        // Update buffer content values if provided
        if let sampleCount = sampleCount {
            audioDataBufferContent.sampleCount = sampleCount
        }
        
        if let fftDataSize = fftDataSize {
            audioDataBufferContent.fftDataSize = fftDataSize
        }
        
        // Create Metal buffers for audio data if needed
        if audioDataBufferContent.samples == nil {
            guard let samplesBuffer = device.makeBuffer(bytes: audioSamples, length: audioSamples.count * MemoryLayout<Float>.size, options: .storageModeShared) else {
                print("Failed to create samples buffer")
                return
            }
            audioDataBufferContent.samples = UnsafeMutablePointer<Float>(OpaquePointer(samplesBuffer.contents()))
        } else {
            // Update existing buffer
            memcpy(audioDataBufferContent.samples!, audioSamples, audioSamples.count * MemoryLayout<Float>.size)
        }
        
        if audioDataBufferContent.fftData == nil {
            guard let fftBuffer = device.makeBuffer(bytes: fftData, length: fftData.count * MemoryLayout<Float>.size, options: .storageModeShared) else {
                print("Failed to create FFT buffer")
                return
            }
            audioDataBufferContent.fftData = UnsafeMutablePointer<Float>(OpaquePointer(fftBuffer.contents()))
        } else {
            // Update existing buffer
            memcpy(audioDataBufferContent.fftData!, fftData, fftData.count * MemoryLayout<Float>.size)
        }
        
        // Copy the structure to the Metal buffer
        let bufferPointer = audioDataBuffer.contents().bindMemory(to: AudioDataBufferContent.self, capacity: 1)
        bufferPointer.pointee = audioDataBufferContent
    }
    
    // MARK: - Visualization Configuration
    
    func setVisualizationType(_ type: VisualizationType) {
        visualizationType = type
        uniforms.visualizationType = Float(type.rawValue)
        updateUniforms()
    }
    
    func setAmplitude(_ amplitude: Float) {
        audioDataBufferContent.amplitude = amplitude
        updateAudioDataBuffer()
    }
    
    func setIntensity(_ intensity: Float) {
        uniforms.intensity = intensity
        updateUniforms()
    }
    
    func updateProjectionMatrix(viewSize: CGSize) {
        let aspect = Float(viewSize.width / viewSize.height)
        uniforms.projectionMatrix = matrix_float4x4_ortho(left: -aspect, right: aspect, bottom: -1, top: 1, near: 0.1, far: 100)
        updateUniforms()
    }
    
    private func updateUniforms() {
        // Update time value
        uniforms.time = Float(CACurrentMediaTime() - startTime)
        
        // Copy uniforms to Metal buffer
        let bufferPointer = uniformBuffer.contents().bindMemory(to: Uniforms.self, capacity: 1)
        bufferPointer.pointee = uniforms
    }
    
    // MARK: - MTKViewDelegate
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        updateProjectionMatrix(viewSize: size)
    }
    
    func draw(in view: MTKView) {
        // Update uniform values (especially time)
        updateUniforms()
        
        // Get current drawable and command buffer
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        // Create render command encoder
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        // Set up render pipeline and buffers
        let pipelineState: MTLRenderPipelineState
        switch visualizationType {
        case .waveform:
            pipelineState = waveformPipelineState
            renderEncoder.setRenderPipelineState(pipelineState)
            
            // Set buffers for waveform visualization
            renderEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
            renderEncoder.setVertexBuffer(audioDataBuffer, offset: 0, index: 2)
            renderEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
            
            // Draw points or lines based on audio samples
            renderEncoder.drawPrimitives(type: .lineStrip, vertexStart: 0, vertexCount: Int(audioDataBufferContent.sampleCount))
            
        case .spectrum:
            pipelineState = spectrumPipelineState
            renderEncoder.setRenderPipelineState(pipelineState)
            
            // Set buffers for spectrum visualization
            renderEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
            renderEncoder.setVertexBuffer(audioDataBuffer, offset: 0, index: 2)
            renderEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
            
            // Draw bars for spectrum visualization
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: Int(audioDataBufferContent.fftDataSize) * 2)
            
        case .circular:
            pipelineState = circularPipelineState
            renderEncoder.setRenderPipelineState(pipelineState)
            
            // Set buffers for circular visualization
            renderEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
            renderEncoder.setVertexBuffer(audioDataBuffer, offset: 0, index: 2)
            renderEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
            
            // Draw circle shape
            renderEncoder.drawPrimitives(type: .lineStrip, vertexStart: 0, vertexCount: Int(audioDataBufferContent.sampleCount))
        }
        
        renderEncoder.endEncoding()
        
        // Present drawable and commit command buffer
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    // MARK: - Utility Functions
    
    private func matrix_float4x4_ortho(left: Float, right: Float, bottom: Float, top: Float, near: Float, far: Float) -> simd_float4x4 {
        let ral = right + left
        let rsl = right - left
        let tab = top + bottom
        let tsb = top - bottom
        let fan = far + near
        let fsn = far - near
        
        var m = simd_float4x4(0)
        m[0][0] = 2.0 / rsl
        m[1][1] = 2.0 / tsb
        m[2][2] = -2.0 / fsn
        m[3][0] = -ral / rsl
        m[3][1] = -tab / tsb
        m[3][2] = -fan / fsn
        m[3][3] = 1.0
        
        return m
    }
    
    /// Creates a translation matrix with the specified offsets
    private func matrix_float4x4_translation(x: Float, y: Float, z: Float) -> simd_float4x4 {
        var matrix = matrix_identity_float4x4
        matrix[3][0] = x
        matrix[3][1] = y
        matrix[3][2] = z
        return matrix
    }
    
    /// Creates a scaling matrix with the specified scale factors
    private func matrix_float4x4_scale(x: Float, y: Float, z: Float) -> simd_float4x4 {
        var matrix = matrix_identity_float4x4
        matrix[0][0] = x
        matrix[1][1] = y
        matrix[2][2] = z
        return matrix
    }
    
    /// Creates a rotation matrix around the Z axis
    private func matrix_float4x4_rotation_z(radians: Float) -> simd_float4x4 {
        let cosine = cos(radians)
        let sine = sin(radians)
        
        var matrix = matrix_identity_float4x4
        matrix[0][0] = cosine
        matrix[0][1] = sine
        matrix[1][0] = -sine
        matrix[1][1] = cosine
        
        return matrix
    }
    
    // MARK: - Lifecycle Management
    
    /// Releases resources when the visualizer is no longer needed
    func cleanup() {
        // Release any manually managed resources
        if let samples = audioDataBufferContent.samples {
            // If we were manually allocating memory, we would free it here
            audioDataBufferContent.samples = nil
        }
        
        if let fftData = audioDataBufferContent.fftData {
            // If we were manually allocating memory, we would free it here
            audioDataBufferContent.fftData = nil
        }
    }
    
    deinit {
        cleanup()
    }
    
    // MARK: - Error Handling
    
    /// Updates audio data buffers with error handling
    func safelyUpdateAudioSamples(_ samples: [Float]) {
        do {
            // Check if the device is still valid
            guard device.registryID != 0 else {
                print("Metal device is no longer valid")
                return
            }
            
            updateAudioSamples(samples)
        } catch {
            print("Error updating audio samples: \(error.localizedDescription)")
        }
    }
    
    /// Updates FFT data with error handling
    func safelyUpdateFFTData(_ fftMagnitudes: [Float]) {
        do {
            // Check if the device is still valid
            guard device.registryID != 0 else {
                print("Metal device is no longer valid")
                return
            }
            
            updateFFTData(fftMagnitudes)
        } catch {
            print("Error updating FFT data: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Animation and Effects
    
    /// Animate a transition between visualization types
    func animateTransitionToType(_ type: VisualizationType, duration: TimeInterval = 0.5) {
        // Store the target visualization type
        let targetType = type
        let startTime = CACurrentMediaTime()
        let endTime = startTime + duration
        
        // Set up an animation timer
        Timer.scheduledTimer(withTimeInterval: 1/60, repeats: true) { [weak self] timer in
            guard let self = self else {
                timer.invalidate()
                return
            }
            
            let currentTime = CACurrentMediaTime()
            if currentTime >= endTime {
                // Animation complete
                self.setVisualizationType(targetType)
                timer.invalidate()
                return
            }
            
            // Calculate transition progress (0 to 1)
            let progress = Float((currentTime - startTime) / duration)
            
            // Update visualization parameters based on progress
            // This could be customized to create different transition effects
            self.uniforms.intensity = progress
            self.updateUniforms()
        }
    }
    
    /// Apply an effect to the current visualization
    func applyEffect(amplitude: Float, frequency: Float) {
        // Example of applying a dynamic effect to the visualization
        let scale = 1.0 + amplitude * sin(uniforms.time * frequency) * 0.2
        
        // Apply effect through model matrix transformation
        let scaleMatrix = matrix_float4x4_scale(x: scale, y: scale, z: 1.0)
        let rotationMatrix = matrix_float4x4_rotation_z(radians: uniforms.time * 0.1)
        
        // Combine transformations
        uniforms.modelMatrix = rotationMatrix * scaleMatrix
        updateUniforms()
    }
    
    /// Reset all effects and transformations to default values
    func resetEffects() {
        uniforms.modelMatrix = matrix_identity_float4x4
        uniforms.intensity = 0.5
        updateUniforms()
    }
}
