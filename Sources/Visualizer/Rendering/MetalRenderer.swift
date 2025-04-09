import Foundation
import Metal
import MetalKit
import QuartzCore
import Combine

/// Visualization mode for audio rendering
public enum VisualizationMode: Float {
    case spectrum = 0.0
    case waveform = 1.0
    case particles = 2.0
    case neural = 3.0
}

/// Theme for audio visualization
public enum Theme: Float {
    case classic = 0.0
    case neon = 1.0
    case monochrome = 2.0
    case cosmic = 3.0
    case sunset = 4.0
    case ocean = 5.0
    case forest = 6.0
    case custom = 7.0
}

/// Uniform structure passed to Metal shaders
struct AudioUniforms {
    // Audio data array (raw FFT or waveform data)
    var audioData: [Float] = [Float](repeating: 0, count: 1024)
    
    // Audio Analysis Parameters
    var bassLevel: Float = 0.0
    var midLevel: Float = 0.0
    var trebleLevel: Float = 0.0
    var leftLevel: Float = 0.0
    var rightLevel: Float = 0.0
    
    // Theme colors
    var primaryColor: SIMD4<Float> = SIMD4<Float>(1.0, 0.3, 0.7, 1.0)
    var secondaryColor: SIMD4<Float> = SIMD4<Float>(0.2, 0.8, 1.0, 1.0)
    var backgroundColor: SIMD4<Float> = SIMD4<Float>(0.05, 0.05, 0.1, 1.0)
    var accentColor: SIMD4<Float> = SIMD4<Float>(1.0, 0.8, 0.2, 1.0)
    
    // Animation parameters
    var time: Float = 0.0
    var sensitivity: Float = 0.8
    var motionIntensity: Float = 0.7
    var themeIndex: Float = 0.0
    
    // Visualization settings
    var visualizationMode: Float = 0.0
    var previousMode: Float = 0.0
    var transitionProgress: Float = 0.0
    var colorIntensity: Float = 0.8
    
    // Additional parameters
    var spectrumSmoothing: Float = 0.3
    var particleCount: Float = 50.0
    
    // Neural visualization parameters
    var neuralEnergy: Float = 0.5
    var neuralPleasantness: Float = 0.5
    var neuralComplexity: Float = 0.5
    var beatDetected: Float = 0.0
    
    // Padding to ensure alignment (if needed)
    private var _padding: [Float] = [Float](repeating: 0, count: 8)
}

/// Renderer that uses Metal to visualize audio data
public class MetalRenderer {
    // MARK: - Constants
    
    private enum Constants {
        // Maximum number of audio frames to process
        static let maxAudioFrames = 1024
        
        // Default buffer sizes
        static let initialUniformBufferSize = MemoryLayout<AudioUniforms>.stride
        
        // Resource management
        static let maxInflightBuffers = 3
        
        // Shader function names
        static let vertexFunctionName = "audio_vertex_shader"
        static let fragmentFunctionName = "audio_fragment_shader"
        
        // Default colors
        static let defaultPrimaryColor = SIMD4<Float>(1.0, 0.3, 0.7, 1.0)
        static let defaultSecondaryColor = SIMD4<Float>(0.2, 0.8, 1.0, 1.0)
        static let defaultBackgroundColor = SIMD4<Float>(0.05, 0.05, 0.1, 1.0)
        static let defaultAccentColor = SIMD4<Float>(1.0, 0.8, 0.2, 1.0)
    }
    
    // MARK: - Metal Properties
    
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var renderPipelineState: MTLRenderPipelineState!
    private var vertexBuffer: MTLBuffer!
    private var uniformBuffer: MTLBuffer!
    private var semaphore: DispatchSemaphore
    
    // Performance trackers
    private var frameCounter = 0
    private var lastFrameTimestamp: CFTimeInterval = 0
    
    // MARK: - Render State
    
    private var viewportSize = CGSize(width: 1, height: 1)
    private var startTime: CFTimeInterval = CACurrentMediaTime()
    private var lastUpdateTime: CFTimeInterval = 0
    
    // Memory management
    private var bufferIndex = 0
    private var uniforms = AudioUniforms()
    
    // MARK: - Audio Visualization State
    
    private var audioData = [Float](repeating: 0, count: Constants.maxAudioFrames)
    private var bassLevel: Float = 0
    private var midLevel: Float = 0
    private var trebleLevel: Float = 0
    private var leftLevel: Float = 0
    private var rightLevel: Float = 0
    
    // Animation parameters
    private var sensitivity: Float = 0.8
    private var motionIntensity: Float = 0.7
    private var themeIndex: Float = 0.0  // 0: Classic, 1: Neon, 2: Monochrome, 3: Cosmic
    
    // Visualization modes
    private var currentVisualizationMode: Float = 0.0  // 0: Spectrum, 1: Waveform, 2: Particles, 3: Neural
    private var previousVisualizationMode: Float = 0.0
    private var transitionProgress: Float = 0.0
    private var isInTransition = false
    private var transitionStartTime: CFTimeInterval = 0
    private var transitionDuration: CFTimeInterval = 0.75
    
    // Theme colors
    private var primaryColor = Constants.defaultPrimaryColor
    private var secondaryColor = Constants.defaultSecondaryColor
    private var backgroundColor = Constants.defaultBackgroundColor
    private var accentColor = Constants.defaultAccentColor
    
    // Additional visualization parameters
    private var colorIntensity: Float = 0.8
    private var spectrumSmoothing: Float = 0.3
    private var particleCount: Float = 50
    
    // Neural visualization parameters
    private var neuralEnergy: Float = 0.5
    private var neuralPleasantness: Float = 0.5
    private var neuralComplexity: Float = 0.5
    private var beatDetected: Float = 0.0
    
    // MARK: - Error Handling
    
    /// Errors that can occur in the Metal renderer
    enum RendererError: Error {
        case deviceNotAvailable
        case commandQueueCreationFailed
        case libraryCreationFailed(Error)
        case vertexFunctionNotFound
        case fragmentFunctionNotFound
        case pipelineCreationFailed(Error)
        case bufferAllocationFailed
    }
    
    // MARK: - Initialization
    
    /// Initialize the Metal renderer
    public init() throws {
        semaphore = DispatchSemaphore(value: Constants.maxInflightBuffers)
        
        try setupMetal()
        try setupRenderPipeline()
        try setupBuffers()
        
        // Initialize default audio data
        resetAudioData()
    }
    
    /// Set audio levels for visualization (truncated and duplicated - removed)
    ///   - left: Left channel volume (0.0-1.0)
    ///   - right: Right channel volume (0.0-1.0)
    public func setAudioLevels(bass: Float, mid: Float, treble: Float, left: Float, right: Float) {
        bassLevel = bass
        midLevel = mid
        trebleLevel = treble
        leftLevel = left
        rightLevel = right
    }
    
    /// Set visualization parameters
    /// - Parameters:
    ///   - sensitivity: Audio sensitivity multiplier (0.0-1.0)
    ///   - motionIntensity: Motion intensity multiplier (0.0-1.0)
    ///   - colorIntensity: Color intensity parameter (0.0-1.0)
    public func setVisualizationParameters(sensitivity: Float, motionIntensity: Float, colorIntensity: Float) {
        self.sensitivity = sensitivity
        self.motionIntensity = motionIntensity
        self.colorIntensity = colorIntensity
    }
    
    /// Set the visualization mode
    /// - Parameter mode: The visualization mode (0: Spectrum, 1: Waveform, 2: Particles, 3: Neural)
    public func setVisualizationMode(_ mode: VisualizationMode) {
        if currentVisualizationMode != mode.rawValue {
            previousVisualizationMode = currentVisualizationMode
            currentVisualizationMode = mode.rawValue
            
            // Start transition
            isInTransition = true
            transitionStartTime = CACurrentMediaTime()
            transitionProgress = 0.0
        }
    }
    
    /// Set the theme for visualization
    /// - Parameter theme: The theme index (0-7)
    public func setTheme(_ theme: Theme) {
        themeIndex = theme.rawValue
        
        // Update colors based on the theme
        switch theme {
        case .classic:
            primaryColor = Constants.defaultPrimaryColor
            secondaryColor = Constants.defaultSecondaryColor
            backgroundColor = Constants.defaultBackgroundColor
            accentColor = Constants.defaultAccentColor
            
        case .neon:
            primaryColor = SIMD4<Float>(0.0, 1.0, 0.7, 1.0)
            secondaryColor = SIMD4<Float>(1.0, 0.0, 0.5, 1.0)
            backgroundColor = SIMD4<Float>(0.05, 0.0, 0.1, 1.0)
            accentColor = SIMD4<Float>(1.0, 1.0, 0.0, 1.0)
            
        case .monochrome:
            primaryColor = SIMD4<Float>(1.0, 1.0, 1.0, 1.0)
            secondaryColor = SIMD4<Float>(0.8, 0.8, 0.8, 1.0)
            backgroundColor = SIMD4<Float>(0.0, 0.0, 0.0, 1.0)
            accentColor = SIMD4<Float>(0.4, 0.4, 0.4, 1.0)
            
        case .cosmic:
            primaryColor = SIMD4<Float>(0.5, 0.0, 1.0, 1.0)
            secondaryColor = SIMD4<Float>(0.0, 0.5, 1.0, 1.0)
            backgroundColor = SIMD4<Float>(0.0, 0.0, 0.15, 1.0)
            accentColor = SIMD4<Float>(1.0, 0.5, 0.0, 1.0)
            
        case .sunset:
            primaryColor = SIMD4<Float>(1.0, 0.4, 0.0, 1.0)
            secondaryColor = SIMD4<Float>(0.8, 0.0, 0.3, 1.0)
            backgroundColor = SIMD4<Float>(0.15, 0.05, 0.1, 1.0)
            accentColor = SIMD4<Float>(1.0, 0.8, 0.0, 1.0)
            
        case .ocean:
            primaryColor = SIMD4<Float>(0.0, 0.6, 0.9, 1.0)
            secondaryColor = SIMD4<Float>(0.0, 0.3, 0.6, 1.0)
            backgroundColor = SIMD4<Float>(0.0, 0.1, 0.2, 1.0)
            accentColor = SIMD4<Float>(0.0, 1.0, 0.8, 1.0)
            
        case .forest:
            primaryColor = SIMD4<Float>(0.0, 0.8, 0.4, 1.0)
            secondaryColor = SIMD4<Float>(0.4, 0.6, 0.0, 1.0)
            backgroundColor = SIMD4<Float>(0.05, 0.1, 0.05, 1.0)
            accentColor = SIMD4<Float>(1.0, 0.9, 0.0, 1.0)
            
        case .custom:
            // Custom theme preserves the current colors
            break
        }
    }
    
    /// Set custom colors for visualization
    /// - Parameters:
    ///   - primary: Primary theme color
    ///   - secondary: Secondary theme color
    ///   - background: Background color
    ///   - accent: Accent color for highlights
    public func setCustomColors(primary: SIMD4<Float>, secondary: SIMD4<Float>, 
                              background: SIMD4<Float>, accent: SIMD4<Float>) {
        primaryColor = primary
        secondaryColor = secondary
        backgroundColor = background
        accentColor = accent
        
        // Switch to custom theme
        themeIndex = Theme.custom.rawValue
    }
    
    /// Set neural visualization parameters
    /// - Parameters:
    ///   - energy: Energy parameter for neural visualization (0.0-1.0)
    ///   - pleasantness: Pleasantness parameter (0.0-1.0)
    ///   - complexity: Complexity parameter (0.0-1.0)
    ///   - beatDetected: Beat detection flag (0.0 or 1.0)
    public func setNeuralParameters(energy: Float, pleasantness: Float, complexity: Float, beatDetected: Float) {
        neuralEnergy = energy
        neuralPleasantness = pleasantness
        neuralComplexity = complexity
        self.beatDetected = beatDetected
    }
    
    /// Reset the audio data to zeros
    private func resetAudioData() {
        for i in 0..<Constants.maxAudioFrames {
            audioData[i] = 0.0
        }
        
        bassLevel = 0.0
        midLevel = 0.0
        trebleLevel = 0.0
        leftLevel = 0.0
        rightLevel = 0.0
    }
    
    /// Set up Metal device and command queue
    private func setupMetal() throws {
        // Get the default device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RendererError.deviceNotAvailable
        }
        self.device = device
        
        // Create the command queue
        guard let commandQueue = device.makeCommandQueue() else {
            throw RendererError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
    }
    
    /// Set up the render pipeline state
    private func setupRenderPipeline() throws {
        // Load the shader library
        let library: MTLLibrary
        do {
            // Attempt to load the default library which contains our shader functions
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            throw RendererError.libraryCreationFailed(error)
        }
        
        // Get the vertex and fragment functions
        guard let vertexFunction = library.makeFunction(name: Constants.vertexFunctionName) else {
            throw RendererError.vertexFunctionNotFound
        }
        
        guard let fragmentFunction = library.makeFunction(name: Constants.fragmentFunctionName) else {
            throw RendererError.fragmentFunctionNotFound
        }
        
        // Create the render pipeline descriptor
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        
        // Configure the pixel format to match the MetalView's format
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        // Create the render pipeline state
        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            throw RendererError.pipelineCreationFailed(error)
        }
    }
    
    /// Set up the vertex and uniform buffers
    private func setupBuffers() throws {
        // Create a square plane with 2 triangles
        let vertices: [SIMD4<Float>] = [
            SIMD4<Float>(-1.0, -1.0, 0.0, 1.0),  // Bottom left
            SIMD4<Float>( 1.0, -1.0, 0.0, 1.0),  // Bottom right
            SIMD4<Float>(-1.0,  1.0, 0.0, 1.0),  // Top left
            SIMD4<Float>( 1.0,  1.0, 0.0, 1.0)   // Top right
        ]
        
        // Create vertex buffer
        guard let vertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: vertices.count * MemoryLayout<SIMD4<Float>>.stride,
            options: .storageModeShared
        ) else {
            throw RendererError.bufferAllocationFailed
        }
        self.vertexBuffer = vertexBuffer
        
        // Create uniform buffer (with triple buffering)
        let uniformBufferSize = Constants.initialUniformBufferSize * Constants.maxInflightBuffers
        guard let uniformBuffer = device.makeBuffer(
            length: uniformBufferSize,
            options: .storageModeShared
        ) else {
            throw RendererError.bufferAllocationFailed
        }
        self.uniformBuffer = uniformBuffer
    }
    
    // MARK: - Rendering
    
    /// Render the visualization to the provided drawable
    /// - Parameters:
    ///   - drawable: The Metal drawable to render to
    ///   - renderPassDescriptor: The render pass descriptor
    public func render(to drawable: CAMetalDrawable, with renderPassDescriptor: MTLRenderPassDescriptor) {
        // Wait for a free buffer slot
        _ = semaphore.wait(timeout: .distantFuture)
        
        // Update timing information
        updateTimingInfo()
        
        // Update the uniform buffer with latest audio and visualization data
        updateUniformBuffer()
        
        // Create a command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            semaphore.signal()
            return
        }
        
        // Set completion handler
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.semaphore.signal()
        }
        
        // Configure the render command encoder
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(
            red: Double(backgroundColor.x),
            green: Double(backgroundColor.y),
            blue: Double(backgroundColor.z),
            alpha: Double(backgroundColor.w)
        )
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            semaphore.signal()
            return
        }
        
        // Set the render pipeline state
        renderEncoder.setRenderPipelineState(renderPipelineState)
        
        // Set the vertex buffer
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        
        // Calculate offset into the uniform buffer for the current frame
        let uniformBufferOffset = bufferIndex * Constants.initialUniformBufferSize
        
        // Set the uniform buffer
        renderEncoder.setVertexBuffer(uniformBuffer, offset: uniformBufferOffset, index: 1)
        renderEncoder.setFragmentBuffer(uniformBuffer, offset: uniformBufferOffset, index: 0)
        
        // Draw the visualization (2 triangles, forming a quad)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        
        // Finish encoding
        renderEncoder.endEncoding()
        
        // Present the drawable
        commandBuffer.present(drawable)
        
        // Commit the command buffer
        commandBuffer.commit()
        
        // Update buffer index (cycle through available buffers)
        bufferIndex = (bufferIndex + 1) % Constants.maxInflightBuffers
        
        // Track frame rate
        trackFrameRate()
    }
    
    /// Update the uniform buffer with the latest state
    private func updateUniformBuffer() {
        // Get pointer to the current region of the uniform buffer
        let bufferPointer = uniformBuffer.contents().advanced(by: bufferIndex * Constants.initialUniformBufferSize)
        var localUniforms = AudioUniforms()
        
        // Copy audio data
        for i in 0..<min(Constants.maxAudioFrames, audioData.count) {
            localUniforms.audioData[i] = audioData[i]
        }
        
        // Set audio levels
        localUniforms.bassLevel = bassLevel
        localUniforms.midLevel = midLevel
        localUniforms.trebleLevel = trebleLevel
        localUniforms.leftLevel = leftLevel
        localUniforms.rightLevel = rightLevel
        
        // Set theme colors
        localUniforms.primaryColor = primaryColor
        localUniforms.secondaryColor = secondaryColor
        localUniforms.backgroundColor = backgroundColor
        localUniforms.accentColor = accentColor
        
        // Set animation parameters
        localUniforms.time = Float(CACurrentMediaTime() - startTime)
        localUniforms.sensitivity = sensitivity
        localUniforms.motionIntensity = motionIntensity
        localUniforms.themeIndex = themeIndex
        
        // Update visualization mode and transition
        updateVisualizationTransition()
        localUniforms.visualizationMode = currentVisualizationMode
        localUniforms.previousMode = previousVisualizationMode
        localUniforms.transitionProgress = transitionProgress
        
        // Set additional visualization parameters
        localUniforms.colorIntensity = colorIntensity
        localUniforms.spectrumSmoothing = spectrumSmoothing
        localUniforms.particleCount = particleCount
        
        // Set neural visualization parameters
        localUniforms.neuralEnergy = neuralEnergy
        localUniforms.neuralPleasantness = neuralPleasantness
        localUniforms.neuralComplexity = neuralComplexity
        localUniforms.beatDetected = beatDetected
        
        // Copy to the uniform buffer
        memcpy(bufferPointer, &localUniforms, Constants.initialUniformBufferSize)
    }
    
    /// Update visualization mode transition progress
    private func updateVisualizationTransition() {
        if isInTransition {
            let currentTime = CACurrentMediaTime()
            let elapsed = currentTime - transitionStartTime
            
            if elapsed >= transitionDuration {
                // Transition complete
                transitionProgress = 1.0
                isInTransition = false
            } else {
                // Update transition progress (0.0 to 1.0)
                transitionProgress = Float(elapsed / transitionDuration)
            }
        }
    }
    
    /// Update timing information for animations
    private func updateTimingInfo() {
        lastUpdateTime = CACurrentMediaTime()
    }
    
    /// Track frame rate for performance monitoring
    private func trackFrameRate() {
        frameCounter += 1
        
        let currentTime = CACurrentMediaTime()
        let elapsedTime = currentTime - lastFrameTimestamp
        
        // Update frame rate calculation every second
        if elapsedTime > 1.0 {
            let frameRate = Double(frameCounter) / elapsedTime
            print("Frame rate: \(Int(frameRate)) fps")
            
            frameCounter = 0
            lastFrameTimestamp = currentTime
        }
    }
    
    // MARK: - Public Methods
    
    /// Set the viewport size
    /// - Parameter size: The new viewport size
    public func setViewportSize(_ size: CGSize) {
        viewportSize = size
    }
    
    /// Update audio data for visualization
    /// - Parameter data: FFT frequency data (normalized 0.0-1.0)
    public func updateAudioData(_ data: [Float]) {
        // Copy incoming data, ensuring we don't exceed the buffer size
        let count = min(data.count, Constants.maxAudioFrames)
        for i in 0..<count {
            audioData[i] = data[i]
        }
    }
    
    /// Set audio levels for visualization
    /// - Parameters:
    ///   - bass: Bass frequency level (0.0-1.0)
    ///   - mid: Mid frequency level (0.0-1.0)
    ///   - treble: Tre
