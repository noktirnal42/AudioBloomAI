import Foundation
import Metal
import MetalKit
import SwiftUI
import AudioBloomCore
import QuartzCore
import CoreVideo
import simd
/// Structure that matches the AudioUniforms in the Metal shader
struct AudioUniforms {
    var audioData: [Float]       // FFT frequency data (1024 elements)
    var bassLevel: Float         // Low frequency intensity
    var midLevel: Float          // Mid frequency intensity
    var trebleLevel: Float       // High frequency intensity
    var leftLevel: Float         // Left channel volume
    var rightLevel: Float        // Right channel volume
    
    // Theme colors
    var primaryColor: SIMD4<Float>    // Primary theme color
    var secondaryColor: SIMD4<Float>  // Secondary theme color
    var backgroundColor: SIMD4<Float> // Background color
    
    // Animation parameters
    var time: Float              // Current time in seconds
    var sensitivity: Float       // Audio sensitivity (0.0-1.0)
    var motionIntensity: Float   // Motion intensity (0.0-1.0)
    var themeIndex: Float        // Current theme index (0-3 for Classic, Neon, Monochrome, Cosmic)
    
    // Fixed size to match shader expectations
    static let audioDataSize = 1024
    
    init() {
        audioData = [Float](repeating: 0, count: Self.audioDataSize)
        bassLevel = 0
        midLevel = 0
        trebleLevel = 0
        leftLevel = 0
        rightLevel = 0
        primaryColor = SIMD4<Float>(0, 0.5, 1.0, 1.0)
        secondaryColor = SIMD4<Float>(1.0, 0, 0.5, 1.0)
        backgroundColor = SIMD4<Float>(0, 0, 0.1, 1.0)
        time = 0
        sensitivity = 0.75
        motionIntensity = 0.8
        themeIndex = 0 // Classic
    }
    
    var bufferSize: Int {
        // Calculate the total size of the structure
        return MemoryLayout<Float>.size * Self.audioDataSize + // audioData array
               MemoryLayout<Float>.size * 6 +                  // 6 float levels
               MemoryLayout<SIMD4<Float>>.size * 3 +           // 3 color vectors
               MemoryLayout<Float>.size * 4                    // 4 animation parameters
    }
}

/// Metal-based renderer for audio visualizations
public class MetalRenderer: NSObject, ObservableObject, VisualizationRenderer, VisualizationParameterReceiver {
    /// Published property that indicates if the renderer is ready
    @Published public private(set) var isReady: Bool = false
    
    /// The Metal device used for rendering
    private var device: MTLDevice?
    
    /// The command queue for submitting work to the GPU
    private var commandQueue: MTLCommandQueue?
    
    /// The render pipeline state
    private var pipelineState: MTLRenderPipelineState?
    
    /// The vertex buffer
    private var vertexBuffer: MTLBuffer?
    
    /// The uniform buffer for audio data
    private var uniformBuffer: MTLBuffer?
    
    /// The uniforms structure for Metal shader
    private var uniforms = AudioUniforms()
    
    /// The current audio data for visualization
    private var currentAudioData: [Float] = []
    
    /// The current audio levels
    private var currentLevels: (left: Float, right: Float) = (0, 0)
    
    /// Start time for animation timing
    private let startTime = CACurrentMediaTime()
    
    /// The MetalKit view for rendering
    private var metalView: MTKView?
    
    /// Display link for macOS rendering synchronization
    private var displayLink: CVDisplayLink?
    
    /// Queue for handling display link callbacks
    private let displayLinkQueue = DispatchQueue(label: "com.audiobloom.displaylink", qos: .userInteractive)
    
    /// Flag indicating if rendering is active
    private var isRenderingActive = false
    
    /// Prepares the renderer for drawing
    public func prepareRenderer() {
        // Get the default Metal device using the proper macOS API
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal device not found")
            return
        }
        
        self.device = device
        
        // Create a command queue
        guard let commandQueue = device.makeCommandQueue() else {
            print("Failed to create Metal command queue")
            return
        }
        
        self.commandQueue = commandQueue
        
        // Create basic vertex data for a quad
        createVertexBuffer()
        
        // Create render pipeline state
        createRenderPipelineState()
        
        // Set up display link for rendering
        setupDisplayLink()
        
        isReady = true
    }
    
    /// Updates parameters for the visualization
    public func updateParameters(_ parameters: [String: Any]) {
        // Update color parameters
        if let primaryColor = parameters["primaryColor"] as? [CGFloat], primaryColor.count >= 4 {
            uniforms.primaryColor = SIMD4<Float>(Float(primaryColor[0]), Float(primaryColor[1]), 
                                               Float(primaryColor[2]), Float(primaryColor[3]))
        }
        
        if let secondaryColor = parameters["secondaryColor"] as? [CGFloat], secondaryColor.count >= 4 {
            uniforms.secondaryColor = SIMD4<Float>(Float(secondaryColor[0]), Float(secondaryColor[1]), 
                                                 Float(secondaryColor[2]), Float(secondaryColor[3]))
        }
        
        if let backgroundColor = parameters["backgroundColor"] as? [CGFloat], backgroundColor.count >= 4 {
            uniforms.backgroundColor = SIMD4<Float>(Float(backgroundColor[0]), Float(backgroundColor[1]), 
                                                  Float(backgroundColor[2]), Float(backgroundColor[3]))
        }
        
        // Update other parameters
        if let sensitivity = parameters["sensitivity"] as? Float {
            uniforms.sensitivity = sensitivity
        }
        
        if let motionIntensity = parameters["motionIntensity"] as? Float {
            uniforms.motionIntensity = motionIntensity
        }
        
        // Update theme index based on the theme name
        if let themeName = parameters["theme"] as? String {
            switch themeName {
            case AudioBloomCore.VisualTheme.classic.rawValue:
                uniforms.themeIndex = 0
            case AudioBloomCore.VisualTheme.neon.rawValue:
                uniforms.themeIndex = 1
            case AudioBloomCore.VisualTheme.monochrome.rawValue:
                uniforms.themeIndex = 2
            case AudioBloomCore.VisualTheme.cosmic.rawValue:
                uniforms.themeIndex = 3
            default:
                uniforms.themeIndex = 0
            }
        }
    }
    
    /// Updates the renderer with new audio data
    public func update(audioData: [Float], levels: (left: Float, right: Float)) {
        currentAudioData = audioData
        currentLevels = levels
        
        // Process audio data into frequency bands
        processBands(from: audioData)
        
        // Update audio data in uniforms
        updateAudioData(audioData: audioData, levels: levels)
    }
    
    /// Processes audio data into frequency bands (bass, mid, treble)
    private func processBands(from audioData: [Float]) {
        guard !audioData.isEmpty else { return }
        
        // Simple band calculation based on index ranges
        // In a more sophisticated implementation, we would use proper frequency ranges
        let bandSize = audioData.count / 3
        
        // Bass frequencies (low end)
        let bassRange = 0..<min(bandSize, audioData.count)
        let bassSum = bassRange.reduce(0) { $0 + audioData[$1] }
        let bassAvg = bassSum / Float(bassRange.count)
        
        // Mid frequencies
        let midRange = min(bandSize, audioData.count)..<min(bandSize * 2, audioData.count)
        let midSum = midRange.reduce(0) { $0 + audioData[$1] }
        let midAvg = midSum / Float(midRange.count)
        
        // Treble frequencies (high end)
        let trebleRange = min(bandSize * 2, audioData.count)..<audioData.count
        let trebleSum = trebleRange.reduce(0) { $0 + audioData[$1] }
        let trebleAvg = trebleSum / Float(trebleRange.count)
        
        // Update the uniforms with smoothed values (apply some damping)
        let damping: Float = 0.3 // Lower = more smoothing
        uniforms.bassLevel = uniforms.bassLevel * (1 - damping) + bassAvg * damping
        uniforms.midLevel = uniforms.midLevel * (1 - damping) + midAvg * damping
        uniforms.trebleLevel = uniforms.trebleLevel * (1 - damping) + trebleAvg * damping
    }
    
    /// Updates the uniform buffer with current audio data
    private func updateAudioData(audioData: [Float], levels: (left: Float, right: Float)) {
        // Copy audio data to uniforms (with bounds checking)
        let count = min(audioData.count, AudioUniforms.audioDataSize)
        for i in 0..<count {
            uniforms.audioData[i] = audioData[i]
        }
        
        // Update audio levels
        uniforms.leftLevel = levels.left
        uniforms.rightLevel = levels.right
        
        // Update animation time
        uniforms.time = Float(CACurrentMediaTime() - startTime)
    }
    /// Renders a frame
    public func render() {
        guard let metalView = self.metalView else { return }
        
        // Update animation time
        uniforms.time = Float(CACurrentMediaTime() - startTime)
        
        // Copy uniforms to the buffer
        updateUniformBuffer()
        
        // Render using the MetalKit view's drawable
        guard let drawable = metalView.currentDrawable,
              let commandBuffer = commandQueue?.makeCommandBuffer(),
              let renderPassDescriptor = metalView.currentRenderPassDescriptor,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor),
              let pipelineState = pipelineState,
              let vertexBuffer = vertexBuffer,
              let uniformBuffer = uniformBuffer else {
            return
        }
        
        // Set up the render encoder
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        renderEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 0)
        
        // Draw primitives
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        // End encoding and commit
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    /// Creates a basic vertex buffer for rendering a quad
    private func createVertexBuffer() {
        // Define a simple quad (two triangles) that fills the screen
        let vertices: [Float] = [
            -1.0, -1.0, 0.0, 1.0,   // Bottom left
             1.0, -1.0, 0.0, 1.0,   // Bottom right
            -1.0,  1.0, 0.0, 1.0,   // Top left
            -1.0,  1.0, 0.0, 1.0,   // Top left
             1.0, -1.0, 0.0, 1.0,   // Bottom right
             1.0,  1.0, 0.0, 1.0    // Top right
        ]
        // Create a buffer from our vertex data
        vertexBuffer = device?.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Float>.size, options: [])
        
        // Create the uniform buffer
        createUniformBuffer()
    }
    
    /// Creates the uniform buffer for audio data
    private func createUniformBuffer() {
        guard let device = device else { return }
        
        // Initialize with zeros
        uniformBuffer = device.makeBuffer(length: uniforms.bufferSize, options: [.storageModeShared])
        
        // Set initial values
        updateUniformBuffer()
    }
    
    /// Updates the contents of the uniform buffer
    /// Updates the contents of the uniform buffer
    private func updateUniformBuffer() {
        guard let uniformBuffer = uniformBuffer else { return }
        
        // Get a pointer to the buffer contents
        let bufferPointer = uniformBuffer.contents()
        
        // Copy the audio data array
        let audioDataSize = AudioUniforms.audioDataSize * MemoryLayout<Float>.size
        memcpy(bufferPointer, uniforms.audioData, audioDataSize)
        
        // Calculate offsets for other fields
        var offset = audioDataSize
        
        // Copy individual float fields
        let floatSize = MemoryLayout<Float>.size
        memcpy(bufferPointer + offset, &uniforms.bassLevel, floatSize)
        offset += floatSize
        
        memcpy(bufferPointer + offset, &uniforms.midLevel, floatSize)
        offset += floatSize
        
        memcpy(bufferPointer + offset, &uniforms.trebleLevel, floatSize)
        offset += floatSize
        
        memcpy(bufferPointer + offset, &uniforms.leftLevel, floatSize)
        offset += floatSize
        
        memcpy(bufferPointer + offset, &uniforms.rightLevel, floatSize)
        offset += floatSize
        
        // Copy color values
        let vectorSize = MemoryLayout<SIMD4<Float>>.size
        memcpy(bufferPointer + offset, &uniforms.primaryColor, vectorSize)
        offset += vectorSize
        
        memcpy(bufferPointer + offset, &uniforms.secondaryColor, vectorSize)
        offset += vectorSize
        
        memcpy(bufferPointer + offset, &uniforms.backgroundColor, vectorSize)
        offset += vectorSize
        
        // Copy animation parameters
        memcpy(bufferPointer + offset, &uniforms.time, floatSize)
        offset += floatSize
        
        memcpy(bufferPointer + offset, &uniforms.sensitivity, floatSize)
        offset += floatSize
        
        memcpy(bufferPointer + offset, &uniforms.motionIntensity, floatSize)
        offset += floatSize
        
        memcpy(bufferPointer + offset, &uniforms.themeIndex, floatSize)
    }
    
    /// Creates the render pipeline state
    private func createRenderPipelineState() {
        guard let device = device else { return }
        
        // Create options for loading the shader library
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        
        // Try to load our custom audio visualization shaders
        var library: MTLLibrary?
        do {
            let shaderPath = Bundle.module.path(forResource: "AudioVisualizer", ofType: "metal", inDirectory: "Resources/Shaders")
            if let shaderPath = shaderPath, let shaderSource = try? String(contentsOfFile: shaderPath) {
                library = try device.makeLibrary(source: shaderSource, options: options)
            } else {
                // If we can't load the file directly, try using the default library
                library = device.makeDefaultLibrary()
            }
        } catch {
            print("Error loading shader library: \(error)")
            library = device.makeDefaultLibrary()
        }
        
        // Try to load our audio visualization functions or fall back to defaults
        var vertexFunction: MTLFunction?
        var fragmentFunction: MTLFunction?
        
        if let library = library {
            vertexFunction = library.makeFunction(name: "audio_vertex_shader")
            fragmentFunction = library.makeFunction(name: "audio_fragment_shader")
        }
        
        // If we still couldn't load the functions, create basic ones
        if vertexFunction == nil {
            vertexFunction = createBasicVertexFunction(device: device)
        }
        
        if fragmentFunction == nil {
            fragmentFunction = createBasicFragmentFunction(device: device)
        }
        
        guard let vertexFunction = vertexFunction, let fragmentFunction = fragmentFunction else {
            print("Failed to create shader functions")
            return
        }
        
        // Create a pipeline descriptor
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        // Create the pipeline state
        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create render pipeline state: \(error)")
        }
    }
    
    /// Creates a basic vertex function if none is available in the bundle
    private func createBasicVertexFunction(device: MTLDevice) -> MTLFunction? {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        
        struct VertexOut {
            float4 position [[position]];
        };
        
        vertex VertexOut basic_vertex(uint vertexID [[vertex_id]],
                                    constant float4 *positions [[buffer(0)]]) {
            VertexOut out;
            out.position = positions[vertexID];
            return out;
        }
        """
        
        do {
            let library = try device.makeLibrary(source: source, options: nil)
            return library.makeFunction(name: "basic_vertex")
        } catch {
            print("Failed to create vertex function: \(error)")
            return nil
        }
    }
    
    /// Creates a basic fragment function if none is available in the bundle
    private func createBasicFragmentFunction(device: MTLDevice) -> MTLFunction? {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        
        fragment float4 basic_fragment() {
            return float4(0.0, 0.5, 1.0, 1.0); // Blue color
        }
        """
        
        do {
            let library = try device.makeLibrary(source: source, options: nil)
            return library.makeFunction(name: "basic_fragment")
        } catch {
            print("Failed to create fragment function: \(error)")
            return nil
        }
    }
    
    /// Sets up the display link for rendering on macOS
    private func setupDisplayLink() {
        // Create CVDisplayLink for macOS
        var displayLink: CVDisplayLink?
        let error = CVDisplayLinkCreateWithActiveCGDisplays(&displayLink)
        
        guard error == kCVReturnSuccess, let displayLink = displayLink else {
            print("Failed to create CVDisplayLink")
            return
        }
        
        // Set up the display link output callback
        let displayLinkCallback: CVDisplayLinkOutputCallback = { (displayLink, inNow, inOutputTime, flagsIn, flagsOut, displayLinkContext) -> CVReturn in
            // Get the renderer instance from the context
            let rendererPointer = unsafeBitCast(displayLinkContext, to: MetalRenderer.self)
            
            // Dispatch rendering to the main thread
            rendererPointer.displayLinkQueue.async {
                DispatchQueue.main.async {
                    if rendererPointer.isRenderingActive {
                        rendererPointer.render()
                    }
                }
            }
            
            return kCVReturnSuccess
        }
        
        // Set the output callback
        let opaqueRenderer = Unmanaged.passUnretained(self).toOpaque()
        CVDisplayLinkSetOutputCallback(displayLink, displayLinkCallback, opaqueRenderer)
        
        // Start the display link
        CVDisplayLinkStart(displayLink)
        self.displayLink = displayLink
        self.isRenderingActive = true
    }
    
    /// Stops the display link when it's no longer needed
    private func stopDisplayLink() {
        if let displayLink = displayLink {
            CVDisplayLinkStop(displayLink)
            self.displayLink = nil
        }
        isRenderingActive = false
    }
    
    deinit {
        stopDisplayLink()
    }
    
    /// Configures a MetalKit view for rendering
    public func configure(metalView: MTKView) {
        guard let device = device else { return }
        
        metalView.device = device
        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.enableSetNeedsDisplay = false
        metalView.isPaused = true  // We'll control rendering with our CVDisplayLink
        
        self.metalView = metalView
    }
}

// MARK: - CVDisplayLink support extensions

extension CVDisplayLink {
    var currentRefreshRate: Double {
        var actualRefreshRate = CVDisplayLinkGetActualOutputVideoRefreshPeriod(self)
        if actualRefreshRate <= 0 {
            let period = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(self)
            actualRefreshRate = Double(period.timeScale) / Double(period.timeValue)
        }
        return actualRefreshRate
    }
}
/// A SwiftUI wrapper for a MetalKit view
public struct MetalView: NSViewRepresentable {
    /// The renderer to use
    public var renderer: MetalRenderer
    
    /// Initializes a new MetalView with the given renderer
    public init(renderer: MetalRenderer) {
        self.renderer = renderer
    }
    
    /// Creates the NSView
    public func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        renderer.configure(metalView: mtkView)
        return mtkView
    }
    
    /// Updates the NSView
    public func updateNSView(_ nsView: MTKView, context: Context) {
        // Update view if needed
    }
}

