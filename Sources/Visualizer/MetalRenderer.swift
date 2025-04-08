import Foundation
import Metal
import MetalKit
import SwiftUI
import AudioBloomCore
import QuartzCore
import CoreVideo
/// Metal-based renderer for audio visualizations
public class MetalRenderer: NSObject, ObservableObject, VisualizationRenderer {
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
    
    /// The current audio data for visualization
    private var currentAudioData: [Float] = []
    
    /// The current audio levels
    private var currentLevels: (left: Float, right: Float) = (0, 0)
    
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
    
    /// Updates the renderer with new audio data
    public func update(audioData: [Float], levels: (left: Float, right: Float)) {
        currentAudioData = audioData
        currentLevels = levels
    }
    
    /// Renders a frame
    public func render() {
        guard let metalView = self.metalView else { return }
        
        // A real implementation would use the current audio data to influence the visualization
        // For now, this is just a placeholder
        
        // Render using the MetalKit view's drawable
        guard let drawable = metalView.currentDrawable,
              let commandBuffer = commandQueue?.makeCommandBuffer(),
              let renderPassDescriptor = metalView.currentRenderPassDescriptor,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor),
              let pipelineState = pipelineState,
              let vertexBuffer = vertexBuffer else {
            return
        }
        
        // Set up the render encoder
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        
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
    }
    
    /// Creates the render pipeline state
    private func createRenderPipelineState() {
        guard let device = device else { return }
        
        // Load default shaders from the bundle
        let defaultLibrary = device.makeDefaultLibrary()
        
        // In a real implementation, we would load custom shaders from a file
        // For now, we'll use default shaders if available, or create simple ones
        let vertexFunction = defaultLibrary?.makeFunction(name: "basic_vertex") ?? createBasicVertexFunction(device: device)
        let fragmentFunction = defaultLibrary?.makeFunction(name: "basic_fragment") ?? createBasicFragmentFunction(device: device)
        
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
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.framebufferOnly = true
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

