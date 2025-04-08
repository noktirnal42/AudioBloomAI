import Foundation
import Metal
import MetalKit
import SwiftUI
import AudioBloomCore

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
    
    /// Timer for animation
    private var displayLink: CADisplayLink?
    
    /// Prepares the renderer for drawing
    public func prepareRenderer() {
        // Get the default Metal device
        guard let device = MTLDevice.default else {
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
    
    /// Sets up the display link for rendering
    private func setupDisplayLink() {
        displayLink = CADisplayLink(target: self, selector: #selector(displayLinkDidFire))
        displayLink?.preferredFramesPerSecond = AudioBloomCore.Constants.defaultFrameRate
        displayLink?.add(to: .main, forMode: .common)
    }
    
    /// Called when the display link fires
    @objc private func displayLinkDidFire() {
        render()
    }
    
    /// Configures a MetalKit view for rendering
    public func configure(metalView: MTKView) {
        guard let device = device else { return }
        
        metalView.device = device
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.framebufferOnly = true
        
        self.metalView = metalView
    }
}

/// A SwiftUI wrapper for a MetalKit view
public struct MetalView: NSViewRepresentable {
    /// The renderer to use
    public var renderer: MetalRenderer
    
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

