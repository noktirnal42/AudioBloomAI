import Foundation
import SwiftUI
import Metal
import MetalKit

#if os(macOS)
import AppKit

/// SwiftUI wrapper for the Metal-based audio visualizer view on macOS
public struct AudioVisualizerView: NSViewRepresentable {
    // The audio visualizer instance
    private let visualizer: AudioVisualizer
    
    /// Initialize with an audio visualizer
    /// - Parameter visualizer: The audio visualizer instance
    public init(visualizer: AudioVisualizer) {
        self.visualizer = visualizer
    }
    
    /// Create the NSView for the visualization
    public func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 60
        mtkView.enableSetNeedsDisplay = true
        mtkView.framebufferOnly = true
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.device = MTLCreateSystemDefaultDevice()
        
        return mtkView
    }
    
    /// Update the NSView when SwiftUI updates
    public func updateNSView(_ nsView: MTKView, context: Context) {
        // Update view size if needed
        if let metalLayer = nsView.layer as? CAMetalLayer {
            let scale = nsView.window?.screen?.backingScaleFactor ?? 1.0
            metalLayer.contentsScale = scale
            
            let size = nsView.bounds.size
            visualizer.updateViewportSize(size)
        }
    }
    
    /// Create the coordinator for handling MTKView delegate callbacks
    public func makeCoordinator() -> Coordinator {
        Coordinator(visualizer: visualizer)
    }
    
    /// Coordinator class for handling MTKView delegate methods
    public class Coordinator: NSObject, MTKViewDelegate {
        private let visualizer: AudioVisualizer
        
        init(visualizer: AudioVisualizer) {
            self.visualizer = visualizer
            super.init()
        }
        
        /// Called when the view needs to be drawn
        public func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let renderPassDescriptor = view.currentRenderPassDescriptor else {
                return
            }
            
            visualizer.render(to: drawable, with: renderPassDescriptor)
        }
        
        /// Called when the drawable size changes
        public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            visualizer.updateViewportSize(size)
        }
    }
}

#else
import UIKit

/// SwiftUI wrapper for the Metal-based audio visualizer view on iOS/tvOS
public struct AudioVisualizerView: UIViewRepresentable {
    // The audio visualizer instance
    private let visualizer: AudioVisualizer
    
    /// Initialize with an audio visualizer
    /// - Parameter visualizer: The audio visualizer instance
    public init(visualizer: AudioVisualizer) {
        self.visualizer = visualizer
    }
    
    /// Create the UIView for the visualization
    public func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 60
        mtkView.enableSetNeedsDisplay = true
        mtkView.framebufferOnly = true
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.device = MTLCreateSystemDefaultDevice()
        
        return mtkView
    }
    
    /// Update the UIView when SwiftUI updates
    public func updateUIView(_ uiView: MTKView, context: Context) {
        // Update view size if needed
        if let metalLayer = uiView.layer as? CAMetalLayer {
            let scale = uiView.window?.screen.scale ?? 1.0
            metalLayer.contentsScale = scale
            
            let size = uiView.bounds.size
            visualizer.updateViewportSize(size)
        }
    }
    
    /// Create the coordinator for handling MTKView delegate callbacks
    public func makeCoordinator() -> Coordinator {
        Coordinator(visualizer: visualizer)
    }
    
    /// Coordinator class for handling MTKView delegate methods
    public class Coordinator: NSObject, MTKViewDelegate {
        private let visualizer: AudioVisualizer
        
        init(visualizer: AudioVisualizer) {
            self.visualizer = visualizer
            super.init()
        }
        
        /// Called when the view needs to be drawn
        public func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let renderPassDescriptor = view.currentRenderPassDescriptor else {
                return
            }
            
            visualizer.render(to: drawable, with: renderPassDescriptor)
        }
        
        /// Called when the drawable size changes
        public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            visualizer.updateViewportSize(size)
        }
    }
}
#endif

// MARK: - Preview Provider

struct AudioVisualizerView_Previews: PreviewProvider {
    static var previews: some View {
        AudioVisualizerView(visualizer: AudioVisualizer())
            .frame(width: 600, height: 400)
    }
}

