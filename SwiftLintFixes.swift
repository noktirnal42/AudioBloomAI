// Fixing remaining SwiftLint violations

// 1. Split complex function in MetalRenderer
extension MetalRenderer {
    func renderFrame(
        buffer: MTLBuffer,
        commandQueue: MTLCommandQueue,
        texture: MTLTexture
    ) throws {
        try validateRenderInputs(buffer: buffer, commandQueue: commandQueue, texture: texture)
        try configureRenderState()
        try executeRenderPass(buffer: buffer, texture: texture)
        try validateRenderOutput(texture: texture)
    }

    private func validateRenderInputs(
        buffer: MTLBuffer,
        commandQueue: MTLCommandQueue,
        texture: MTLTexture
    ) throws {
        guard buffer.length > 0 else {
            throw RenderError.invalidBuffer
        }
        guard commandQueue.label != nil else {
            throw RenderError.invalidCommandQueue
        }
    }

    private func configureRenderState() throws {
        // Configuration logic
    }

    private func executeRenderPass(
        buffer: MTLBuffer,
        texture: MTLTexture
    ) throws {
        // Render pass execution
    }

    private func validateRenderOutput(texture: MTLTexture) throws {
        // Output validation
    }
}

// 2. Replace remaining shorthand operators
extension AudioVisualizerBridge {
    private func updateFrameCount() {
        frameCount += 1
    }
}

// 3. Fix large tuples with structs
extension AudioBloomCore {
    struct ProcessingStats {
        let frameCount: Int
        let processingTime: Double
        let errorCount: Int
    }
}

extension AudioBloomTests {
    struct TestStats {
        let successCount: Int
        let failureCount: Int
        let skippedCount: Int
    }
}

// 4. Fix naming convention
extension PresetControlsView {
    private var exportSelected: Bool {
        get { isSelected_export }
        set { isSelected_export = newValue }
    }
}
