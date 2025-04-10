// MetalRenderer extensions for SwiftLint compliance
import Metal
import Foundation

extension MetalRenderer {
    // MARK: - Render Frame Processing
    
    func renderFrame(
        buffer: MTLBuffer,
        commandQueue: MTLCommandQueue,
        texture: MTLTexture
    ) throws {
        try validateRenderInputs(
            buffer: buffer,
            commandQueue: commandQueue,
            texture: texture
        )
        try configureRenderState()
        try executeRenderPass(buffer: buffer, texture: texture)
        try validateRenderOutput(texture: texture)
    }
    
    // MARK: - Private Methods
    
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
        guard texture.width > 0, texture.height > 0 else {
            throw RenderError.invalidTexture
        }
    }
    
    private func configureRenderState() throws {
        guard let renderPipelineState = renderPipelineState else {
            throw RenderError.processingFailed("Missing render pipeline state")
        }
        
        currentRenderPipelineState = renderPipelineState
    }
    
    private func executeRenderPass(
        buffer: MTLBuffer,
        texture: MTLTexture
    ) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw RenderError.processingFailed("Failed to create command buffer")
        }
        
        try setupRenderPass(commandBuffer: commandBuffer)
        try executeRenderCommands(
            commandBuffer: commandBuffer,
            buffer: buffer,
            texture: texture
        )
        
        commandBuffer.commit()
    }
    
    private func setupRenderPass(commandBuffer: MTLCommandBuffer) throws {
        // Render pass setup logic
    }
    
    private func executeRenderCommands(
        commandBuffer: MTLCommandBuffer,
        buffer: MTLBuffer,
        texture: MTLTexture
    ) throws {
        // Render commands execution
    }
    
    private func validateRenderOutput(texture: MTLTexture) throws {
        // Output validation logic
    }
}
