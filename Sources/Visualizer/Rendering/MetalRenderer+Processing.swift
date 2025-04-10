// MetalRenderer extensions
import Metal
import Foundation

extension MetalRenderer {
    func processFrame(_ frame: MTLBuffer) throws -> MTLTexture {
        let validatedInput = try validateInput(frame)
        let preparedState = try prepareRenderState()
        let processedFrame = try processFrameData(frame: validatedInput, state: preparedState)
        return try finalizeOutput(processedFrame)
    }
    
    private func validateInput(_ frame: MTLBuffer) throws -> MTLBuffer {
        guard frame.length > 0 else {
            throw RenderError.invalidInput
        }
        return frame
    }
    
    private func prepareRenderState() throws -> RenderState {
        guard let state = renderPipelineState else {
            throw RenderError.invalidState
        }
        return RenderState(pipeline: state)
    }
    
    private func processFrameData(frame: MTLBuffer, state: RenderState) throws -> MTLBuffer {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw RenderError.commandCreationFailed
        }
        try executeRenderPass(commandBuffer, frame: frame, state: state)
        return frame
    }
    
    private func finalizeOutput(_ frame: MTLBuffer) throws -> MTLTexture {
        guard let texture = createOutputTexture(from: frame) else {
            throw RenderError.outputCreationFailed
        }
        return texture
    }
}
