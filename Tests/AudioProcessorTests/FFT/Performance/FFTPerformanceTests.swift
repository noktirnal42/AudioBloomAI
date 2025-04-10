// FFTPerformanceTests.swift 
import XCTest
@testable import AudioProcessor

@available(macOS 15.0, *)
final class FFTPerformanceTests: XCTestCase {
    var fftNode: FFTNode!
    var pipelineCore: AudioPipelineCore!
    
    override func setUp() async throws {
        super.setUp()
        fftNode = FFTNode()
        pipelineCore = try AudioPipelineCore()
    }
    
    override func tearDown() {
        fftNode = nil
        pipelineCore = nil
        super.tearDown()
    }
    
    func testFFTProcessingPerformance() async throws {
        // Setup test parameters
        let sampleCount = 1024
        let targetLevel = 0.9
        let testData = generateTestData(sampleCount: sampleCount)
        
        // Create buffers
        let inputBufferId = try createTestBuffer(data: testData)
        let outputBufferId = try createTestBuffer(size: sampleCount)
        
        // Measure performance
        measure {
            Task {
                try await fftNode.process(
                    inputBufferId: inputBufferId,
                    outputBufferId: outputBufferId,
                    sampleCount: sampleCount,
                    targetLevel: targetLevel
                )
            }
        }
    }
    
    private func generateTestData(sampleCount: Int) -> [Float] {
        var data = [Float](repeating: 0, count: sampleCount)
        for sampleIndex in 0..<sampleCount {
            let phase = Float(sampleIndex) / Float(sampleCount) * 2 * Float.pi
            data[sampleIndex] = sin(phase) * 0.5
        }
        return data
    }
    
    private func createTestBuffer(data: [Float]) throws -> AudioBufferID {
        let bufferId = try pipelineCore.allocateBuffer(size: data.count * MemoryLayout<Float>.stride)
        try pipelineCore.updateBuffer(
            id: bufferId,
            data: data,
            size: data.count * MemoryLayout<Float>.stride
        )
        return bufferId
    }
    
    private func createTestBuffer(size: Int) throws -> AudioBufferID {
        return try pipelineCore.allocateBuffer(size: size * MemoryLayout<Float>.stride)
    }
}
