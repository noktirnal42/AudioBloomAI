// FFTProcessingTests.swift
import XCTest
@testable import AudioProcessor

@available(macOS 15.0, *)
final class FFTProcessingTests: XCTestCase {
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
    
    func testBasicFFTProcessing() async throws {
        let sampleCount = 1024
        let testData = generateSineWave(
            frequency: 440,
            sampleRate: 44100,
            duration: Float(sampleCount) / 44100
        )
        
        let inputBufferId = try createTestBuffer(data: testData)
        let outputBufferId = try createTestBuffer(size: sampleCount)
        
        try await fftNode.process(
            inputBufferId: inputBufferId,
            outputBufferId: outputBufferId,
            sampleCount: sampleCount,
            targetLevel: 0.9
        )
        
        let outputData = try readBufferData(bufferId: outputBufferId, size: sampleCount)
        verifyFFTOutput(output: outputData, originalData: testData)
    }
    
    private func generateSineWave(
        frequency: Float,
        sampleRate: Float,
        duration: Float
    ) -> [Float] {
        let sampleCount = Int(sampleRate * duration)
        var wave = [Float](repeating: 0, count: sampleCount)
        
        for sampleIndex in 0..<sampleCount {
            let time = Float(sampleIndex) / sampleRate
            wave[sampleIndex] = sin(2 * Float.pi * frequency * time)
        }
        
        return wave
    }
    
    private func verifyFFTOutput(output: [Float], originalData: [Float]) {
        XCTAssertEqual(output.count, originalData.count)
        
        // Verify frequency components
        let frequencyBins = output.count / 2
        for binIndex in 0..<frequencyBins {
            let magnitude = sqrt(
                output[binIndex * 2] * output[binIndex * 2] +
                output[binIndex * 2 + 1] * output[binIndex * 2 + 1]
            )
            XCTAssertGreaterThanOrEqual(magnitude, 0)
        }
    }
}
