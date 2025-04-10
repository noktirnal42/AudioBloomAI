// MetalPerformanceTests.swift
import XCTest
@testable import AudioProcessor

@available(macOS 15.0, *)
final class MetalPerformanceTests: XCTestCase {
    var metalCore: MetalComputeCore!
    
    override func setUp() {
        super.setUp()
        metalCore = try! MetalComputeCore()
    }
    
    override func tearDown() {
        metalCore = nil
        super.tearDown()
    }
    
    func testAudioNormalization() async throws {
        // Setup test data
        let sampleCount = 1024
        let targetLevel = 0.9
        let testAudio = generateTestAudio(sampleCount: sampleCount)
        
        // Create buffers
        let inputBufferId = try createTestBuffer(data: testAudio)
        let outputBufferId = try createTestBuffer(size: sampleCount, fillValue: 0.0)
        
        // Create expectation
        let expectation = XCTestExpectation(description: "Normalization Completion")
        
        // Perform normalization
        metalCore.normalizeAudio(
            inputBufferId: inputBufferId,
            outputBufferId: outputBufferId,
            sampleCount: sampleCount,
            targetLevel: targetLevel
        ) { result in
            switch result {
            case .success:
                expectation.fulfill()
            case .failure(let error):
                XCTFail("Audio normalization failed: \(error)")
            }
        }
        
        await fulfillment(of: [expectation], timeout: 5.0)
        
        // Verify results
        let normalizedAudio = try readBufferData(bufferId: outputBufferId, size: sampleCount)
        verifyNormalization(original: testAudio, normalized: normalizedAudio, targetLevel: targetLevel)
    }
    
    private func generateTestAudio(sampleCount: Int) -> [Float] {
        var audio = [Float](repeating: 0, count: sampleCount)
        for sampleIndex in 0..<sampleCount {
            let phase = Float(sampleIndex) / Float(sampleCount) * 2 * Float.pi
            audio[sampleIndex] = sin(phase) * 0.5
        }
        return audio
    }
    
    private func verifyNormalization(original: [Float], normalized: [Float], targetLevel: Float) {
        var originalPeak: Float = 0
        var normalizedPeak: Float = 0
        
        for sampleIndex in 0..<original.count {
            originalPeak = max(originalPeak, abs(original[sampleIndex]))
            normalizedPeak = max(normalizedPeak, abs(normalized[sampleIndex]))
        }
        
        XCTAssertEqual(normalizedPeak, targetLevel, accuracy: 0.01)
        
        let scaleFactor = targetLevel / originalPeak
        for sampleIndex in 0..<min(10, original.count) {
            let expectedSample = original[sampleIndex] * scaleFactor
            XCTAssertEqual(normalized[sampleIndex], expectedSample, accuracy: 0.001)
        }
    }
}
