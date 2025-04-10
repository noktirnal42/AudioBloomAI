// MetalProcessingTests.swift
import XCTest
@testable import AudioProcessor

@available(macOS 15.0, *)
final class MetalProcessingTests: XCTestCase {
    var metalCore: MetalComputeCore!
    
    override func setUp() {
        super.setUp()
        metalCore = try MetalComputeCore()
    }
    
    override func tearDown() {
        metalCore = nil
        super.tearDown()
    }
    
    func testConcurrentProcessing() async throws {
        let operationCount = 20
        let bufferSize = 1024
        
        var inputBuffers: [UInt64] = []
        var outputBuffers: [UInt64] = []
        
        // Create test buffers
        for _ in 0..<operationCount {
            let inputId = try createTestBuffer(size: bufferSize)
            let outputId = try createTestBuffer(size: bufferSize, fillValue: 0.0)
            
            inputBuffers.append(inputId)
            outputBuffers.append(outputId)
        }
        
        // Create expectations for concurrent operations
        var expectations: [XCTestExpectation] = []
        
        for operationIndex in 0..<operationCount {
            let expectation = XCTestExpectation(
                description: "Concurrent Operation \(operationIndex)"
            )
            expectations.append(expectation)
            
            metalCore.normalizeAudio(
                inputBufferId: inputBuffers[operationIndex],
                outputBufferId: outputBuffers[operationIndex],
                sampleCount: bufferSize,
                targetLevel: 0.9
            ) { result in
                switch result {
                case .success:
                    expectation.fulfill()
                case .failure(let error):
                    XCTFail("Operation \(operationIndex) failed: \(error)")
                }
            }
        }
        
        await fulfillment(of: expectations, timeout: 10.0)
        
        let stats = metalCore.getBufferStats()
        XCTAssertEqual(stats.activeCount, 0, "No active buffers expected")
        XCTAssertEqual(
            stats.pooledCount,
            operationCount,
            "Buffer pool should match operation count"
        )
    }
}
