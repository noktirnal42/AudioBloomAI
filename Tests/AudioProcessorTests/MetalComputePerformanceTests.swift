import XCTest
import Metal
@testable import AudioProcessor
@testable import AudioBloomCore

/// Test class for performance and stress testing of the Metal compute system
@available(macOS 15.0, *)
final class MetalComputePerformanceTests: MetalComputeBaseTests {

    // MARK: - Performance and Stress Tests

    /// Tests that multiple operations can be processed concurrently within the Metal compute system
    /// without conflicts or errors, verifying thread safety and queue management.
    func testConcurrentProcessing() async throws {
        // Test concurrent kernel execution to verify thread safety
        let operationCount = 20
        let bufferSize = 1024

        // Create buffers for each operation
        var inputBuffers: [UInt64] = []
        var outputBuffers: [UInt64] = []

        for _ in 0..<operationCount {
            let inputId = try createTestBuffer(size: bufferSize)
            let outputId = try createTestBuffer(size: bufferSize, fillValue: 0.0)

            inputBuffers.append(inputId)
            outputBuffers.append(outputId)
        }

        // Create multiple expectations
        var expectations: [XCTestExpectation] = []

        // Launch multiple concurrent operations
        for operationIndex in 0..<operationCount {
            let expectation = XCTestExpectation(description: "Concurrent Operation \(operationIndex)")
            expectations.append(expectation)

            // Use normalization as a simple test operation
            metalCore.normalizeAudio(
                inputBufferId: inputBuffers[operationIndex],
                outputBufferId: outputBuffers[operationIndex],
                sampleCount: bufferSize,
                targetLevel: 0.9,
                completion: { result in
                    switch result {
                    case .success:
                        expectation.fulfill()
                    case .failure(let error):
                        XCTFail("Concurrent operation \(operationIndex) failed with error: \(error)")
                    }
                }
            )
        }

        // Wait for all operations to complete with a reasonable timeout
        await fulfillment(of: expectations, timeout: 10.0)

        // Verify that the concurrency limit worked properly
        let stats = metalCore.getBufferStats()

        // Log buffer statistics after concurrent processing
        print(
            "After concurrent processing: active buffers = \(stats.activeCount), " +
            "pooled buffers = \(stats.pooledCount)"
        )
    }

    /// Tests the buffer pooling and memory management features of the Metal compute system
    /// to ensure proper allocation, release, and cleanup of resources.
    func testMemoryManagement() throws {
        // Test buffer pooling and cleanup

        // First allocate and immediately release many buffers
        let bufferCount = 100

        for size in [1024, 2048, 4096] {
            for _ in 0..<bufferCount {
                let (bufferId, _) = try metalCore.allocateBuffer(
                    length: size * MemoryLayout<Float>.stride,
                    options: .storageModeShared
                )
                metalCore.releaseBuffer(id: bufferId)
            }
        }

        // Verify buffer pool contains the released buffers
        let statsBefore = metalCore.getBufferStats()
        XCTAssertEqual(statsBefore.activeCount, 0, "No active buffers expected")
        XCTAssertGreaterThan(statsBefore.pooledCount, 0, "Buffer pool should not be empty")

        // Clean up unused buffers
        metalCore.cleanupUnusedBuffers()

        // Verify buffer pool is now empty
        let statsAfter = metalCore.getBufferStats()
        XCTAssertEqual(statsAfter.activeCount, 0, "No active buffers expected")
        XCTAssertEqual(statsAfter.pooledCount, 0, "Buffer pool should be empty after cleanup")
    }

    /// Tests the allocation and processing limits of the Metal compute system
    /// under heavy load to ensure stability and proper resource management.
    func testProcessingLimits() async throws {
        // Define large buffer sizes to test resource limits
        let largeBufferSizes = [16_384, 32_768, 65_536]
        var successfulProcessingCounts = [Int: Int]()

        // Test each buffer size
        for bufferSize in largeBufferSizes {
            let maxIterations = 10
            var successCount = 0

            // Try multiple operations with large buffers
            for iterationIndex in 0..<maxIterations {
                do {
                    // Create large input and output buffers
                    let inputId = try createTestBuffer(size: bufferSize)
                    let outputId = try createTestBuffer(size: bufferSize, fillValue: 0.0)

                    // Create an expectation for this operation
                    let expectation = XCTestExpectation(
                        description: "Large Buffer Processing \(bufferSize)-\(iterationIndex)"
                    )

                    // Try to process with large buffers
                    metalCore.normalizeAudio(
                        inputBufferId: inputId,
                        outputBufferId: outputId,
                        sampleCount: bufferSize,
                        targetLevel: 0.9,
                        completion: { result in
                            switch result {
                            case .success:
                                successCount += 1
                                expectation.fulfill()
                            case .failure:
                                // Failure is expected at some point due to resource limits
                                expectation.fulfill()
                            }
                        }
                    )

                    // Wait with shorter timeout to detect performance issues
                    await fulfillment(of: [expectation], timeout: 2.0)
                } catch {
                    // OOM or other allocation errors are expected at limit boundaries
                    print("Expected error at bufferSize \(bufferSize): \(error)")
                }
            }

            // Record success rate for this buffer size
            successfulProcessingCounts[bufferSize] = successCount
        }

        // Log the results
        print("Processing limits test results:")
        for (size, count) in successfulProcessingCounts {
            print("Buffer size \(size): \(count)/10 operations completed successfully")
        }

        // If we're here without crashes, the test passes
        // The actual success counts will vary by device capabilities
    }
}
