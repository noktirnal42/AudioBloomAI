import XCTest
import Metal
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for initialization and buffer management in MetalComputeCore
@available(macOS 15.0, *)
final class MetalComputeSetupTests: MetalComputeBaseTests {

    // MARK: - Initialization Tests

    /// Tests that a Metal compute core instance can be properly initialized
    func testInitialization() throws {
        // Test that we can create a Metal compute core instance
        XCTAssertNotNil(metalCore, "Metal compute core should be initialized")

        // Verify buffer stats at initialization
        let stats = metalCore.getBufferStats()
        XCTAssertEqual(stats.activeCount, 0, "No active buffers at initialization")
        XCTAssertEqual(stats.pooledCount, 0, "No pooled buffers at initialization")
    }

    // MARK: - Buffer Management Tests

    /// Tests that buffers can be allocated with the correct size and options
    func testBufferAllocation() throws {
        // Allocate a buffer
        let bufferSize = 1024
        let (bufferId, buffer) = try metalCore.allocateBuffer(
            length: bufferSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )

        // Track for cleanup
        allocatedBuffers.append(bufferId)

        // Verify buffer properties
        XCTAssertNotNil(buffer, "Buffer should be created")
        XCTAssertEqual(buffer.length, bufferSize * MemoryLayout<Float>.stride, "Buffer size should match")

        // Verify buffer stats
        let stats = metalCore.getBufferStats()
        XCTAssertEqual(stats.activeCount, 1, "One active buffer")
    }

    /// Tests that buffer data can be updated and read correctly
    func testBufferUpdateAndRead() throws {
        // Allocate a buffer
        let bufferSize = 1024
        let (bufferId, _) = try metalCore.allocateBuffer(
            length: bufferSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )

        // Track for cleanup
        allocatedBuffers.append(bufferId)

        // Create test data
        let testData = [Float](repeating: 42.0, count: bufferSize)

        // Update buffer
        try testData.withUnsafeBytes { rawBufferPointer in
            try metalCore.updateBuffer(
                id: bufferId,
                from: rawBufferPointer.baseAddress!,
                length: bufferSize * MemoryLayout<Float>.stride
            )
        }

        // Read back data
        var readData = [Float](repeating: 0, count: bufferSize)
        try readData.withUnsafeMutableBytes { rawBufferPointer in
            try metalCore.readBuffer(
                id: bufferId,
                into: rawBufferPointer.baseAddress!,
                length: bufferSize * MemoryLayout<Float>.stride
            )
        }

        // Verify data
        for index in 0..<bufferSize {
            XCTAssertEqual(readData[index], 42.0, "Data should match at index \(index)")
        }
    }

    /// Tests that buffers are properly released and pooled for reuse
    func testBufferReleaseAndPooling() throws {
        // Allocate a buffer
        let bufferSize = 1024
        let (bufferId, _) = try metalCore.allocateBuffer(
            length: bufferSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )

        // Release the buffer (should go to pool)
        metalCore.releaseBuffer(id: bufferId)

        // Verify buffer stats
        let stats1 = metalCore.getBufferStats()
        XCTAssertEqual(stats1.activeCount, 0, "No active buffers after release")
        XCTAssertEqual(stats1.pooledCount, 1, "One pooled buffer after release")

        // Allocate a buffer of the same size (should reuse from pool)
        let (newBufferId, _) = try metalCore.allocateBuffer(
            length: bufferSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )

        // Track for cleanup
        allocatedBuffers.append(newBufferId)

        // Verify buffer stats after reuse
        let stats2 = metalCore.getBufferStats()
        XCTAssertEqual(stats2.activeCount, 1, "One active buffer after reuse")
        XCTAssertEqual(stats2.pooledCount, 0, "No pooled buffers after reuse")
    }

    /// Tests error handling for invalid buffer operations
    func testErrorHandling() throws {
        // Try to read from non-existent buffer
        XCTAssertThrowsError(try metalCore.readBuffer(
            id: 9999,
            into: UnsafeMutableRawPointer.allocate(byteCount: 4, alignment: 4),
            length: 4
        ), "Reading from non-existent buffer should throw")

        // Try to update non-existent buffer
        var data: Float = 42.0
        XCTAssertThrowsError(try withUnsafeBytes(of: &data) { bytes in
            try metalCore.updateBuffer(
                id: 9999,
                from: bytes.baseAddress!,
                length: 4
            )
        }, "Updating non-existent buffer should throw")

        // Create valid buffer but try to read beyond its bounds
        let bufferSize = 10
        let (bufferId, _) = try metalCore.allocateBuffer(
            length: bufferSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )
        allocatedBuffers.append(bufferId)

        let largerSize = (bufferSize + 10) * MemoryLayout<Float>.stride
        XCTAssertThrowsError(try metalCore.readBuffer(
            id: bufferId,
            into: UnsafeMutableRawPointer.allocate(
                byteCount: largerSize,
                alignment: 4
            ),
            length: largerSize
        ))
    }
}
