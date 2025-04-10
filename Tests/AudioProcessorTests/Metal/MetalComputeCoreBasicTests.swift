// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import XCTest
import Metal
import Accelerate
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for basic Metal compute core functionality
@available(macOS 15.0, *)
final class MetalComputeCoreBasicTests: XCTestCase {
    // MARK: - Test Variables
    
    /// Metal compute core instance under test
    private var metalCore: MetalComputeCore!
    
    /// Buffer IDs to track and clean up
    private var allocatedBuffers: [UInt64] = []
    
    // MARK: - Setup and Teardown
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Skip Metal tests if we're on a device that doesn't support Metal
        try XCTSkipIf(
            MTLCreateSystemDefaultDevice() == nil,
            "Skipping Metal tests - no Metal device available"
        )
        
        // Initialize Metal compute core
        metalCore = try MetalComputeCore(maxConcurrentOperations: 3)
        
        // Reset tracking arrays
        allocatedBuffers = []
    }
    
    override func tearDown() async throws {
        // Release any allocated buffers
        for bufferId in allocatedBuffers {
            metalCore.releaseBuffer(id: bufferId)
        }
        allocatedBuffers = []
        
        // Release Metal compute core
        metalCore = nil
        
        try await super.tearDown()
    }
    
    // MARK: - Helper Methods
    
    /// Create a buffer with test data
    /// - Parameters:
    ///   - size: Size of the buffer in elements
    ///   - fillValue: Value to fill the buffer with (nil for sequential values)
    /// - Returns: Buffer ID
    private func createTestBuffer(size: Int, fillValue: Float? = nil) throws -> UInt64 {
        // Allocate buffer
        let (bufferId, _) = try metalCore.allocateBuffer(
            length: size * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )
        
        // Track for cleanup
        allocatedBuffers.append(bufferId)
        
        // Create test data
        var testData = [Float](repeating: 0, count: size)
        
        if let value = fillValue {
            // Fill with constant value
            for index in 0..<size {
                testData[index] = value
            }
        } else {
            // Fill with sequential values
            for index in 0..<size {
                testData[index] = Float(index)
            }
        }
        
        // Update buffer with test data
        try testData.withUnsafeBytes { rawBufferPointer in
            try metalCore.updateBuffer(
                id: bufferId,
                from: rawBufferPointer.baseAddress!,
                length: size * MemoryLayout<Float>.stride
            )
        }
        
        return bufferId
    }
    
    /// Read float data from a buffer
    /// - Parameters:
    ///   - bufferId: Buffer ID
    ///   - size: Number of elements to read
    /// - Returns: Array of float values
    private func readBufferData(bufferId: UInt64, size: Int) throws -> [Float] {
        var result = [Float](repeating: 0, count: size)
        
        try result.withUnsafeMutableBytes { rawBufferPointer in
            try metalCore.readBuffer(
                id: bufferId,
                into: rawBufferPointer.baseAddress!,
                length: size * MemoryLayout<Float>.stride
            )
        }
        
        return result
    }
    
    // MARK: - Core Functionality Tests
    
    func testInitialization() throws {
        // Test that we can create a Metal compute core instance
        XCTAssertNotNil(metalCore, "Metal compute core should be initialized")
        
        // Verify buffer stats at initialization
        let stats = metalCore.getBufferStats()
        XCTAssertEqual(stats.activeCount, 0, "No active buffers at initialization")
        XCTAssertEqual(stats.pooledCount, 0, "No pooled buffers at initialization")
    }
    
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
        XCTAssertEqual(
            buffer.length,
            bufferSize * MemoryLayout<Float>.stride,
            "Buffer size should match"
        )
        
        // Verify buffer stats
        let stats = metalCore.getBufferStats()
        XCTAssertEqual(stats.activeCount, 1, "One active buffer")
    }
    
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
    
    func testErrorHandling() throws {
        // Test error handling for invalid buffer operations
        
        // Try to read from non-existent buffer
        XCTAssertThrowsError(
            try metalCore.readBuffer(
                id: 9999,
                into: UnsafeMutableRawPointer.allocate(byteCount: 4, alignment: 4),
                length: 4
            ),
            "Reading from non-existent buffer should throw"
        )
        
        // Try to update non-existent buffer
        var data: Float = 42.0
        XCTAssertThrowsError(
            try withUnsafeBytes(of: &data) { bytes in
                try metalCore.updateBuffer(
                    id: 9999,
                    from: bytes.baseAddress!,
                    length: 4
                )
            },
            "Updating non-existent buffer should throw"
        )
        
        // Create valid buffer but try to read beyond its bounds
        let bufferSize = 10
        let (bufferId, _) = try metalCore.allocateBuffer(
            length: bufferSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )
        allocatedBuffers.append(bufferId)
        
        let bufferSizeExtended = bufferSize + 10
        XCTAssertThrowsError(
            try metalCore.readBuffer(
                id: bufferId,
                into: UnsafeMutableRawPointer.allocate(
                    byteCount: bufferSizeExtended * MemoryLayout<Float>.stride,
                    alignment: 4
                ),
                length: bufferSizeExtended * MemoryLayout<Float>.stride
            ),
            "Reading beyond buffer bounds should throw"
        )
    }
}

