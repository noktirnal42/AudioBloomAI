// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import XCTest
import Metal
import Accelerate
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for Metal-accelerated audio processing functionality
@available(macOS 15.0, *)
final class MetalComputeAudioTests: XCTestCase {
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
    
    /// Generate a sine wave for testing
    /// - Parameters:
    ///   - frequency: Frequency in Hz
    ///   - sampleRate: Sample rate in Hz
    ///   - duration: Duration in seconds
    /// - Returns: Array of samples
    private func generateSineWave(
        frequency: Float,
        sampleRate: Float,
        duration: Float
    ) -> [Float] {
        let sampleCount = Int(sampleRate * duration)
        var samples = [Float](repeating: 0, count: sampleCount)
        
        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Float.pi * frequency * Float(sampleIndex) / sampleRate
            samples[sampleIndex] = sin(phase)
        }
        
        return samples
    }
    
    // Calculate next power of 2 helper function
    private func nextPowerOfTwo(_ value: Int) -> Int {
        return 1 << Int(log2(Double(value - 1)) + 1)
    }
    
    // MARK: - Audio Processing Tests
    
    func testFFTProcessing() async throws {
        // Create sine wave input for FFT
        let sampleRate: Float = 44100.0
        let frequency: Float = 1000.0 // 1kHz sine wave
        let duration: Float

