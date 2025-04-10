import XCTest
import Metal
import Accelerate
@testable import AudioProcessor
@testable import AudioBloomCore

/// Base class for Metal compute tests with shared setup/teardown and helper methods
@available(macOS 15.0, *)
class MetalComputeBaseTests: XCTestCase {

    // MARK: - Test Variables

    /// Metal compute core instance under test
    var metalCore: MetalComputeCore!

    /// Buffer IDs to track and clean up
    var allocatedBuffers: [UInt64] = []

    /// Verification data
    var inputData: [Float] = []
    var outputData: [Float] = []

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
        inputData = []
        outputData = []
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
    func createTestBuffer(size: Int, fillValue: Float? = nil) throws -> UInt64 {
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
    func readBufferData(bufferId: UInt64, size: Int) throws -> [Float] {
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
    func generateSineWave(frequency: Float, sampleRate: Float, duration: Float) -> [Float] {
        let sampleCount = Int(sampleRate * duration)
        var samples = [Float](repeating: 0, count: sampleCount)

        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Float.pi * frequency * Float(sampleIndex) / sampleRate
            samples[sampleIndex] = sin(phase)
        }

        return samples
    }

    /// Gets the next power of two greater than or equal to the given number
    /// - Parameter number: The input number
    /// - Returns: The next power of two
    func nextPowerOfTwo(_ number: Int) -> Int {
        var result = 1
        while result < number {
            result *= 2
        }
        return result
    }
}
