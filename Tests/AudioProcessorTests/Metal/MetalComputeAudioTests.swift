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
        let duration: Float = 0.05 // 50ms
        
        // Generate sine wave and ensure it's a power of 2 in length
        var sineWave = generateSineWave(frequency: frequency, sampleRate: sampleRate, duration: duration)
        let fftSize = nextPowerOfTwo(sineWave.count)
        
        // Pad to power of 2 if needed
        if sineWave.count < fftSize {
            sineWave.append(contentsOf: [Float](repeating: 0, count: fftSize - sineWave.count))
        }
        
        // Create input and output buffers
        let inputBufferId = try createTestBuffer(size: fftSize)
        try sineWave.withUnsafeBytes { bytes in
            try metalCore.updateBuffer(
                id: inputBufferId,
                from: bytes.baseAddress!,
                length: fftSize * MemoryLayout<Float>.stride
            )
        }
        
        // Create output buffer for complex FFT results (each complex number has 2 float components)
        let outputBufferId = try createTestBuffer(size: fftSize, fillValue: 0.0)
        
        // Perform FFT
        let expectation = XCTestExpectation(description: "FFT Completion")
        
        metalCore.performFFT(
            inputBufferId: inputBufferId,
            outputBufferId: outputBufferId,
            sampleCount: fftSize,
            inverse: false
        ) { result in
            switch result {
            case .success:
                expectation.fulfill()
            case .failure(let error):
                XCTFail("FFT failed with error: \(error)")
            }
        }
        
        // Wait for FFT to complete
        await fulfillment(of: [expectation], timeout: 5.0)
        
        // Read output buffer
        let fftOutput = try readBufferData(bufferId: outputBufferId, size: fftSize)
        
        // Analyze the FFT results
        // For a sine wave, we expect significant energy at the frequency bin corresponding to our input
        
        // Calculate which bin should have our frequency
        let binWidth = Float(sampleRate) / Float(fftSize)
        let expectedBin = Int(frequency / binWidth)
        
        // Find the bin with maximum magnitude
        var maxBin = 0
        var maxMagnitude: Float = 0.0
        
        // Process the first half of the FFT result (due to Nyquist)
        for binIndex in 0..<(fftSize / 2) {
            // For a real FFT, results are stored as complex values
            let realValue = fftOutput[binIndex * 2]
            let imagValue = fftOutput[binIndex * 2 + 1]
            let magnitude = sqrt(realValue * realValue + imagValue * imagValue)
            
            if magnitude > maxMagnitude {
                maxMagnitude = magnitude
                maxBin = binIndex
            }
        }
        
        // Due to FFT binning and windowing effects, the peak may not be exactly at the expected bin
        // We allow for a small margin of error
        let binError = 2
        XCTAssertTrue(
            abs(maxBin - expectedBin) <= binError,
            "FFT should detect frequency at correct bin (expected: \(expectedBin), got: \(maxBin))"
        )
    }
}

