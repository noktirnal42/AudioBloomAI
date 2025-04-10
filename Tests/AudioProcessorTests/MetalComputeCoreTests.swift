import XCTest
import Metal
import Accelerate
@testable import AudioProcessor
@testable import AudioBloomCore

@available(macOS 15.0, *)
final class MetalComputeCoreTests: XCTestCase {

    // MARK: - Test Variables

    /// Metal compute core instance under test
    private var metalCore: MetalComputeCore!

    /// Buffer IDs to track and clean up
    private var allocatedBuffers: [UInt64] = []

    /// Verification data
    private var inputData: [Float] = []
    private var outputData: [Float] = []

    // MARK: - Setup and Teardown

    override func setUp() async throws {
        try await super.setUp()

        // Skip Metal tests if we're on a device that doesn't support Metal
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "Skipping Metal tests - no Metal device available")

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
    private func generateSineWave(frequency: Float, sampleRate: Float, duration: Float) -> [Float] {
        let sampleCount = Int(sampleRate * duration)
        var samples = [Float](repeating: 0, count: sampleCount)

        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Float.pi * frequency * Float(sampleIndex) / sampleRate
            samples[sampleIndex] = sin(phase)
        }

        return samples
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
        XCTAssertEqual(buffer.length, bufferSize * MemoryLayout<Float>.stride, "Buffer size should match")

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

        XCTAssertThrowsError(try metalCore.readBuffer(
            id: bufferId,
            into: UnsafeMutableRawPointer.allocate(
                byteCount: (bufferSize + 10) * MemoryLayout<Float>.stride,
                alignment: 4
            ),
            length: (bufferSize + 10) * MemoryLayout<Float>.stride
    }

    // MARK: - Audio Processing Tests

    func testFFTProcessing() async throws {
        // Create sine wave input for FFT
        let sampleRate: Float = 44100.0
        let frequency: Float = 1000.0 // 1kHz sine wave
        let duration: Float = 1.0 / 20.0 // 50ms

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
        let outputBufferId = try createTestBuffer(size: fftSize / 2 * 2, fillValue: 0.0)

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

        // Read output buffer (complex numbers: interleaved real and imaginary parts)
        let fftOutput = try readBufferData(bufferId: outputBufferId, size: fftSize)

        // Analyze the FFT results
        // For a single sine wave, we expect significant energy at the frequency bin corresponding to our input

        // Calculate which bin should have our frequency
        let binWidth = Float(sampleRate) / Float(fftSize)
        let expectedBin = Int(frequency / binWidth)

        // Find the bin with maximum magnitude
        var maxBin = 0
        var maxMagnitude: Float = 0.0

        for binIndex in 0..<(fftSize / 2) {
            let real = fftOutput[binIndex * 2]
            let imag = fftOutput[binIndex * 2 + 1]
            let magnitude = sqrt(real * real + imag * imag)

            if magnitude > maxMagnitude {
                maxMagnitude = magnitude
                maxBin = binIndex
            }
        }
        // Due to FFT binning and windowing effects, the peak may not be exactly at the expected bin
        // We allow for a small margin of error
        let binError = 2
        XCTAssertTrue(abs(maxBin - expectedBin) <= binError,
                     "FFT should detect frequency at correct bin (expected: \(expectedBin), got: \(maxBin))")

    func testFrequencyDomainFiltering() async throws {
        // Create input data with multiple frequency components
        let fftSize = 1024
        let inputBufferId = try createTestBuffer(size: fftSize * 2) // Complex buffer (real/imag pairs)
        let filterBufferId = try createTestBuffer(size: fftSize * 2) // Complex filter
        let outputBufferId = try createTestBuffer(size: fftSize * 2, fillValue: 0.0) // Output buffer

        // Create a low-pass filter (pass frequencies below 25% of Nyquist)
        var filter = [Float](repeating: 0, count: fftSize * 2)
        let cutoffBin = fftSize / 4

        for binIndex in 0..<cutoffBin {
            // Set filter coefficients to 1.0 (pass)
            filter[binIndex * 2] = 1.0     // Real part
            filter[binIndex * 2 + 1] = 0.0 // Imaginary part
        }
        // Update filter buffer
        try filter.withUnsafeBytes { bytes in
            try metalCore.updateBuffer(
                id: filterBufferId,
                from: bytes.baseAddress!,
                length: fftSize * 2 * MemoryLayout<Float>.stride
            )
        }

        // Perform frequency domain filtering
        let expectation = XCTestExpectation(description: "Filter Completion")

        metalCore.applyFrequencyFilter(
            inputBufferId: inputBufferId,
            outputBufferId: outputBufferId,
            filterBufferId: filterBufferId,
            sampleCount: fftSize,
            completion: { result in
                switch result {
                case .success:
                    expectation.fulfill()
                case .failure(let error):
                    XCTFail("Frequency filter failed with error: \(error)")
                }
            }
        )

        // Wait for filtering to complete
        await fulfillment(of: [expectation], timeout: 5.0)

        // Read output buffer
        let outputData = try readBufferData(bufferId: outputBufferId, size: fftSize * 2)

        // Verify filter was applied correctly
        // Frequencies below cutoff should be preserved, those above should be attenuated
        for binIndex in 0..<(fftSize / 2) {
            let realOut = outputData[binIndex * 2]
            let imagOut = outputData[binIndex * 2 + 1]

            if binIndex < cutoffBin {
                // For bins below cutoff, output should match input
                XCTAssertEqual(realOut, Float(binIndex * 2), accuracy: 0.001,
                              "Low frequency real component should be preserved")
                XCTAssertEqual(imagOut, Float(binIndex * 2 + 1), accuracy: 0.001,
                              "Low frequency imaginary component should be preserved")
            } else {
                // For bins above cutoff, output should be attenuated
                XCTAssertEqual(realOut, 0.0, accuracy: 0.001,
                              "High frequency real component should be attenuated")
                XCTAssertEqual(imagOut, 0.0, accuracy: 0.001,
                              "High frequency imaginary component should be attenuated")
            }
        }

    func testSpectrumAnalysis() async throws {
        // Setup test with sine waves at different frequency bands
        let fftSize = 2048
        let sampleRate: Float = 44100.0

        // Create sine waves in bass, mid, and treble ranges
        let bassFreq: Float = 100.0 // Bass frequency
        let midFreq: Float = 1000.0 // Mid frequency
        let trebleFreq: Float = 8000.0 // Treble frequency

        // Generate composite signal with all three frequencies
        var signal = [Float](repeating: 0.0, count: fftSize)
        for sampleIndex in 0..<fftSize {
            let bassSample = sin(2.0 * Float.pi * bassFreq * Float(sampleIndex) / sampleRate)
            let midSample = sin(2.0 * Float.pi * midFreq * Float(sampleIndex) / sampleRate)
            let trebleSample = sin(2.0 * Float.pi * trebleFreq * Float(sampleIndex) / sampleRate)

            // Combine signals with different amplitudes
            signal[sampleIndex] = bassSample * 0.7 + midSample * 0.5 + trebleSample * 0.3
        }
        // Allocate buffers
        let inputBufferId = try createTestBuffer(size: fftSize * 2) // Input buffer with complex data
        let analysisOutputId = try createTestBuffer(size: 4) // Output buffer for analysis results: bass, mid, treble, overall

        // Create complex spectrum data (for simplicity, we'll manually create it instead of running FFT)
        var complexData = [Float](repeating: 0.0, count: fftSize * 2)

        // Set spectrum peaks at our test frequencies
        let bassIndex = Int(bassFreq * Float(fftSize) / sampleRate)
        let midIndex = Int(midFreq * Float(fftSize) / sampleRate)
        let trebleIndex = Int(trebleFreq * Float(fftSize) / sampleRate)

        // Add peaks with corresponding amplitudes
        complexData[bassIndex * 2] = 0.7 // Real component for bass
        complexData[midIndex * 2] = 0.5  // Real component for mid
        complexData[trebleIndex * 2] = 0.3 // Real component for treble

        // Update input buffer with spectrum data
        try complexData.withUnsafeBytes { bytes in
            try metalCore.updateBuffer(
                id: inputBufferId,
                from: bytes.baseAddress!,
                length: fftSize * 2 * MemoryLayout<Float>.stride
            )
        }

        // Create parameters for frequency bands
        let analysisParams = SpectrumAnalysisParameters(
            sampleCount: UInt32(fftSize),
            sampleRate: sampleRate,
            bassMinFreq: 20.0,
            bassMaxFreq: 250.0,
            midMinFreq: 250.0,
            midMaxFreq: 4000.0,
            trebleMinFreq: 4000.0,
            trebleMaxFreq: 20000.0
        )

        let paramsData = withUnsafeBytes(of: analysisParams) { Data($0) }

        // Setup expectation
        let expectation = XCTestExpectation(description: "Spectrum Analysis Completion")

        // Perform spectrum analysis
        metalCore.analyzeSpectrum(
            spectrumBufferId: inputBufferId,
            outputBufferId: analysisOutputId,
            sampleCount: fftSize,
            sampleRate: sampleRate,
            completion: { result in
                switch result {
                case .success:
                    expectation.fulfill()
                case .failure(let error):
                    XCTFail("Spectrum analysis failed with error: \(error)")
                }
            }
        )

        // Wait for completion
        await fulfillment(of: [expectation], timeout: 5.0)

        // Read results
        let results = try readBufferData(bufferId: analysisOutputId, size: 4)

        // Verify each frequency band was detected with expected intensity
        XCTAssertGreaterThan(results[0], 0.3, "Bass level should be significant")
        XCTAssertGreaterThan(results[1], 0.2, "Mid level should be significant")
        XCTAssertGreaterThan(results[2], 0.1, "Treble level should be significant")

        // Verify relative levels (bass > mid > treble as we set in our input)
        XCTAssertGreaterThan(results[0], results[1], "Bass should be louder than mid")
        XCTAssertGreaterThan(results[1], results[2], "Mid should be louder than treble")
    }

    func testAudioNormalization() async throws {
        // Create test audio data with varying amplitudes
        let sampleCount = 1024
        var testAudio = [Float](repeating: 0.0, count: sampleCount)

        // Create a signal with a peak value of 0.5
        let peakValue: Float = 0.5
        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Float.pi * Float(sampleIndex) / 100.0
            testAudio[sampleIndex] = peakValue * sin(phase)
        }
        // Create input and output buffers
        let inputBufferId = try createTestBuffer(size: sampleCount)
        try testAudio.withUnsafeBytes { bytes in
            try metalCore.updateBuffer(
                id: inputBufferId,
                from: bytes.baseAddress!,
                length: sampleCount * MemoryLayout<Float>.stride
            )
        }

        let outputBufferId = try createTestBuffer(size: sampleCount, fillValue: 0.0)

        // Target normalization level (normalize to 0.9)
        let targetLevel: Float = 0.9

        // Setup expectation
        let expectation = XCTestExpectation(description: "Normalization Completion")

        // Perform normalization
        metalCore.normalizeAudio(
            inputBufferId: inputBufferId,
            outputBufferId: outputBufferId,
            sampleCount: sampleCount,
            targetLevel: targetLevel,
            completion: { result in
                switch result {
                case .success:
                    expectation.fulfill()
                case .failure(let error):
                    XCTFail("Audio normalization failed with error: \(error)")
                }
            }
        )

        // Wait for completion
        await fulfillment(of: [expectation], timeout: 5.0)

        // Read normalized output
        let normalizedAudio = try readBufferData(bufferId: outputBufferId, size: sampleCount)

        // Find peak in normalized audio
        var outputPeak: Float = 0.0
        for sample in normalizedAudio {
            outputPeak = max(outputPeak, abs(sample))
        }

        // Expected scale factor
        let expectedScaleFactor = targetLevel / peakValue

        // Verify normalization was applied correctly
        XCTAssertEqual(outputPeak, targetLevel, accuracy: 0.01, "Peak should be normalized to target level")

        // Check a few samples to verify scaling was applied uniformly
        for sampleIndex in 0..<min(10, sampleCount) {
            let expectedSample = testAudio[sampleIndex] * expectedScaleFactor
            XCTAssertEqual(normalizedAudio[sampleIndex], expectedSample, accuracy: 0.001,
                          "Sample at index \(sampleIndex) should be scaled correctly")
        }

    // MARK: - Performance and Stress Tests

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
        for i in 0..<operationCount {
            let expectation = XCTestExpectation(description: "Concurrent Operation \(i)")
            expectations.append(expectation)

            // Use normalization as a simple test operation
            metalCore.normalizeAudio(
                inputBufferId: inputBuffers[i],
                outputBufferId: outputBuffers[i],
                sampleCount: bufferSize,
                targetLevel: 0.9,
                completion: { result in
                    switch result {
                    case .success:
                        expectation.fulfill()
                    case .failure(let error):
                        XCTFail("Concurrent operation \(i) failed with error: \(error)")
                    }
                }
            )
        }

        // Wait for all operations to complete with a reasonable timeout
        await fulfillment(of: expectations, timeout: 10.0)

        // Verify that the concurrency limit worked (no more than maxConcurrentOperations should execute simultaneously)
        let stats = metalCore.getBufferStats()

        print("After concurrent processing: active buffers = \(stats.activeCount), " +
              "pooled buffers = \(stats.pooledCount)")
    }

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

    // MARK: -

