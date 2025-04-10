import XCTest
import Metal
@testable import AudioProcessor
@testable import AudioBloomCore

/// Test class for FFT and frequency domain processing using Metal compute
@available(macOS 15.0, *)
final class MetalComputeAudioTests: MetalComputeBaseTests {

    // MARK: - Types

    /// Parameters for FFT tests
    private struct FFTParams {
        let frequency: Float
        let sampleRate: Float
        let duration: Float
    }

    /// Information about a generated sine wave
    private struct SineWaveInfo {
        let data: [Float]
        let fftSize: Int
    }

    /// Structure to hold FFT operation buffers
    private struct FFTBuffers {
        let input: UInt64
        let output: UInt64
    }

    /// Structure to hold frequency filtering buffers
    private struct FilterBuffers {
        let input: UInt64
        let filter: UInt64
        let output: UInt64
    }

    /// Result of peak frequency detection
    private struct PeakFrequency {
        let binIndex: Int
        let magnitude: Float
    }

    // MARK: - FFT Tests

    /// Tests that FFT processing correctly identifies frequency components
    func testFFTProcessing() async throws {
        // Setup FFT test parameters
        let params = FFTParams(
            frequency: 1000.0,  // 1kHz sine wave
            sampleRate: 44100.0, // 44.1kHz sample rate
            duration: 0.05       // 50ms
        )

        // Create input buffer with sine wave
        let waveInfo = generatePaddedSineWave(
            frequency: params.frequency,
            sampleRate: params.sampleRate,
            duration: params.duration
        )

        // Setup buffers for FFT test
        let buffers = try setupFFTBuffers(
            sineWave: waveInfo.data,
            fftSize: waveInfo.fftSize
        )

        // Run FFT operation
        await runFFTOperation(
            inputBufferId: buffers.input,
            outputBufferId: buffers.output,
            fftSize: waveInfo.fftSize
        )

        // Verify FFT results
        try verifyFFTResults(
            outputBufferId: buffers.output,
            fftSize: waveInfo.fftSize,
            frequency: params.frequency,
            sampleRate: params.sampleRate
        )
    }

    /// Generates a sine wave and pads it to a power of 2 size for FFT
    private func generatePaddedSineWave(
        frequency: Float,
        sampleRate: Float,
        duration: Float
    ) -> SineWaveInfo {
        var sineWave = generateSineWave(frequency: frequency, sampleRate: sampleRate, duration: duration)
        let fftSize = nextPowerOfTwo(sineWave.count)

        // Pad to power of 2 if needed
        if sineWave.count < fftSize {
            sineWave.append(contentsOf: [Float](repeating: 0, count: fftSize - sineWave.count))
        }

        return SineWaveInfo(data: sineWave, fftSize: fftSize)
    }

    /// Sets up input and output buffers for FFT tests
    private func setupFFTBuffers(sineWave: [Float], fftSize: Int) throws -> FFTBuffers {
        // Create input buffer
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

        return FFTBuffers(input: inputBufferId, output: outputBufferId)
    }

    /// Runs the FFT operation on the Metal compute system
    /// - Parameters:
    ///   - inputBufferId: ID of the input buffer
    ///   - outputBufferId: ID of the output buffer
    ///   - fftSize: Size of the FFT operation
    private func runFFTOperation(
        inputBufferId: UInt64,
        outputBufferId: UInt64,
        fftSize: Int
    ) async {
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
    }

    /// Verifies the results of an FFT operation
    /// - Parameters:
    ///   - outputBufferId: ID of the buffer containing FFT results
    ///   - fftSize: Size of the FFT operation
    ///   - frequency: Expected frequency to detect
    ///   - sampleRate: Sample rate of the audio data
    private func verifyFFTResults(
        outputBufferId: UInt64,
        fftSize: Int,
        frequency: Float,
        sampleRate: Float
    ) throws {
        // Read output buffer (complex numbers: interleaved real and imaginary parts)
        let fftOutput = try readBufferData(bufferId: outputBufferId, size: fftSize)

        // Calculate which bin should have our frequency
        let binWidth = Float(sampleRate) / Float(fftSize)
        let expectedBin = Int(frequency / binWidth)

        // Find the bin with maximum magnitude
        let peak = findPeakFrequency(in: fftOutput, fftSize: fftSize)

        // Due to FFT binning and windowing effects, the peak may not be exactly at the expected bin
        // We allow for a small margin of error
        let binError = 2
        XCTAssertTrue(
            abs(peak.binIndex - expectedBin) <= binError,
            "FFT should detect frequency at correct bin (expected: \(expectedBin), got: \(peak.binIndex))"
        )
    }

    /// Finds the frequency bin with the highest magnitude in FFT output
    /// - Parameters:
    ///   - fftOutput: Array containing FFT output data
    ///   - fftSize: Size of the FFT operation
    /// - Returns: The peak frequency information
    private func findPeakFrequency(in fftOutput: [Float], fftSize: Int) -> PeakFrequency {
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

        return PeakFrequency(binIndex: maxBin, magnitude: maxMagnitude)
    }

    // MARK: - Frequency Filtering Tests

    /// Tests that frequency domain filtering correctly attenuates frequencies above cutoff
    func testFrequencyDomainFiltering() async throws {
        // Create test parameters
        let fftSize = 1024
        let cutoffBin = fftSize / 4

        // Setup buffers
        let buffers = try setupFilterBuffers(
            fftSize: fftSize,
            cutoffBin: cutoffBin
        )

        // Run filtering operation
        await runFilterOperation(
            inputBufferId: buffers.input,
            filterBufferId: buffers.filter,
            outputBufferId: buffers.output,
            fftSize: fftSize
        )

        // Verify filter results
        try verifyFilterResults(
            outputBufferId: buffers.output,
            fftSize: fftSize,
            cutoffBin: cutoffBin
        )
    }

    /// Sets up buffers for frequency domain filtering tests
    /// - Parameters:
    ///   - fftSize: Size of the FFT data
    ///   - cutoffBin: Frequency bin cutoff index
    /// - Returns: The filter buffer information
    private func setupFilterBuffers(
        fftSize: Int,
        cutoffBin: Int
    ) throws -> FilterBuffers {
        // Create input, filter, and output buffers
        let inputBufferId = try createTestBuffer(size: fftSize * 2)
        let filterBufferId = try createTestBuffer(size: fftSize * 2)
        let outputBufferId = try createTestBuffer(size: fftSize * 2, fillValue: 0.0)

        // Create a low-pass filter (pass frequencies below cutoff)
        var filter = [Float](repeating: 0, count: fftSize * 2)

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

        return FilterBuffers(
            input: inputBufferId,
            filter: filterBufferId,
            output: outputBufferId
        )
    }

    /// Runs the frequency filter operation
    /// - Parameters:
    ///   - inputBufferId: ID of the input buffer
    ///   - filterBufferId: ID of the filter buffer
    ///   - outputBufferId: ID of the output buffer
    ///   - fftSize: Size of the FFT data
    private func runFilterOperation(
        inputBufferId: UInt64,
        filterBufferId: UInt64,
        outputBufferId: UInt64,
        fftSize: Int
    ) async {
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
    }

    /// Verifies the results of the frequency filter operation
    /// - Parameters:
    ///   - outputBufferId: ID of the output buffer
    ///   - fftSize: Size of the FFT data
    ///   - cutoffBin: Frequency bin cutoff index
    private func verifyFilterResults(
        outputBufferId: UInt64,
        fftSize: Int,
        cutoffBin: Int
    ) throws {
        // Read output buffer
        let outputData = try readBufferData(bufferId: outputBufferId, size: fftSize * 2)

        // Verify filter was applied correctly
        // Frequencies below cutoff should be preserved, those above should be attenuated
        for binIndex in 0..<(fftSize / 2) {
            let realOut = outputData[binIndex * 2]
            let imagOut = outputData[binIndex * 2 + 1]

            if binIndex < cutoffBin {
                // For bins below cutoff, output should match input
                XCTAssertEqual(
                    realOut,
                    Float(binIndex * 2),
                    accuracy: 0.001,
                    "Low frequency real component should be preserved"
                )
                XCTAssertEqual(
                    imagOut,
                    Float(binIndex * 2 + 1),
                    accuracy: 0.001,
                    "Low frequency imaginary component should be preserved"
                )
            } else {
                // For bins above cutoff, output should be attenuated
                XCTAssertEqual(
                    realOut,
                    0.0,
                    accuracy: 0.001,
                    "High frequency real component should be attenuated"
                )
                XCTAssertEqual(
                    imagOut,
                    0.0,
                    accuracy: 0.001,
                    "High frequency imaginary component should be attenuated"
                )
            }
        }
    }
}
