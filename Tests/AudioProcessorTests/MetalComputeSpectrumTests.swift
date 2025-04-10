import XCTest
import Metal
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for spectrum analysis and audio normalization functionality
@available(macOS 15.0, *)
final class MetalComputeSpectrumTests: MetalComputeBaseTests {

    // MARK: - Spectrum Analysis Tests

    /// Spectrum data setup information
    struct SpectrumSetup {
        let inputBufferId: UInt64
        let analysisOutputId: UInt64
        let frequencies: [Float]
        let amplitudes: [Float]
    }

    /// Tests basic functionality of the spectrum analysis operation
    func testSpectrumAnalysisBasicFunctionality() async throws {
        // Setup test parameters
        let fftSize = 2048
        let sampleRate: Float = 44100.0

        // Prepare spectrum data with known frequency components
        let setup = try prepareSpectrumAnalysisBuffers(
            fftSize: fftSize,
            sampleRate: sampleRate
        )

        // Run spectrum analysis
        await runSpectrumAnalysis(
            spectrumBufferId: setup.inputBufferId,
            outputBufferId: setup.analysisOutputId,
            fftSize: fftSize,
            sampleRate: sampleRate
        )

        // Verify analysis results
        try verifySpectrumAnalysis(analysisOutputId: setup.analysisOutputId)
    }

    /// Prepares test data for spectrum analysis with specific frequency components
    private func prepareSpectrumAnalysisBuffers(
        fftSize: Int,
        sampleRate: Float
    ) throws -> SpectrumSetup {
        // Define test frequencies
        let bassFreq: Float = 100.0 // Bass frequency
        let midFreq: Float = 1000.0 // Mid frequency
        let trebleFreq: Float = 8000.0 // Treble frequency

        let frequencies = [bassFreq, midFreq, trebleFreq]
        let amplitudes: [Float] = [0.7, 0.5, 0.3] // Corresponding amplitudes

        // Allocate buffers
        let inputBufferId = try createTestBuffer(size: fftSize * 2) // Input buffer with complex data
        let analysisOutputId = try createTestBuffer(size: 4) // Output for analysis results

        // Create complex spectrum data
        var complexData = [Float](repeating: 0.0, count: fftSize * 2)

        // Set spectrum peaks at our test frequencies
        for index in 0..<frequencies.count {
            let binIndex = Int(frequencies[index] * Float(fftSize) / sampleRate)
            complexData[binIndex * 2] = amplitudes[index] // Real component
        }

        // Update input buffer with spectrum data
        try complexData.withUnsafeBytes { bytes in
            try metalCore.updateBuffer(
                id: inputBufferId,
                from: bytes.baseAddress!,
                length: fftSize * 2 * MemoryLayout<Float>.stride
            )
        }

        return SpectrumSetup(
            inputBufferId: inputBufferId,
            analysisOutputId: analysisOutputId,
            frequencies: frequencies,
            amplitudes: amplitudes
        )
    }

    /// Runs spectrum analysis on the Metal compute system
    private func runSpectrumAnalysis(
        spectrumBufferId: UInt64,
        outputBufferId: UInt64,
        fftSize: Int,
        sampleRate: Float
    ) async {
        // Setup expectation
        let expectation = XCTestExpectation(description: "Spectrum Analysis Completion")

        // Perform spectrum analysis
        metalCore.analyzeSpectrum(
            spectrumBufferId: spectrumBufferId,
            outputBufferId: outputBufferId,
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
    }

    /// Verifies the spectrum analysis results
    private func verifySpectrumAnalysis(analysisOutputId: UInt64) throws {
        // Read results (bass, mid, treble, overall)
        let results = try readBufferData(bufferId: analysisOutputId, size: 4)

        // Verify each frequency band was detected with expected intensity
        XCTAssertGreaterThan(results[0], 0.3, "Bass level should be significant")
        XCTAssertGreaterThan(results[1], 0.2, "Mid level should be significant")
        XCTAssertGreaterThan(results[2], 0.1, "Treble level should be significant")

        // Verify relative levels (bass > mid > treble as we set in our input)
        XCTAssertGreaterThan(results[0], results[1], "Bass should be louder than mid")
        XCTAssertGreaterThan(results[1], results[2], "Mid should be louder than treble")
    }

    // MARK: - Audio Normalization Tests

    /// Tests audio normalization correctly scales audio levels
    func testAudioNormalization() async throws {
        // Create test audio data with varying amplitudes
        let sampleCount = 1024
        let testAudio = createTestAudioSignal(sampleCount: sampleCount, peakValue: 0.5)

        // Setup buffers for normalization test
        let buffers = try setupNormalizationBuffers(
            sampleCount: sampleCount,
            testAudio: testAudio
        )

        // Run normalization
        let targetLevel: Float = 0.9
        await runNormalization(
            inputBufferId: buffers.inputBufferId,
            outputBufferId: buffers.outputBufferId,
            sampleCount: sampleCount,
            targetLevel: targetLevel
        )

        // Verify normalization results
        try verifyNormalization(
            outputBufferId: buffers.outputBufferId,
            sampleCount: sampleCount,
            testAudio: testAudio,
            targetLevel: targetLevel
        )
    }

    /// Creates a test audio signal with a specific peak value
    private func createTestAudioSignal(sampleCount: Int, peakValue: Float) -> [Float] {
        var testAudio = [Float](repeating: 0.0, count: sampleCount)

        // Create a signal with the specified peak value
        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Float.pi * Float(sampleIndex) / 100.0
            testAudio[sampleIndex] = peakValue * sin(phase)
        }

        return testAudio
    }

    /// Buffer information for normalization tests
    struct NormalizationBuffers {
        let inputBufferId: UInt64
        let outputBufferId: UInt64
    }

    /// Sets up buffers for audio normalization test
    private func setupNormalizationBuffers(
        sampleCount: Int,
        testAudio: [Float]
    ) throws -> NormalizationBuffers {
        // Create input buffer
        let inputBufferId = try createTestBuffer(size: sampleCount)
        try testAudio.withUnsafeBytes { bytes in
            try metalCore.updateBuffer(
                id: inputBufferId,
                from: bytes.baseAddress!,
                length: sampleCount * MemoryLayout<Float>.stride
            )
        }

        // Create output buffer
        let outputBufferId = try createTestBuffer(size: sampleCount, fillValue: 0.0)

        return NormalizationBuffers(inputBufferId: inputBufferId, outputBufferId: outputBufferId)
    }

    /// Runs audio normalization on the Metal compute system
    private func runNormalization(
        inputBufferId: UInt64,
        outputBufferId: UInt64,
        sampleCount: Int,
        targetLevel: Float
    ) async {
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
    }

    /// Verifies the results of audio normalization
    private func verifyNormalization(
        outputBufferId: UInt64,
        sampleCount: Int,
        testAudio: [Float],
        targetLevel: Float
    ) throws {
        // Read normalized output
        let normalizedAudio = try readBufferData(bufferId: outputBufferId, size: sampleCount)

        // Find peak in original audio
        var inputPeak: Float = 0.0
        for sample in testAudio {
            inputPeak = max(inputPeak, abs(sample))
        }

        // Find peak in normalized audio
        var outputPeak: Float = 0.0
        for sample in normalizedAudio {
            outputPeak = max(outputPeak, abs(sample))
        }

        // Expected scale factor
        let expectedScaleFactor = targetLevel / inputPeak

        // Verify normalization was applied correctly
        XCTAssertEqual(outputPeak, targetLevel, accuracy: 0.01, "Peak should be normalized to target level")

        // Check a few samples to verify scaling was applied uniformly
        for sampleIndex in 0..<min(10, sampleCount) {
            let expectedSample = testAudio[sampleIndex] * expectedScaleFactor
            XCTAssertEqual(
                normalizedAudio[sampleIndex],
                expectedSample,
                accuracy: 0.001,
                "Sample at index \(sampleIndex) should be scaled correctly"
            )
        }
    }
}
