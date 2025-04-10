import XCTest
import AVFoundation
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for basic initialization and configuration of the FFT processing node
final class FFTProcessorBasicTests: FFTProcessorBaseTests {

    // MARK: - Initialization Tests

    /// Tests basic initialization with default parameters
    func testInitialization() {
        // Create a new node with default parameters
        let node = FFTProcessingNode()

        // Verify default values
        XCTAssertEqual(node.name, "FFT Node", "Default name should be 'FFT Node'")
        XCTAssertEqual(node.fftSize, 1024, "Default FFT size should be 1024")
        XCTAssertEqual(node.windowFunction, .hann, "Default window function should be Hann")
        XCTAssertEqual(node.hopSize, 512, "Default hop size should be 512")
        XCTAssertFalse(node.zeroPadding, "Zero padding should be disabled by default")
        XCTAssertTrue(node.enabled, "Node should be enabled by default")
}

    /// Tests initialization with custom parameters
    func testInitializationWithCustomParameters() {
        // Create node with custom parameters
        let node = FFTProcessingNode(
            name: "Custom FFT",
            fftSize: 2048,
            windowFunction: .hamming,
            hopSize: 1024,
            zeroPadding: true,
            enabled: false
        )

        // Verify custom values
        XCTAssertEqual(node.name, "Custom FFT", "Custom name should be set correctly")
        XCTAssertEqual(node.fftSize, 2048, "Custom FFT size should be set correctly")
        XCTAssertEqual(node.windowFunction, .hamming, "Custom window function should be set correctly")
        XCTAssertEqual(node.hopSize, 1024, "Custom hop size should be set correctly")
        XCTAssertTrue(node.zeroPadding, "Zero padding should be enabled when set")
        XCTAssertFalse(node.enabled, "Node should be disabled when set")
}

    /// Tests that FFT size is normalized to a power of 2
    func testFFTSizeNormalization() {
        // Test with non-power-of-2 sizes
        let node1 = FFTProcessingNode(name: "Test 1", fftSize: 1000)
        let node2 = FFTProcessingNode(name: "Test 2", fftSize: 1500)
        let node3 = FFTProcessingNode(name: "Test 3", fftSize: 2500)

        // Verify sizes are normalized to next power of 2
        XCTAssertEqual(node1.fftSize, 1024, "Size 1000 should be normalized to 1024")
        XCTAssertEqual(node2.fftSize, 2048, "Size 1500 should be normalized to 2048")
        XCTAssertEqual(node3.fftSize, 4096, "Size 2500 should be normalized to 4096")
}

    // MARK: - Configuration Tests

    /// Tests configuration parameter updates
    func testConfigurationParameters() throws {
        // Create a node with initial parameters
        let node = FFTProcessingNode(name: "Config Test", fftSize: 1024)

        // Update parameters
        try node.setWindowFunction(.blackman)
        try node.setFFTSize(2048)
        try node.setHopSize(512)
        try node.setZeroPadding(true)

        // Verify parameter updates
        XCTAssertEqual(node.windowFunction, .blackman, "Window function should be updated")
        XCTAssertEqual(node.fftSize, 2048, "FFT size should be updated")
        XCTAssertEqual(node.hopSize, 512, "Hop size should be updated")
        XCTAssertTrue(node.zeroPadding, "Zero padding should be updated")
}

    /// Tests validation of invalid configurations
    func testInvalidConfiguration() throws {
        // Create a test node
        let node = FFTProcessingNode(name: "Invalid Config Test")

        // Test invalid FFT size (too small)
        XCTAssertThrowsError(try node.setFFTSize(8), "Too small FFT size should throw error")

        // Test invalid FFT size (too large)
        XCTAssertThrowsError(try node.setFFTSize(1_048_576), "Too large FFT size should throw error")

        // Test invalid hop size (less than 1)
        XCTAssertThrowsError(try node.setHopSize(0), "Zero hop size should throw error")

        // Test invalid hop size (larger than FFT size)
        XCTAssertThrowsError(try node.setHopSize(2048), "Hop size > FFT size should throw error")
}

    // MARK: - Window Function Tests

    /// Tests various window functions and their effect on processing
    func testWindowFunctions() async throws {
        // Generate a test signal (440 Hz sine wave)
        let testSignal = generateSineWave(
            frequency: 440.0,
            amplitude: 0.8,
            sampleCount: defaultFFTSize,
            sampleRate: sampleRate
        )

        // Test each window function
        let windowFunctions: [WindowFunction] = [.none, .hann, .hamming, .blackman]
        var spectrumResults: [[Float]] = []

        for windowFunction in windowFunctions {
            // Configure the FFT node with the window function
            try fftNode.setWindowFunction(windowFunction)

            // Process the signal
            let spectrum = try await processTestData(testSignal)
            spectrumResults.append(spectrum)

            // Verify that a peak is detected at the expected frequency
            let peakDetected = verifyFrequencyPeak(
                in: spectrum,
                frequency: 440.0,
                sampleRate: sampleRate,
                fftSize: defaultFFTSize
            )

            XCTAssertTrue(
                peakDetected,
                "Window function \(windowFunction) should detect 440 Hz peak"
            )
    }

        // Verify spectral leakage differences between window functions
        // No window should have the most leakage
        let noWindowSpectrum = spectrumResults[0]
        let hannWindowSpectrum = spectrumResults[1]

        // Find the 440 Hz peak bin
        let expectedBin = binIndexForFrequency(440.0, fftSize: defaultFFTSize, sampleRate: sampleRate)
        let binRange = max(0, expectedBin - 5)..<min(noWindowSpectrum.count, expectedBin + 6)

        // Calculate energy in adjacent bins (spectral leakage)
        var noWindowLeakage: Float = 0
        var hannWindowLeakage: Float = 0

        for binIndex in binRange where abs(binIndex - expectedBin) > 1 {
            noWindowLeakage += noWindowSpectrum[binIndex]
            hannWindowLeakage += hannWindowSpectrum[binIndex]
        }

        // No window should have more spectral leakage than Hann window
        XCTAssertGreaterThan(
            noWindowLeakage,
            hannWindowLeakage,
            "No window should have more spectral leakage than Hann window"
        )
    }
}
