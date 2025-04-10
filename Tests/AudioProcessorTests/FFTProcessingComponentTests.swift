import XCTest
import AVFoundation
import Accelerate
import Metal
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for FFT processing components and functionality
final class FFTProcessingComponentTests: XCTestCase {
    // MARK: - Test Properties

    /// Audio pipeline under test
    private var audioPipeline: AudioPipelineCore!

    /// FFT processing node under test
    private var fftNode: FFTProcessingNode!

    // MARK: - Test Lifecycle

    override func setUp() async throws {
        try await super.setUp()
        audioPipeline = try AudioPipelineCore(
            configuration: AudioPipelineTestHelpers.createDefaultConfig()
        )

        // Initialize FFT node with default settings
        fftNode = FFTProcessingNode(
            name: "Test FFT Node",
            fftSize: 2048,
            windowFunction: .hann
        )

        // Add node to pipeline
        try audioPipeline.addNode(fftNode, connections: [])
    }

    override func tearDown() async throws {
        audioPipeline = nil
        fftNode = nil
        try await super.tearDown()
    }

    // MARK: - Component Tests

    /// Tests basic component initialization
    /// - Verifies that:
    ///   - Component is created with proper defaults
    ///   - Input and output requirements are set correctly
    ///   - FFT size is normalized to power of 2
    func testComponentInitialization() throws {
        // Test with default initialization
        let component = FFTProcessingNode()
        XCTAssertNotNil(component)
        XCTAssertEqual(component.name, "FFT Processor")

        // Verify FFT size is normalized to power of 2
        let oddSizedComponent = FFTProcessingNode(fftSize: 1000)
        XCTAssertTrue(
            [512, 1024, 2048].contains(oddSizedComponent.outputCapabilities.bufferSize.min),
            "FFT size should be normalized to nearest power of 2"
        )
    }

    /// Tests window function application
    /// - Verifies that:
    ///   - Different window functions produce expected spectral characteristics
    ///   - Window function selection properly affects output
    func testWindowFunctions() throws {
        // Create test buffers
        let inputBuffer = try audioPipeline.allocateBuffer(
            size: 2048 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        let outputBuffer = try audioPipeline.allocateBuffer(
            size: 1024 * MemoryLayout<Float>.stride,
            type: .cpu
        )

        defer {
            audioPipeline.releaseBuffer(id: inputBuffer)
            audioPipeline.releaseBuffer(id: outputBuffer)
        }

        // Generate test signal
        let testSignal = AudioPipelineTestHelpers.generateSineWave(
            frequency: 1000,
            duration: 0.1,
            amplitude: 0.5
        )

        // Compare results with different windows
        let windows: [(FFTWindowFunction, String)] = [
            (.hann, "Hann"),
            (.hamming, "Hamming"),
            (.blackman, "Blackman"),
            (.none, "Rectangular")
        ]

        var results: [String: Float] = [:]

        for (window, name) in windows {
            try fftNode.configure(parameters: ["windowFunction": window])

            try testSignal.withUnsafeBytes { bytes in
                try audioPipeline.updateBuffer(
                    id: inputBuffer,
                    data: bytes.baseAddress!,
                    size: testSignal.count * MemoryLayout<Float>.stride,
                    options: [.waitForCompletion]
                )
            }

            _ = try audioPipeline.process(
                inputBuffers: [inputBuffer],
                outputBuffers: [outputBuffer],
                context: audioPipeline
            )

            let spectrum = try AudioPipelineTestHelpers.getBufferContents(
                outputBuffer,
                pipeline: audioPipeline
            )
            results[name] = AudioPipelineTestHelpers.calculateRMSLevel(spectrum)
        }

        // Verify spectral leakage characteristics
        XCTAssertLessThan(
            results["Rectangular"]!,
            results["Hann"]!,
            "Rectangular window should have more spectral leakage than Hann"
        )
        XCTAssertLessThan(
            results["Hamming"]!,
            results["Blackman"]!,
            "Hamming window should have more spectral leakage than Blackman"
        )
    }

    /// Tests frequency band analysis
    /// - Verifies that:
    ///   - Band analysis correctly identifies frequency content
    ///   - Band energy is properly calculated
    ///   - Band boundaries are respected
    func testFrequencyBandAnalysis() throws {
        // Create test buffers
        let inputBuffer = try audioPipeline.allocateBuffer(
            size: 2048 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        let outputBuffer = try audioPipeline.allocateBuffer(
            size: 1024 * MemoryLayout<Float>.stride,
            type: .cpu
        )

        defer {
            audioPipeline.releaseBuffer(id: inputBuffer)
            audioPipeline.releaseBuffer(id: outputBuffer)
        }

        // Configure band settings
        try fftNode.configure(parameters: [
            "bassMin": 20.0,
            "bassMax": 250.0,
            "midMin": 250.0,
            "midMax": 4000.0,
            "trebleMin": 4000.0,
            "trebleMax": 20000.0
        ])

        // Test with bass frequency
        let bassSignal = AudioPipelineTestHelpers.generateSineWave(
            frequency: 100.0,
            duration: 0.1,
            amplitude: 0.5
        )

        try bassSignal.withUnsafeBytes { bytes in
            try audioPipeline.updateBuffer(
                id: inputBuffer,
                data: bytes.baseAddress!,
                size: bassSignal.count * MemoryLayout<Float>.stride,
                options: [.waitForCompletion]
            )
        }

        try audioPipeline.startStream()

        _ = try audioPipeline.process(
            inputBuffers: [inputBuffer],
            outputBuffers: [outputBuffer],
            context: audioPipeline
        )

        let levels = fftNode.getFrequencyBandLevels()
        XCTAssertGreaterThan(levels.bass, levels.mid)
        XCTAssertGreaterThan(levels.bass, levels.treble)
    }

    /// Tests component error handling
    /// - Verifies that:
    ///   - Invalid configurations are handled gracefully
    ///   - Processing errors are properly propagated
    ///   - Component state remains valid after errors
    func testComponentErrorHandling() throws {
        // Test invalid FFT size
        XCTAssertNoThrow(
            try fftNode.configure(parameters: ["fftSize": 0]),
            "Should handle invalid FFT size gracefully"
        )

        // Test invalid window function
        XCTAssertNoThrow(
            try fftNode.configure(parameters: ["windowFunction": "invalid"]),
            "Should handle invalid window function gracefully"
        )

        // Test processing with invalid buffer size
        let smallBuffer = try audioPipeline.allocateBuffer(
            size: 64 * MemoryLayout<Float>.stride,
            type: .cpu
        )
        let outputBuffer = try audioPipeline.allocateBuffer(
            size: 32 * MemoryLayout<Float>.stride,
            type: .cpu
        )

        defer {
            audioPipeline.releaseBuffer(id: smallBuffer)
            audioPipeline.releaseBuffer(id: outputBuffer)
        }

        XCTAssertThrowsError(
            try audioPipeline.process(
                inputBuffers: [smallBuffer],
                outputBuffers: [outputBuffer],
                context: audioPipeline
            ),
            "Should throw error for invalid buffer size"
        )
    }
}

