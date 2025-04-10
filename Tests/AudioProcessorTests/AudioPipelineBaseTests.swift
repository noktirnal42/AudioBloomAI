import XCTest
import Accelerate
@testable import AudioProcessor
@testable import AudioBloomCore

/// Base class for audio pipeline tests that provides shared utilities and setup code
class AudioPipelineBaseTests: XCTestCase {

    // MARK: - Test Variables

    /// The audio pipeline instance under test
    var audioPipeline: AudioPipelineCore!

    /// Test audio buffer size
    let defaultBufferSize = 1024

    /// Default sample rate for testing
    let defaultSampleRate: Double = 44100.0

    /// Default channel count for testing
    let defaultChannelCount = 2

    // MARK: - Setup and Teardown

    override func setUp() async throws {
        try await super.setUp()

        // Create a fresh audio pipeline for each test
        audioPipeline = AudioPipelineCore(
            bufferSize: defaultBufferSize,
            sampleRate: defaultSampleRate,
            channelCount: defaultChannelCount
        )
    }

    override func tearDown() async throws {
        // Clean up any resources
        audioPipeline = nil
        try await super.tearDown()
    }

    // MARK: - Test Data Generation

    /// Generates a sine wave test signal
    /// - Parameters:
    ///   - frequency: Frequency in Hz
    ///   - sampleRate: Sample rate in Hz
    ///   - duration: Duration in seconds
    ///   - amplitude: Signal amplitude (0.0-1.0)
    /// - Returns: Array of float samples
    func generateSineWave(
        frequency: Double,
        sampleRate: Double = 44100.0,
        duration: Double = 1.0,
        amplitude: Double = 0.8
    ) -> [Float] {
        let sampleCount = Int(sampleRate * duration)
        var samples = [Float](repeating: 0, count: sampleCount)

        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Double.pi * frequency * Double(sampleIndex) / sampleRate
            samples[sampleIndex] = Float(amplitude * sin(phase))
        }

        return samples
    }

    /// Generates a test audio buffer with the specified number of channels
    /// - Parameters:
    ///   - data: Array of audio samples for the first channel
    ///   - channelCount: Number of audio channels
    /// - Returns: Audio buffer with interleaved channel data
    func generateTestBuffer(data: [Float], channelCount: Int = 2) -> [Float] {
        // Create an interleaved buffer with the specified number of channels
        var buffer = [Float](repeating: 0, count: data.count * channelCount)

        for sampleIndex in 0..<data.count {
            for channel in 0..<channelCount {
                // For channels beyond the first, use the same data but with reduced amplitude
                let amplitude = channel == 0 ? 1.0 : 0.5
                buffer[sampleIndex * channelCount + channel] = data[sampleIndex] * Float(amplitude)
            }
        }

        return buffer
    }

    /// Creates an audio buffer with random noise
    /// - Parameters:
    ///   - sampleCount: Number of samples per channel
    ///   - channelCount: Number of channels
    ///   - amplitude: Maximum amplitude of the noise
    /// - Returns: Buffer with random audio data
    func generateNoiseBuffer(
        sampleCount: Int,
        channelCount: Int = 2,
        amplitude: Float = 0.5
    ) -> [Float] {
        var buffer = [Float](repeating: 0, count: sampleCount * channelCount)

        for index in 0..<buffer.count {
            // Generate random value between -amplitude and +amplitude
            buffer[index] = (Float.random(in: -1.0...1.0) * amplitude)
        }

        return buffer
    }

    /// Verifies that two audio buffers are approximately equal within a tolerance
    /// - Parameters:
    ///   - buffer1: First buffer to compare
    ///   - buffer2: Second buffer to compare
    ///   - tolerance: Maximum allowed difference between samples
    func assertAudioBuffersEqual(
        _ buffer1: [Float],
        _ buffer2: [Float],
        tolerance: Float = 0.001
    ) {
        XCTAssertEqual(buffer1.count, buffer2.count, "Buffer sizes should match")

        for index in 0..<min(buffer1.count, buffer2.count) {
            XCTAssertEqual(
                buffer1[index],
                buffer2[index],
                accuracy: tolerance,
                "Sample mismatch at index \(index)"
            )
        }
    }

    /// Calculates the RMS (Root Mean Square) level of an audio buffer
    /// - Parameter buffer: Audio buffer to analyze
    /// - Returns: RMS level
    func calculateRMSLevel(_ buffer: [Float]) -> Float {
        guard !buffer.isEmpty else { return 0.0 }

        var sumSquares: Float = 0.0
        for sample in buffer {
            sumSquares += sample * sample
        }

        return sqrt(sumSquares / Float(buffer.count))
    }
}
