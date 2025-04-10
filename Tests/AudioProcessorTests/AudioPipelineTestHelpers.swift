import XCTest
import AVFoundation
@testable import AudioProcessor
@testable import AudioBloomCore

/// Shared test helpers for audio pipeline tests
enum AudioPipelineTestHelpers {
    /// Default sample rate for test signals
    static let defaultSampleRate = 44100.0

    /// Creates a default configuration for testing
    static func createDefaultConfig() -> AudioPipelineConfiguration {
        return AudioPipelineConfiguration(
            enableMetalCompute: false,
            defaultFormat: AVAudioFormat(standardFormatWithSampleRate: defaultSampleRate, channels: 2)!,
            bufferSize: 1024,
            maxProcessingLoad: 0.8
        )
    }

    /// Generates a test sine wave buffer
    /// - Parameters:
    ///   - frequency: Frequency in Hz
    ///   - duration: Duration in seconds
    ///   - amplitude: Signal amplitude (0.0-1.0)
    /// - Returns: Array of audio samples
    static func generateSineWave(
        frequency: Double,
        duration: Double,
        amplitude: Float = 1.0
    ) -> [Float] {
        let sampleCount = Int(duration * defaultSampleRate)
        var samples = [Float](repeating: 0.0, count: sampleCount)

        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Double.pi * frequency * Double(sampleIndex) / defaultSampleRate
            samples[sampleIndex] = amplitude * Float(sin(phase))
        }

        return samples
    }

    /// Creates a multi-channel test buffer
    /// - Parameters:
    ///   - data: Source mono audio data
    ///   - channelCount: Number of channels
    /// - Returns: Interleaved multi-channel buffer
    static func generateTestBuffer(data: [Float], channelCount: Int) -> [Float] {
        var buffer = [Float](repeating: 0.0, count: data.count * channelCount)

        for frameIndex in 0..<data.count {
            for channelIndex in 0..<channelCount {
                buffer[frameIndex * channelCount + channelIndex] = data[frameIndex]
            }
        }

        return buffer
    }

    /// Calculates the RMS level of an audio buffer
    /// - Parameter samples: Audio samples to analyze
    /// - Returns: RMS level
    static func calculateRMSLevel(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0.0 }

        let squaredSum = samples.reduce(0.0) { sum, sample in
            sum + sample * sample
        }
        return sqrt(squaredSum / Float(samples.count))
    }

    /// Gets the contents of an audio buffer
    /// - Parameters:
    ///   - bufferId: Buffer ID to read
    ///   - pipeline: Audio pipeline containing the buffer
    /// - Returns: Array of audio samples
    /// - Throws: AudioPipelineError if buffer cannot be accessed
    static func getBufferContents(
        _ bufferId: AudioBufferID,
        pipeline: AudioPipelineCore
    ) throws -> [Float] {
        let buffer = try pipeline.getBuffer(id: bufferId)
        guard let cpuBuffer = buffer.cpuBuffer else {
            throw AudioPipelineError.invalidBufferAccess(
                "Buffer has no CPU-accessible data"
            )
        }

        let floatPtr = cpuBuffer.assumingMemoryBound(to: Float.self)
        let count = buffer.size / MemoryLayout<Float>.stride
        return Array(UnsafeBufferPointer(start: floatPtr, count: count))
    }
}
