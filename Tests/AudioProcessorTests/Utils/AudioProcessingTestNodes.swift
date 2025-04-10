import XCTest
@testable import AudioProcessor
@testable import AudioBloomCore

// MARK: - Basic Test Nodes

/// Simple passthrough node that doesn't modify the signal
final class PassthroughNode: AudioProcessingNode {

    /// Process the audio buffer without any modifications
    /// - Parameters:
    ///   - buffer: Input audio buffer
    ///   - config: Processing configuration
    /// - Returns: Unmodified audio buffer
    func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
        return buffer
    }
}

/// Node that applies a fixed gain to the signal
final class GainNode: AudioProcessingNode {
    /// The gain factor to apply to the audio samples
    let gain: Float

    /// Initialize a gain node with the specified gain factor
    /// - Parameter gain: Gain multiplier (1.0 = unity gain)
    init(gain: Float) {
        self.gain = gain
    }

    /// Apply gain to all samples in the buffer
    /// - Parameters:
    ///   - buffer: Input audio buffer
    ///   - config: Processing configuration
    /// - Returns: Gain-adjusted audio buffer
    func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
        return buffer.map { $0 * gain }
    }
}

/// Node that inverts the phase of the signal
final class InvertNode: AudioProcessingNode {

    /// Invert the phase of all samples (multiply by -1)
    /// - Parameters:
    ///   - buffer: Input audio buffer
    ///   - config: Processing configuration
    /// - Returns: Phase-inverted audio buffer
    func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
        return buffer.map { -$0 }
    }
}

// MARK: - Channel Processing Nodes

/// Node that applies different processing to each channel
final class ChannelAwareNode: AudioProcessingNode {

    /// Process stereo audio with channel-specific processing
    /// Left channel is amplified by 2.0, right channel is attenuated by 0.5
    /// - Parameters:
    ///   - buffer: Input audio buffer (interleaved stereo)
    ///   - config: Processing configuration
    /// - Returns: Processed audio buffer with channel-specific gain
    func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
        var result = buffer

        // Process each stereo pair (left and right channels)
        for channelPairIndex in stride(from: 0, to: buffer.count, by: 2) {
            // Check if left channel index is valid
            if channelPairIndex < buffer.count {
                // Amplify left channel
                result[channelPairIndex] = buffer[channelPairIndex] * 2.0
            }

            // Check if right channel index is valid
            let rightChannelIndex = channelPairIndex + 1
            if rightChannelIndex < buffer.count {
                // Attenuate right channel
                result[rightChannelIndex] = buffer[rightChannelIndex] * 0.5
            }
        }

        return result
    }
}

/// Node that swaps stereo channels
final class ChannelMapperNode: AudioProcessingNode {

    /// Process stereo audio by swapping left and right channels
    /// For mono audio, passes through unchanged
    /// - Parameters:
    ///   - buffer: Input audio buffer
    ///   - config: Processing configuration
    /// - Returns: Audio buffer with swapped channels if stereo
    func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
        guard config.channelCount == 2 else {
            // Only works on stereo signals
            return buffer
        }

        var output = buffer

        // Process each stereo pair, swapping left and right channels
        for channelPairIndex in stride(from: 0, to: buffer.count, by: 2) {
            // Only swap if we have both left and right channels
            let rightIndex = channelPairIndex + 1
            guard rightIndex < buffer.count else {
                continue
            }

            // Swap left and right channels
            output[channelPairIndex] = buffer[rightIndex]
            output[rightIndex] = buffer[channelPairIndex]
        }

        return output
    }
}

// MARK: - Sample Rate and Format Conversion Nodes

/// Node that converts between sample rates
final class SampleRateConverterNode: AudioProcessingNode {
    /// The input sample rate in Hz
    let inputRate: Double

    /// The ratio between output and input sample rates
    let inputToOutputRatio: Double

    /// Initialize a sample rate converter node
    /// - Parameters:
    ///   - inputRate: The input audio sample rate in Hz
    ///   - outputRate: The desired output audio sample rate in Hz
    init(inputRate: Double, outputRate: Double) {
        self.inputRate = inputRate
        self.inputToOutputRatio = outputRate / inputRate
    }

    /// Process audio by simulating sample rate conversion
    /// This is a simple nearest-neighbor implementation for testing purposes
    /// - Parameters:
    ///   - buffer: Input audio buffer at the original sample rate
    ///   - config: Processing configuration
    /// - Returns: Audio buffer resampled to the target sample rate
    func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
        // Simple sample rate conversion simulation
        let outputSize = Int(Double(buffer.count) * inputToOutputRatio)
        var output = [Float](repeating: 0.0, count: outputSize)

        for outputSampleIndex in 0..<outputSize {
            let inputIndexFloat = Double(outputSampleIndex) / inputToOutputRatio
            let inputIndex = Int(inputIndexFloat)

            // Only copy sample if it's within the input buffer bounds
            if inputIndex < buffer.count {
                output[outputSampleIndex] = buffer[inputIndex]
            }
        }

        return output
    }
}

// MARK: - Test Utility Nodes

/// Node that throws an error during processing
final class ErrorThrowingNode: AudioProcessingNode {
    /// Error types that can be thrown during processing
    enum TestError: Error {
        /// Simulated processing failure for testing error handling
        case processingFailed
    }

    /// Always throws a processing error
    /// - Parameters:
    ///   - buffer: Input audio buffer
    ///   - config: Processing configuration
    /// - Returns: Never returns successfully, always throws
    /// - Throws: TestError.processingFailed
    func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
        throw TestError.processingFailed
    }
}

/// Node that counts how many times it has processed audio
final class OperationCounterNode: AudioProcessingNode {
    /// The number of times process() has been called
    var processCount = 0

    /// Process the audio buffer and increment the operation counter
    /// - Parameters:
    ///   - buffer: Input audio buffer
    ///   - config: Processing configuration
    /// - Returns: Unmodified audio buffer
    func process(buffer: [Float], config: AudioProcessingConfig) throws -> [Float] {
        processCount += 1
        return buffer
    }
}



