
    /// Creates a default configuration for testing
    static func createDefaultConfig() -> AudioPipelineConfiguration {
        return AudioPipelineConfiguration(
            enableMetalCompute: false,
            defaultFormat: AVAudioFormat(standardFormatWithSampleRate: defaultSampleRate, channels: 2)!,
            bufferSize: 1024,
            maxProcessingLoad: 0.8
        )
    }

        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Double.pi * frequency * Double(sampleIndex) / defaultSampleRate
            samples[sampleIndex] = amplitude * Float(sin(phase))
        }

        return samples
    }

        )
    }

    /// Generates a test sine wave buffer
        }

        let floatPtr = cpuBuffer.assumingMemoryBound(to: Float.self)
se { return 0.0 }

        let squaredSum = samples.reduce(0.0) { sum, sample in
ating: 0.0, count: data.count * channelCount)

        for frameIndex in 0..<data.count {
eating: 0.0, count: sampleCount)

        for sampleIndex in 0..<sampleCount {
    }

    /// Calculates the RMS level of an audio buffer


    /// Creates a multi-channel test buffer
    }

    /// Gets the contents of an audio buffer
