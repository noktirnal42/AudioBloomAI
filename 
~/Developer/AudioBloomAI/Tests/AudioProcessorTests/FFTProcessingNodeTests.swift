        for sampleIndex in 0..<sampleCount {
            let phase = 2.0 * Double.pi * frequency * Double(sampleIndex) / sampleRate
            samples[sampleIndex] = amplitude * Float(sin(phase))
        var noise = [Float](repeating: 0.0, count: defaultFFTSize)
        for sampleIndex in 0..<defaultFFTSize {
            noise[sampleIndex] = Float.random(in: -0.5...0.5)
        // Verify levels after multiple silence frames
        let finalLevels = fftNode.getFrequencyBandLevels()
        XCTAssertLessThan(finalLevels.bass, 0.1, 
                         "Bass level should approach zero after multiple silence frames")
        for sampleIndex in 0..<defaultFFTSize {
            for component in components {
                let phase = 2.0 * Double.pi * component.frequency * Double(sampleIndex) / sampleRate
                complexSignal[sampleIndex] += component.amplitude * sin(Float(phase))
            }
            // Normalize to prevent clipping
            complexSignal[sampleIndex] = min(max(complexSignal[sampleIndex], -1.0), 1.0)
        var complexSignal = [Float](repeating: 0.0, count: defaultFFTSize)
        for sampleIndex in 0..<defaultFFTSize {
            // Mix several frequencies
            let phase1 = 2.0 * Double.pi * 100.0 * Double(sampleIndex) / sampleRate
            let phase2 = 2.0 * Double.pi * 1000.0 * Double(sampleIndex) / sampleRate
            let phase3 = 2.0 * Double.pi * 5000.0 * Double(sampleIndex) / sampleRate

            complexSignal[sampleIndex] = 0.3 * sin(Float(phase1)) +
                                        0.3 * sin(Float(phase2)) +
                                        0.3 * sin(Float(phase3))
        for binIndex in lowerBound...upperBound {
            mainLobeEnergy += spectrum[binIndex] * spectrum[binIndex]
        }

        // Calculate side lobe energy (all bins outside the main lobe)
        var sideLobeEnergy: Float = 0
        for binIndex in 0..<spectrum.count where binIndex < lowerBound || binIndex > upperBound {
            sideLobeEnergy += spectrum[binIndex] * spectrum[binIndex]
        }
