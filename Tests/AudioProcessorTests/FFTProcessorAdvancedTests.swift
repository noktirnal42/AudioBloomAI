import XCTest
import AVFoundation
@testable import AudioProcessor
@testable import AudioBloomCore

/// Tests for advanced FFT processing capabilities
final class FFTProcessorAdvancedTests: FFTProcessorBaseTests {

    // MARK: - Basic FFT Processing Tests

    /// Tests processing of single frequency signals
    func testProcessingSingleFrequencies() async throws {
        // Test frequencies to verify
        let testFrequencies = [110.0, 440.0, 880.0, 1760.0]

        for testFrequency in testFrequencies {
            // Generate a sine wave at the test frequency
            let testSignal = generateSineWave(
                frequency: testFrequency,
                amplitude: 0.8,
                sampleCount: defaultFFTSize,
                sampleRate: sampleRate
            )

            // Process the signal through the FFT
            let spectrum = try await processTestData(testSignal)

            // Verify the peak is at the expected frequency
            let expectedBin = binIndexForFrequency(testFrequency, fftSize: defaultFFTSize, sampleRate: sampleRate)
            let (actualPeakBin, _) = findPeakInSpectrum(spectrum)

            // Allow for some bin error due to frequency resolution limitations
            let binError = 2
            XCTAssertTrue(
                abs(actualPeakBin - expectedBin) <= binError,
                "FFT should detect frequency \(testFrequency) Hz at bin \(expectedBin), " +
                "but peak was found at bin \(actualPeakBin)"
            )
        }
    }

    /// Tests the accuracy of FFT magnitude calculations
    func testMagnitudeAccuracy() async throws {
        // Test amplitudes to verify
        let testAmplitudes: [Float] = [0.1, 0.5, 0.9]
        let testFrequency = 1000.0 // 1kHz test tone

        for testAmplitude in testAmplitudes {
            // Generate a sine wave with the test amplitude
            let testSignal = generateSineWave(
                frequency: testFrequency,
                amplitude: testAmplitude,
                sampleCount: defaultFFTSize,
                sampleRate: sampleRate
            )

            // Process the signal through the FFT
            let spectrum = try await processTestData(testSignal)

            // Find the peak magnitude
            let (_, peakMagnitude) = findPeakInSpectrum(spectrum)

            // The relationship between input amplitude and FFT magnitude depends on several factors:
            // - Window function scaling
            // - FFT normalization
            // - Signal length

            // For a Hann window and a sine wave, the relationship is approximately:
            // magnitude ≈ amplitude * windowScale * fftScale
            // where windowScale ≈ 0.5 and fftScale is dependent on the FFT implementation

            // We're testing relative scaling rather than absolute values
            if testAmplitudes.count > 1 {
                let previousAmplitude = testAmplitudes[testAmplitudes.firstIndex(of: testAmplitude)! - 1]
                let previousSpectrum = try await processTestData(generateSineWave(
                    frequency: testFrequency,
                    amplitude: previousAmplitude,
                    sampleCount: defaultFFTSize,
                    sampleRate: sampleRate
                ))
                let (_, previousMagnitude) = findPeakInSpectrum(previousSpectrum)

                // The ratio of magnitudes should approximate the ratio of amplitudes
                let magnitudeRatio = peakMagnitude / previousMagnitude
                let amplitudeRatio = testAmplitude / previousAmplitude

                XCTAssertEqual(
                    magnitudeRatio,
                    amplitudeRatio,
                    accuracy: 0.1,
                    "Magnitude ratio should be proportional to amplitude ratio"
                )
            }
        }
    }

    /// Tests processing of silence and noise
    func testProcessingSilenceAndNoise() async throws {
        // Test with silence (all zeros)
        let silenceSignal = [Float](repeating: 0.0, count: defaultFFTSize)
        let silenceSpectrum = try await processTestData(silenceSignal)

        // Silence should produce very low magnitude across all bins
        let maxSilenceMagnitude = silenceSpectrum.max() ?? 0.0
        XCTAssertLessThan(
            maxSilenceMagnitude,
            0.01,
            "Silence should produce near-zero magnitudes"
        )

        // Test with white noise
        let noiseSignal = generateWhiteNoise(amplitude: 0.5, sampleCount: defaultFFTSize)
        let noiseSpectrum = try await processTestData(noiseSignal)

        // White noise should have relatively even distribution across frequencies
        let noiseAverage = noiseSpectrum.reduce(0, +) / Float(noiseSpectrum.count)
        var binsBelowThreshold = 0
        var binsAboveThreshold = 0

        for magnitude in noiseSpectrum {
            if magnitude < noiseAverage * 0.5 {
                binsBelowThreshold += 1
            } else if magnitude > noiseAverage * 1.5 {
                binsAboveThreshold += 1
            }
        }

        // Most bins should be within 50% of the average for white noise
        let outlierBins = binsBelowThreshold + binsAboveThreshold
        let outlierRatio = Float(outlierBins) / Float(noiseSpectrum.count)

        XCTAssertLessThan(
            outlierRatio,
            0.3,
            "White noise should have relatively even frequency distribution"
        )
    }

    /// Tests processing of signals with multiple frequency components
    func testMultiFrequencySignal() async throws {
        // Create a signal with multiple frequency components
        let frequencies = [261.63, 329.63, 392.0] // C4, E4, G4 (C major chord)
        let amplitudes: [Float] = [0.5, 0.3, 0.4]

        let testSignal = generateMultiFrequencySignal(
            frequencies: frequencies,
            amplitudes: amplitudes,
            sampleCount: defaultFFTSize,
            sampleRate: sampleRate
        )

        // Process the signal
        let spectrum = try await processTestData(testSignal)

        // Find the peaks corresponding to each frequency component
        for (index, frequency) in frequencies.enumerated() {
            let expectedBin = binIndexForFrequency(frequency, fftSize: defaultFFTSize, sampleRate: sampleRate)

            // Check if there's a significant magnitude at the expected bin
            let binRange = max(0, expectedBin - 2)...min(spectrum.count - 1, expectedBin + 2)

            let peakInRange = binRange.contains { binIndex in
                spectrum[binIndex] > 0.1 // Threshold for detection
            }

            XCTAssertTrue(
                peakInRange,
                "Frequency component at \(frequency) Hz should be detected"
            )
        }

        // Verify that the relative magnitudes of the peaks correspond to the input amplitudes
        let expectedPeaks = frequencies.map {
            binIndexForFrequency($0, fftSize: defaultFFTSize, sampleRate: sampleRate)
        }

        // Extract magnitudes at the expected peaks
        var peakMagnitudes: [Float] = []
        for expectedBin in expectedPeaks {
            // Look for the maximum in a small window around the expected bin
            let binStart = max(0, expectedBin - 2)
            let binEnd = min(spectrum.count - 1, expectedBin + 2)
            var maxMagnitude: Float = 0.0

            for binIndex in binStart...binEnd {
                maxMagnitude = max(maxMagnitude, spectrum[binIndex])
            }

            peakMagnitudes.append(maxMagnitude)
        }

        // The highest amplitude should correspond to the highest magnitude
        let highestAmplitudeIndex = amplitudes.firstIndex(of: amplitudes.max()!)!
        let highestMagnitudeIndex = peakMagnitudes.firstIndex(of: peakMagnitudes.max()!)!

        XCTAssertEqual(
            highestAmplitudeIndex,
            highestMagnitudeIndex,
            "Highest input amplitude should produce highest FFT magnitude"
        )
    }

    // MARK: - Frequency Band Tests

    /// Helper method to test frequency band energy for a specific signal
    /// - Parameters:
    ///   - signal: The audio signal to analyze
    ///   - minFreq: The minimum frequency of the band
    ///   - maxFreq: The maximum frequency of the band
    ///   - expectInBand: Whether the energy should be in or out of the band
    /// - Returns: A tuple containing the band energy ratio and total energy
    private func testFrequencyBandEnergy(
        signal: [Float],
        minFreq: Double,
        maxFreq: Double,
        expectInBand: Bool
    ) async throws -> (bandEnergyRatio: Float, totalEnergy: Float) {
        // Process the signal
        let spectrum = try await processTestData(signal)

        // Calculate energy in the frequency band
        var bandEnergy: Float = 0.0
        let minBin = binIndexForFrequency(minFreq, fftSize: defaultFFTSize, sampleRate: sampleRate)
        let maxBin = binIndexForFrequency(maxFreq, fftSize: defaultFFTSize, sampleRate: sampleRate)

        for binIndex in minBin...maxBin where binIndex < spectrum.count {
            bandEnergy += spectrum[binIndex]
        }

        // Calculate total energy in the spectrum
        let totalEnergy = spectrum.reduce(0, +)

        // Calculate the ratio of band energy to total energy
        let bandEnergyRatio = bandEnergy / totalEnergy

        return (bandEnergyRatio, totalEnergy)
    }

    /// Tests detection of frequency bands
    func testFrequencyBandDetection() async throws {
        // Configure frequency bands to test
        let bandRanges = [
            (20.0, 200.0),    // Low frequency band
            (500.0, 2000.0),  // Mid frequency band
            (5000.0, 10000.0) // High frequency band
        ]

        for (minFreq, maxFreq) in bandRanges {
            try await testSingleFrequencyBand(minFreq: minFreq, maxFreq: maxFreq)
        }
    }

    /// Helper method to test a single frequency band
    /// - Parameters:
    ///   - minFreq: The minimum frequency of the band
    ///   - maxFreq: The maximum frequency of the band
    private func testSingleFrequencyBand(minFreq: Double, maxFreq: Double) async throws {
        // Create frequencies in the middle of the band and outside the band
        let inBandFreq = (minFreq + maxFreq) / 2
        let belowBandFreq = minFreq * 0.5
        let aboveBandFreq = maxFreq * 1.5

        // Test with a frequency in the band
        let inBandSignal = generateSineWave(
            frequency: inBandFreq,
            amplitude: 0.8,
            sampleCount: defaultFFTSize,
            sampleRate: sampleRate
        )

        // Test with frequencies outside the band
        let belowBandSignal = generateSineWave(
            frequency: belowBandFreq,
            amplitude: 0.8,
            sampleCount: defaultFFTSize,
            sampleRate: sampleRate
        )

        let aboveBandSignal = generateSineWave(
            frequency: aboveBandFreq,
            amplitude: 0.8,
            sampleCount: defaultFFTSize,
            sampleRate: sampleRate
        )

        // Test the in-band signal
        let (inBandRatio, _) = try await testFrequencyBandEnergy(
            signal: inBandSignal,
            minFreq: minFreq,
            maxFreq: maxFreq,
            expectInBand: true
        )

        // Test the below-band signal
        let (belowBandRatio, _) = try await testFrequencyBandEnergy(
            signal: belowBandSignal,
            minFreq: minFreq,
            maxFreq: maxFreq,
            expectInBand: false
        )

        // Test the above-band signal
        let (aboveBandRatio, _) = try await testFrequencyBandEnergy(
            signal: aboveBandSignal,
            minFreq: minFreq,
            maxFreq: maxFreq,
            expectInBand: false
        )

        // Verify the energy ratios
        XCTAssertGreaterThan(
            inBandRatio,
            0.7,
            "Most energy should be in the frequency band"
        )

        XCTAssertLessThan(
            belowBandRatio,
            0.3,
            "Most energy should be outside the frequency band for below-band signal"
        )

        XCTAssertLessThan(
            aboveBandRatio,
            0.3,
            "Most energy should be outside the frequency band for above-band signal"
        )
    }

    /// Tests the behavior of frequency band smoothing
    func testBandSmoothingBehavior() async throws {
        // Configure the FFT node with smoothing
        try fftNode.setBandSmoothing(0.7) // 70% smoothing (significant)

        // Generate test signals
        let firstFrequency = 440.0
        let secondFrequency = 880.0

        let firstSignal = generateSineWave(
            frequency: firstFrequency,
            amplitude: 0.8,
            sampleCount: defaultFFTSize,
            sampleRate: sampleRate
        )

        let secondSignal = generateSineWave(
            frequency: secondFrequency,
            amplitude: 0.8,
            sampleCount: defaultFFTSize,
            sampleRate: sampleRate
        )

        // Test smoothing behavior
        try await testSmoothingTransition(
            firstSignal: firstSignal,
            secondSignal: secondSignal,
            firstFrequency: firstFrequency,
            secondFrequency: secondFrequency
        )

        // Reset smoothing for other tests
        try fftNode.setBandSmoothing(0.0)
    }

    /// Helper method to test smoothing transition between two signals
    /// - Parameters:
    ///   - firstSignal: The first signal
    ///   - secondSignal: The second signal
    ///   - firstFrequency: The frequency of the first signal
    ///   - secondFrequency: The frequency of the second signal
    private func testSmoothingTransition(
        firstSignal: [Float],
        secondSignal: [Float],
        firstFrequency: Double,
        secondFrequency: Double
    ) async throws {
        // Process the first signal multiple times to establish a baseline
        var firstSpectra: [[Float]] = []
        for _ in 0..<3 {
            let spectrum = try await processTestData(firstSignal)
            firstSpectra.append(spectrum)
        }

        // Process the second signal once (first transition)
        let transitionSpectrum = try await processTestData(secondSignal)

        // Process the second signal multiple times
        var secondSpectra: [[Float]] = []
        for _ in 0..<3 {
            let spectrum = try await processTestData(secondSignal)
            secondSpectra.append(spectrum)
        }

        // The transition spectrum should show influence from the first signal due to smoothing
        let firstBin = binIndexForFrequency(firstFrequency, fftSize: defaultFFTSize, sampleRate: sampleRate)
        let secondBin = binIndexForFrequency(secondFrequency, fftSize: defaultFFTSize, sampleRate: sampleRate)

        // Verify that there's energy at both frequency bins in the transition spectrum
        XCTAssertGreaterThan(
            transitionSpectrum[firstBin],
            0.1,
            "Transition spectrum should retain energy from first frequency due to smoothing"
        )

        XCTAssertGreaterThan(
            transitionSpectrum[secondBin],
            0.1,
            "Transition spectrum should show energy at new frequency"
        )

        // Verify that later spectra converge toward the new frequency
        XCTAssertGreaterThan(
            secondSpectra.last![secondBin],
            transitionSpectrum[secondBin],
            "Energy at new frequency should increase over time"
        )

        XCTAssertLessThan(
            secondSpectra.last![firstBin],
            transitionSpectrum[firstBin],
            "Energy at old frequency should decrease over time"
        )
    }
}
