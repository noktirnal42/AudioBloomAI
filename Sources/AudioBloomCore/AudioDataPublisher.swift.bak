//
// AudioDataPublisher.swift
// Publisher for audio data using Combine
//

import Foundation
import Combine
import Accelerate

/// A publisher that emits audio data at regular intervals
public class AudioDataPublisher {
    /// The subject that publishes audio data
    private let subject = PassthroughSubject<([Float], (left: Float, right: Float)), Never>()
    
    /// Lock for thread-safe publishing
    private let lock = NSLock()
    
    /// Flag to track if the publisher has been canceled
    private var isCanceled = false
    
    /// Subscriber store to maintain references
    private var subscribers = Set<AnyCancellable>()
    
    /// Public initializer
    public init() {}
    
    /// The publisher that emits audio data
    public var publisher: AnyPublisher<([Float], (left: Float, right: Float)), Never> {
        subject.eraseToAnyPublisher()
    }
    
    /// Publishes new audio data in a thread-safe manner
    public func publish(frequencyData: [Float], levels: (left: Float, right: Float)) {
        lock.lock()
        defer { lock.unlock() }
        
        guard !isCanceled else { return }
        
        // Copy the data to ensure thread safety
        let frequencyCopy = frequencyData
        let levelsCopy = levels
        
        // Send on the main thread to ensure UI compatibility
        DispatchQueue.main.async { [weak self] in
            self?.subject.send((frequencyCopy, levelsCopy))
        }
    }
    
    /// Publishes raw audio data after performing FFT
    public func publishRawAudio(_ samples: [Float], sampleRate: Double) {
        guard !samples.isEmpty, !isCanceled else { return }
        
        // Process on a background queue
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            // Calculate levels
            let level = self.calculateLevel(from: samples)
            
            // Perform FFT
            let fftData = self.performFFT(samples: samples)
            
            // Publish processed data
            self.publish(frequencyData: fftData, levels: (left: level, right: level))
        }
    }
    
    /// Calculate level from audio samples
    private func calculateLevel(from samples: [Float]) -> Float {
        var rms: Float = 0.0
        vDSP_measqv(samples, 1, &rms, vDSP_Length(samples.count))
        return sqrt(rms)
    }
    
    /// Perform FFT on audio samples
    private func performFFT(samples: [Float]) -> [Float] {
        // Determine best FFT size (power of 2)
        let idealSize = 1024 // Default FFT size
        let count = min(samples.count, idealSize)
        
        // Create a Hann window for better frequency resolution
        var window = [Float](repeating: 0, count: count)
        vDSP_hann_window(&window, vDSP_Length(count), Int32(vDSP_HANN_NORM))
        
        // Apply window to samples
        var windowedSamples = [Float](repeating: 0, count: count)
        vDSP_vmul(samples, 1, window, 1, &windowedSamples, 1, vDSP_Length(count))
        
        // Convert to frequency domain
        var realValues = [Float](repeating: 0, count: count/2)
        var imagValues = [Float](repeating: 0, count: count/2)
        
        // Create FFT setup
        let fftSetup = vDSP_create_fftsetup(vDSP_Length(log2(Float(count))), FFTRadix(kFFTRadix2))
        
        // Prepare split complex buffer
        var splitComplex = DSPSplitComplex(realp: &realValues, imagp: &imagValues)
        
        // Convert to split complex format
        windowedSamples.withUnsafeBufferPointer { ptr in
            vDSP_ctoz(ptr.baseAddress!.assumingMemoryBound(to: DSPComplex.self), 2, &splitComplex, 1, vDSP_Length(count/2))
        }
        
        // Perform forward FFT
        vDSP_fft_zrip(fftSetup!, &splitComplex, 1, vDSP_Length(log2(Float(count))), FFTDirection(kFFTDirection_Forward))
        
        // Convert to magnitude
        var magnitude = [Float](repeating: 0, count: count/2)
        vDSP_zvmags(&splitComplex, 1, &magnitude, 1, vDSP_Length(count/2))
        
        // Scale the magnitudes
        var scale = Float(1.0 / Float(count))
        vDSP_vsmul(magnitude, 1, &scale, &magnitude, 1, vDSP_Length(count/2))
        
        // Convert to dB and normalize to 0-1 range
        var normalizedMagnitude = [Float](repeating: 0, count: count/2)
        
        for i in 0..<magnitude.count {
            // Convert to dB with suitable scaling
            let logValue = 10.0 * log10f(magnitude[i] + 1e-6)  // Avoid log(0)
            // Normalize to 0-1 range (adjust range as needed)
            normalizedMagnitude[i] = (logValue + 60.0) / 60.0
            // Clamp to valid range
            normalizedMagnitude[i] = min(max(normalizedMagnitude[i], 0.0), 1.0)
        }
        
        // Cleanup
        vDSP_destroy_fftsetup(fftSetup)
        
        return normalizedMagnitude
    }
    
    /// Subscribe to another publisher and relay its data
    public func linkTo<P: Publisher>(publisher: P) where P.Output == ([Float], (left: Float, right: Float)), P.Failure == Never {
        publisher
            .sink { [weak self] frequencyData, levels in
                self?.publish(frequencyData: frequencyData, levels: levels)
            }
            .store(in: &subscribers)
    }
    
    /// Cancel publishing and cleanup
    public func cancel() {
        lock.lock()
        isCanceled = true
        subscribers.removeAll()
        lock.unlock()
    }
    
    /// Cleanup when deallocated
    deinit {
        cancel()
    }
}

/// Extension for convenience methods
public extension AudioDataPublisher {
    /// Forward data from one publisher to another
    static func forward(from source: AudioDataPublisher, to destination: AudioDataPublisher) {
        destination.linkTo(publisher: source.publisher)
    }
    
    /// Create a publisher with a timer that emits at regular intervals
    static func createTimedPublisher(interval: TimeInterval) -> AudioDataPublisher {
        let publisher = AudioDataPublisher()
        
        // Create a timer publisher
        Timer.publish(every: interval, on: .main, in: .common)
            .autoconnect()
            .sink { _ in
                // Generate simple demo data
                let frequency = [Float](repeating: 0, count: 512).map { _ in Float.random(in: 0...1) }
                let levels = (Float.random(in: 0...1), Float.random(in: 0...1))
                publisher.publish(frequencyData: frequency, levels: levels)
            }
            .store(in: &publisher.subscribers)
        
        return publisher
    }
}

