import Foundation
import Combine
import CoreGraphics
import Accelerate
import AudioBloomCore

/// A bridging class that optimizes audio data for visualization purposes
public class AudioVisualizerBridge: ObservableObject {
    // MARK: - Published Properties
    
    /// Optimized frequency data for visualization
    @Published public private(set) var visualizationData: [Float] = []
    
    /// Smoothed audio levels
    @Published public private(set) var smoothedLevels: (left: Float, right: Float) = (0, 0)
    
    /// Frequency band energy levels
    @Published public private(set) var frequencyBands: AudioFrequencyBands = AudioFrequencyBands()
    
    /// Beat detection status
    @Published public private(set) var beatDetected: Bool = false
    
    /// Energy level as detected by signal processing
    @Published public private(set) var energyLevel: Float = 0.0
    
    // MARK: - Configuration Properties
    
    /// Configuration for the audio-visual bridge
    public struct Configuration {
        /// Smoothing factor for audio levels (0.0-1.0, higher = more smoothing)
        public var levelSmoothingFactor: Float = 0.7
        
        /// Smoothing factor for frequency data (0.0-1.0, higher = more smoothing)
        public var fftSmoothingFactor: Float = 0.5
        
        /// Enhancement factor for visualizations (0.0-1.0, higher = more enhancement)
        public var visualEnhancementFactor: Float = 0.8
        
        /// Beat detection sensitivity (0.0-1.0, higher = more sensitive)
        public var beatDetectionSensitivity: Float = 0.6
        
        /// Whether to apply spectral weighting for better visualization
        public var applySpectralWeighting: Bool = true
        
        /// Whether to use logarithmic frequency scaling
        public var useLogFrequencyScaling: Bool = true
        
        /// Frequency resolution for the visualization
        public var visualizationResolution: Int = 128
        
        /// Maximum frequency for visualization (in Hz)
        public var maxFrequency: Float = 16000.0
        
        /// Neural processing weight (0.0-1.0, higher = more neural influence)
        public var neuralProcessingWeight: Float = 0.3
        
        public init() {}
    }
    
    /// The current configuration
    public var configuration: Configuration {
        didSet {
            // Re-initialize components if needed
            initializeFrequencyBands()
            initializeProcessingFilters()
        }
    }
    
    // MARK: - Private Properties
    
    /// Subscription for audio data
    private var audioDataSubscription: AnyCancellable?
    
    /// Last raw frequency data received
    private var lastRawFrequencyData: [Float] = []
    
    /// Last raw audio levels received
    private var lastRawLevels: (left: Float, right: Float) = (0, 0)
    
    /// Last timestamp for processing
    private var lastProcessingTime: TimeInterval = 0
    
    /// Buffer for smoothed frequency data
    private var smoothedFrequencyData: [Float] = []
    
    /// Energy history for beat detection
    private var energyHistory: [Float] = Array(repeating: 0, count: 43)
    
    /// Beat detection timing
    private var lastBeatTime: TimeInterval = 0
    
    /// History of beat intervals (in seconds)
    private var beatIntervals: [TimeInterval] = []
    
    /// FFT scaling factors for perceptual weighting
    private var fftScalingFactors: [Float] = []
    
    /// Resampling filter for visualization resolution
    private var resamplingFilter: ResamplingFilter?
    
    /// Type that holds frequency band energy levels
    public struct AudioFrequencyBands {
        /// Sub-bass frequency band energy (20-60 Hz)
        public var subBass: Float = 0
        
        /// Bass frequency band energy (60-250 Hz)
        public var bass: Float = 0
        
        /// Low-midrange frequency band energy (250-500 Hz)
        public var lowMid: Float = 0
        
        /// Midrange frequency band energy (500-2000 Hz)
        public var mid: Float = 0
        
        /// Upper-midrange frequency band energy (2000-4000 Hz)
        public var highMid: Float = 0
        
        /// Presence frequency band energy (4000-6000 Hz)
        public var presence: Float = 0
        
        /// Brilliance frequency band energy (6000-20000 Hz)
        public var brilliance: Float = 0
        
        public init() {}
    }
    
    /// Neural processing hook response
    public struct NeuralProcessingResponse {
        /// Overall energy level from neural analysis
        public var energy: Float = 0
        
        /// Pleasantness factor from neural analysis
        public var pleasantness: Float = 0
        
        /// Complexity factor from neural analysis
        public var complexity: Float = 0
        
        /// Whether a beat was detected
        public var beatDetected: Bool = false
        
        /// Recommended visualization mode
        public var recommendedMode: Int = 0
        
        public init() {}
    }
    
    /// Resampling filter for changing resolution
    private class ResamplingFilter {
        /// Original data size
        private let originalSize: Int
        
        /// Target data size
        private let targetSize: Int
        
        /// Resampling matrix
        private let resamplingMatrix: [Float]
        
        init(originalSize: Int, targetSize: Int) {
            self.originalSize = originalSize
            self.targetSize = targetSize
            
            // Create a matrix for resampling
            var matrix = [Float](repeating: 0, count: targetSize * originalSize)
            
            // For each output point, calculate interpolation weights
            for i in 0..<targetSize {
                let x = Float(i) / Float(targetSize - 1) * Float(originalSize - 1)
                let x0 = Int(x)
                let x1 = min(x0 + 1, originalSize - 1)
                let t = x - Float(x0)
                
                // Linear interpolation weights
                matrix[i * originalSize + x0] = 1.0 - t
                matrix[i * originalSize + x1] = t
            }
            
            self.resamplingMatrix = matrix
        }
        
        /// Resample data to target size
        func resample(_ data: [Float]) -> [Float] {
            guard data.count == originalSize else {
                // If sizes don't match, return original data or properly sized zeros
                return data.count < targetSize ? 
                    data + [Float](repeating: 0, count: targetSize - data.count) :
                    Array(data.prefix(targetSize))
            }
            
            var result = [Float](repeating: 0, count: targetSize)
            
            // Apply resampling matrix
            for i in 0..<targetSize {
                var sum: Float = 0
                for j in 0..<originalSize {
                    sum += data[j] * resamplingMatrix[i * originalSize + j]
                }
                result[i] = sum
            }
            
            return result
        }
    }
    
    // MARK: - Initialization
    
    /// Initializes a new AudioVisualizerBridge
    /// - Parameter configuration: Configuration for audio-visual processing
    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
        
        // Initialize with default values
        initializeFrequencyBands()
        initializeProcessingFilters()
    }
    
    /// Initialize frequency bands
    private func initializeFrequencyBands() {
        // Reset frequency bands
        frequencyBands = AudioFrequencyBands()
    }
    
    /// Initialize processing filters
    private func initializeProcessingFilters() {
        // Create spectral weighting factors if needed
        if configuration.applySpectralWeighting {
            createSpectralWeightingFactors()
        } else {
            fftScalingFactors = []
        }
        
        // Create resampling filter if needed
        if configuration.visualizationResolution > 0 {
            resamplingFilter = ResamplingFilter(
                originalSize: AudioBloomCore.Constants.defaultFFTSize / 2,
                targetSize: configuration.visualizationResolution
            )
        } else {
            resamplingFilter = nil
        }
    }
    
    /// Create spectral weighting factors based on perceptual curves
    private func createSpectralWeightingFactors() {
        let fftSize = AudioBloomCore.Constants.defaultFFTSize / 2
        let sampleRate = Float(AudioBloomCore.Constants.defaultSampleRate)
        
        // Create a new array for scaling factors
        fftScalingFactors = [Float](repeating: 1.0, count: fftSize)
        
        // Apply A-weighting curve (approximated) for more perceptually relevant visualization
        for i in 0..<fftSize {
            let frequency = Float(i) / Float(fftSize) * (sampleRate / 2.0)
            
            // Skip DC component
            if i == 0 {
                fftScalingFactors[i] = 0.0
                continue
            }
            
            // Approximate A-weighting curve for better visualization
            // This emphasizes frequencies where human hearing is most sensitive
            let f2 = frequency * frequency
            let aWeight = f2 * f2 / ((f2 + 20.6 * 20.6) * (f2 + 12194.0 * 12194.0) * 
                                    sqrt((f2 + 107.7 * 107.7) * (f2 + 737.9 * 737.9)))
            
            // Apply additional visualization boost for mid/high frequencies
            let visualBoost = sqrt(frequency / 1000.0) * configuration.visualEnhancementFactor
            
            // Combine both effects
            fftScalingFactors[i] = Float(aWeight) * (1.0 + visualBoost)
        }
        
        // Normalize the scaling factors
        let maxFactor = fftScalingFactors.max() ?? 1.0
        if maxFactor > 0 {
            for i in 0..<fftSize {
                fftScalingFactors[i] /= maxFactor
            }
        }
    }
    
    // MARK: - Public Methods
    
    /// Subscribes to audio data from an AudioEngine
    /// Subscribes to audio data from an AudioEngine
    /// - Parameter audioEngine: The audio engine to subscribe to
    public func subscribeToAudioEngine(_ audioEngine: AudioDataProvider) {
        // Cancel any existing subscription
        audioDataSubscription?.cancel()
        
        // Subscribe to the audio data publisher
        do {
            if let engine = audioEngine as? AudioEngine {
                // Handle AudioEngine implementation
                audioDataSubscription = engine.getAudioDataPublisher()
                    .publisher
                    .sink { [weak self] (frequencyData, levels) in
                        self?.processAudioData(frequencyData: frequencyData, levels: levels)
                    }
                print("Successfully subscribed to AudioEngine data publisher")
            } else if let processor = audioEngine as? AudioVisualizerProcessor {
                // Handle AudioVisualizerProcessor implementation
                let publisher = processor.getAudioDataPublisher().0
                audioDataSubscription = publisher
                    .sink { [weak self] visualizationData in
                        self?.processAudioData(
                            frequencyData: visualizationData.frequencyData,
                            levels: visualizationData.levels
                        )
                    }
                print("Successfully subscribed to AudioVisualizerProcessor data publisher")
            } else {
                // Generic fallback using AudioDataProvider protocol
                audioDataSubscription = audioEngine.getAudioDataPublisher()
                    .publisher
                    .sink { [weak self] (frequencyData, levels) in
                        self?.processAudioData(frequencyData: frequencyData, levels: levels)
                    }
                print("Successfully subscribed to generic AudioDataProvider")
            }
        } catch {
            print("Error subscribing to audio data: \(error.localizedDescription)")
        }
    
    /// Processes new audio data for visualization
    /// - Parameters:
    ///   - frequencyData: Raw frequency data from FFT
    ///   - levels: Audio levels (left, right)
    private func processAudioData(frequencyData: [Float], levels: (left: Float, right: Float)) {
        // Store raw data
        lastRawFrequencyData = frequencyData
        lastRawLevels = levels
        
        // Get current time for timing calculations
        let currentTime = CACurrentMediaTime()
        let deltaTime = currentTime - lastProcessingTime
        lastProcessingTime = currentTime
        
        // Apply spectral weighting if enabled
        var processedData = applySpectralWeighting(frequencyData)
        
        // Apply smoothing to frequency data
        processedData = smoothFrequencyData(processedData, deltaTime: deltaTime)
        
        // Calculate frequency bands
        calculateFrequencyBands(from: processedData)
        
        // Apply logarithmic scaling if enabled
        if configuration.useLogFrequencyScaling {
            processedData = applyLogFrequencyScaling(processedData)
        }
        
        // Resample to visualization resolution if needed
        if let resamplingFilter = resamplingFilter {
            processedData = resamplingFilter.resample(processedData)
        }
        
        // Calculate overall energy level
        calculateEnergyLevel(from: processedData, levels: levels)
        
        // Detect beats
        detectBeats(energy: energyLevel, time: currentTime)
        
        // Apply smoothing to audio levels
        let smoothedLeft = smoothValue(levels.left, 
                                      previous: smoothedLevels.left, 
                                      factor: configuration.levelSmoothingFactor)
        let smoothedRight = smoothValue(levels.right, 
                                       previous: smoothedLevels.right, 
                                       factor: configuration.levelSmoothingFactor)
        
        // Update published values on main thread
        DispatchQueue.main.async {
            self.visualizationData = processedData
            self.smoothedLevels = (left: smoothedLeft, right: smoothedRight)
        }
    }
    
    /// Applies spectral weighting to frequency data
    /// - Parameter data: Raw frequency data
    /// - Returns: Weighted frequency data
    private func applySpectralWeighting(_ data: [Float]) -> [Float] {
        guard configuration.applySpectralWeighting,
              !fftScalingFactors.isEmpty,
              data.count == fftScalingFactors.count else {
            return data
        }
        
        var weightedData = [Float](repeating: 0, count: data.count)
        
        // Apply weighting factors using Accelerate for better performance
        vDSP_vmul(data, 1,
                 fftScalingFactors, 1,
                 &weightedData, 1,
                 vDSP_Length(data.count))
        
        return weightedData
    }
    
    /// Applies logarithmic scaling to frequency data for better visualization
    /// - Parameter data: Linear frequency data
    /// - Returns: Log-scaled frequency data
    private func applyLogFrequencyScaling(_ data: [Float]) -> [Float] {
        guard data.count > 0 else { return data }
        
        let fftSize = data.count
        let outputSize = configuration.visualizationResolution > 0 ? 
            configuration.visualizationResolution : fftSize
        
        var logScaledData = [Float](repeating: 0, count: outputSize)
        
        // Apply logarithmic scaling
        // This gives more space to lower frequencies where there's more perceptual detail
        for i in 0..<outputSize {
            // Map output index to exponential position in the input array
            let ratio = Float(i) / Float(outputSize)
            let logIndex = exp(ratio * log(Float(fftSize))) - 1
            
            // Interpolate between nearest samples
            let index1 = min(Int(logIndex), fftSize - 1)
            let index2 = min(index1 + 1, fftSize - 1)
            let fraction = logIndex - Float(index1)
            
            // Linear interpolation
            logScaledData[i] = data[index1] * (1 - fraction) + data[index2] * fraction
        }
        
        return logScaledData
    }
    
    /// Smooths frequency data for visualization
    /// - Parameters:
    ///   - data: Input frequency data
    ///   - deltaTime: Time since last update for adaptive smoothing
    /// - Returns: Smoothed frequency data
    private func smoothFrequencyData(_ data: [Float], deltaTime: TimeInterval) -> [Float] {
        // Initialize smoothed data array if needed
        if smoothedFrequencyData.count != data.count {
            smoothedFrequencyData = data
            return data
        }
        
        // Calculate adaptive smoothing factor based on time delta
        let adaptiveFactor = min(1.0, Float(deltaTime) * 30.0)
        let smoothingFactor = configuration.fftSmoothingFactor * (1.0 - adaptiveFactor)
        
        // Apply smoothing with properly bound checked arrays
        let count = min(data.count, smoothedFrequencyData.count)
        var result = [Float](repeating: 0, count: data.count)
        
        // Use Accelerate for efficient vector processing
        if count > 0 {
            // Calculate 1.0 - smoothingFactor for new data weight
            let newDataWeight = 1.0 - smoothingFactor
            
            // Scale previous data by smoothing factor
            vDSP_vsmul(smoothedFrequencyData, 1, &smoothingFactor, &result, 1, vDSP_Length(count))
            
            // Create temporary array for scaled new data
            var scaledNewData = [Float](repeating: 0, count: count)
            vDSP_vsmul(data, 1, &newDataWeight, &scaledNewData, 1, vDSP_Length(count))
            
            // Add scaled new data to result
            vDSP_vadd(result, 1, scaledNewData, 1, &result, 1, vDSP_Length(count))
        }
        
        // Update stored smoothed data
        smoothedFrequencyData = result
        
        return result
    }
    
    /// Utility function to smooth any value with a given factor
    /// - Parameters:
    ///   - newValue: New input value
    ///   - previous: Previous smoothed value
    ///   - factor: Smoothing factor (0.0-1.0, higher = more smoothing)
    /// - Returns: Smoothed value
    private func smoothValue(_ newValue: Float, previous: Float, factor: Float) -> Float {
        return previous * factor + newValue * (1.0 - factor)
    }
    
    /// Calculates frequency bands from frequency data
    /// - Parameter data: Input frequency data
    private func calculateFrequencyBands(from data: [Float]) {
        guard !data.isEmpty else { return }
        
        let fftSize = data.count
        let sampleRate = Float(AudioBloomCore.Constants.defaultSampleRate)
        
        // Define frequency band ranges
        let bandRanges: [(String, Float, Float)] = [
            ("subBass", 20, 60),
            ("bass", 60, 250),
            ("lowMid", 250, 500),
            ("mid", 500, 2000),
            ("highMid", 2000, 4000),
            ("presence", 4000, 6000),
            ("brilliance", 6000, min(sampleRate / 2, 20000))
        ]
        
        var newBands = AudioFrequencyBands()
        
        // Calculate energy for each band
        for (band, lowFreq, highFreq) in bandRanges {
            // Convert frequencies to FFT bin indices
            let lowBin = max(0, Int((lowFreq / sampleRate) * Float(fftSize * 2)))
            let highBin = min(fftSize - 1, Int((highFreq / sampleRate) * Float(fftSize * 2)))
            
            // Compute average energy in the band
            var energy: Float = 0
            var count = 0
            
            for i in lowBin...highBin {
                if i < data.count {
                    energy += data[i]
                    count += 1
                }
            }
            
            // Normalize by bin count
            let bandEnergy = count > 0 ? energy / Float(count) : 0
            
            // Apply smoothing with previous band value for a more stable visualization
            let previousValue: Float
            switch band {
            case "subBass": previousValue = frequencyBands.subBass
            case "bass": previousValue = frequencyBands.bass
            case "lowMid": previousValue = frequencyBands.lowMid
            case "mid": previousValue = frequencyBands.mid
            case "highMid": previousValue = frequencyBands.highMid
            case "presence": previousValue = frequencyBands.presence
            case "brilliance": previousValue = frequencyBands.brilliance
            default: previousValue = 0
            }
            
            let smoothedEnergy = smoothValue(bandEnergy, 
                                           previous: previousValue, 
                                           factor: configuration.levelSmoothingFactor)
            
            // Update the appropriate band
            switch band {
            case "subBass": newBands.subBass = smoothedEnergy
            case "bass": newBands.bass = smoothedEnergy
            case "lowMid": newBands.lowMid = smoothedEnergy
            case "mid": newBands.mid = smoothedEnergy
            case "highMid": newBands.highMid = smoothedEnergy
            case "presence": newBands.presence = smoothedEnergy
            case "brilliance": newBands.brilliance = smoothedEnergy
            default: break
            }
        }
        
        // Update published bands on main thread
        DispatchQueue.main.async {
            self.frequencyBands = newBands
        }
    }
    
    /// Calculates overall energy level from audio data
    /// - Parameters:
    ///   - data: Frequency data
    ///   - levels: Audio levels
    private func calculateEnergyLevel(from data: [Float], levels: (left: Float, right: Float)) {
        // Calculate energy from frequency data
        var energy: Float = 0
        
        // Put more weight on bass and mid frequencies for energy
        if !data.isEmpty {
            let fftSize = data.count
            let bassEnd = min(fftSize / 8, fftSize - 1)  // First 1/8 of spectrum = bass
            let midEnd = min(fftSize / 2, fftSize - 1)   // First half = mid
            
            var bassEnergy: Float = 0
            var midEnergy: Float = 0
            var highEnergy: Float = 0
            
            // Calculate energy in each band
            for i in 0...bassEnd {
                bassEnergy += data[i]
            }
            bassEnergy /= Float(bassEnd + 1)
            
            for i in (bassEnd + 1)...midEnd {
                midEnergy += data[i]
            }
            midEnergy /= Float(midEnd - bassEnd)
            
            if midEnd + 1 < fftSize {
                for i in (midEnd + 1)..<fftSize {
                    highEnergy += data[i]
                }
                highEnergy /= Float(fftSize - midEnd - 1)
            }
            
            // Weight the bands differently for better energy representation
            energy = bassEnergy * 0.5 + midEnergy * 0.35 + highEnergy * 0.15
        }
        
        // Also consider audio levels
        let levelEnergy = (levels.left + levels.right) / 2.0
        
        // Combine frequency energy and level energy
        let combinedEnergy = energy * 0.7 + levelEnergy * 0.3
        
        // Apply time-based smoothing
        let newEnergyLevel = smoothValue(combinedEnergy, 
                                        previous: energyLevel, 
                                        factor: configuration.levelSmoothingFactor * 0.8)
        
        // Update energy level
        DispatchQueue.main.async {
            self.energyLevel = newEnergyLevel
        }
    }
    
    /// Beat detection algorithm
    /// - Parameters:
    ///   - energy: Current energy level
    ///   - time: Current time
    private func detectBeats(energy: Float, time: TimeInterval) {
        // Shift energy history array
        for i in 0..<energyHistory.count - 1 {
            energyHistory[i] = energyHistory[i + 1]
        }
        
        // Add new energy value
        energyHistory[energyHistory.count - 1] = energy
        
        // Calculate local energy average
        var localAverage: Float = 0
        let localHistorySize = min(24, energyHistory.count) // Look at ~0.5 seconds of history
        
        for i in (energyHistory.count - localHistorySize)..<energyHistory.count {
            localAverage += energyHistory[i]
        }
        localAverage /= Float(localHistorySize)
        
        // Calculate beat detection threshold
        let beatThreshold = localAverage * (1.0 + configuration.beatDetectionSensitivity)
        
        // Check if current energy exceeds threshold and enough time has passed since last beat
        let timeSinceLastBeat = time - lastBeatTime
        let minBeatInterval = 0.2 // Minimum time between beats (seconds)
        
        if energy > beatThreshold && timeSinceLastBeat > minBeatInterval {
            // Beat detected!
            let beatDetected = true
            
            // Record beat time
            if lastBeatTime > 0 {
                // Keep track of intervals between beats (for tempo estimation)
                beatIntervals.append(timeSinceLastBeat)
                if beatIntervals.count > 20 {
                    beatIntervals.removeFirst()
                }
            }
            
            lastBeatTime = time
            
            // Update beat detection status
            DispatchQueue.main.async {
                self.beatDetected = beatDetected
                
                // Auto-reset beat flag after a short delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    self.beatDetected = false
                }
            }
        }
    }
    
    // MARK: - Integration Methods
    /// Prepares the bridge with the AudioProcessor
    /// - Parameter audioProcessor: The audio processor to integrate with
    public func prepare(with audioProcessor: AudioEngine) {
        print("Preparing AudioVisualizerBridge with AudioEngine...")
        
        // Cancel any existing subscription to avoid memory leaks
        audioDataSubscription?.cancel()
        
        // Subscribe to the audio data from the engine
        subscribeToAudioEngine(audioProcessor)
        
        // Initialize or reinitialize processing filters based on current configuration
        initializeProcessingFilters()
        
        // Initialize frequency bands
        initializeFrequencyBands()
        
        // Reset state variables
        lastRawFrequencyData = []
        lastRawLevels = (0, 0)
        smoothedFrequencyData = []
        energyHistory = Array(repeating: 0, count: 43)
        lastBeatTime = 0
        beatIntervals = []
        
        // Reset beat detection state
        DispatchQueue.main.async {
            self.beatDetected = false
        }
        
        print("AudioVisualizerBridge preparation complete")
    }
    
    /// Cleans up resources used by the bridge
    public func cleanup() {
        print("Cleaning up AudioVisualizerBridge resources...")
        
        // Cancel subscriptions
        audioDataSubscription?.cancel()
        audioDataSubscription = nil
        
        // Clear data buffers
        lastRawFrequencyData = []
        smoothedFrequencyData = []
        energyHistory = []
        beatIntervals = []
        
        // Reset state
        DispatchQueue.main.async {
            self.beatDetected = false
            self.visualizationData = []
            self.smoothedLevels = (0, 0)
            self.frequencyBands = AudioFrequencyBands()
            self.energyLevel = 0.0
        }
        
        print("AudioVisualizerBridge cleanup complete")
    }
    
    /// Process neural insights from ML processing
    /// - Parameter neuralResponse: Neural processing response from ML analysis
    public func processNeuralInsights(_ neuralResponse: NeuralProcessingResponse) {
        // Apply neural insights to the visualization parameters based on weighting
        let weight = self.configuration.neuralProcessingWeight
        
        // Blend neural energy with signal-based energy
        self.energyLevel = self.energyLevel * (1.0 - weight) + neuralResponse.energy * weight
        
        // Apply beat detection if neural system detected it
        if neuralResponse.beatDetected && !self.beatDetected {
            DispatchQueue.main.async {
                self.beatDetected = true
                
                // Auto-reset beat flag after a short delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    self.beatDetected = false
                }
            }
        }
        
        // Additional visualization parameters could be adjusted here
    }
    }
    
    /// Sets the visualization resolution
    /// - Parameter resolution: The number of frequency points for visualization
    public func setVisualizationResolution(_ resolution: Int) {
        guard resolution > 0 && resolution != configuration.visualizationResolution else { return }
        
        configuration.visualizationResolution = resolution
        
        // Create new resampling filter
        resamplingFilter = ResamplingFilter(
            originalSize: AudioBloomCore.Constants.defaultFFTSize / 2,
            targetSize: resolution
        )
        
        // Process the last data with the new resolution
        if !lastRawFrequencyData.isEmpty {
            processAudioData(frequencyData: lastRawFrequencyData, levels: lastRawLevels)
        }
    }
    
    /// Applies spectral emphasis to enhance specific frequency ranges
    /// - Parameter emphasisRanges: Array of (startFreq, endFreq, gain) tuples
    public func applySpectralEmphasis(emphasisRanges: [(Float, Float, Float)]) {
        let fftSize = AudioBloomCore.Constants.defaultFFTSize / 2
        let sampleRate = Float(AudioBloomCore.Constants.defaultSampleRate)
        
        // Create an emphasis curve
        var emphasis = [Float](repeating: 1.0, count: fftSize)
        
        // Apply each emphasis range
        for (startFreq, endFreq, gain) in emphasisRanges {
            // Convert frequencies to bin indices
            let startBin = Int((startFreq / sampleRate) * Float(fftSize * 2))
            let endBin = Int((endFreq / sampleRate) * Float(fftSize * 2))
            
            // Apply gain to the range (with bounds checking)
            for i in max(0, startBin)..<min(fftSize, endBin) {
                emphasis[i] *= gain
            }
        }
        
        // Apply emphasis to the scaling factors
        if configuration.applySpectralWeighting && !fftScalingFactors.isEmpty {
            for i in 0..<min(fftSize, fftScalingFactors.count) {
                fftScalingFactors[i] *= emphasis[i]
            }
        } else {
            // If we're not already using spectral weighting, use emphasis directly
            fftScalingFactors = emphasis
            configuration.applySpectralWeighting = true
        }
        
        // Process the last data with the new emphasis
        if !lastRawFrequencyData.isEmpty {
            processAudioData(frequencyData: lastRawFrequencyData, levels: lastRawLevels)
        }
    }
    
    // MARK: - Private Methods
    
    /// Processes new audio data for visualization
    /// - Parameters:
    ///   - frequencyData: Raw frequency data from FFT
    ///   - levels: Audio levels (left, right)
    private func processAudioData(frequencyData: [Float], levels: (left: Float, right: Float)) {
        // Store raw data
        lastRawFrequencyData = frequencyData
        lastRawLevels = levels
        
        // Get current time for timing calculations

