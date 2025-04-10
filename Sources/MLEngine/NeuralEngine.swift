// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency
import Foundation
import CoreML
import Combine
import Accelerate
import CreateML
import AudioBloomCore

/// Neural Engine for detecting patterns, beats, and emotional content in audio
@available(macOS 15.0, *)
public class NeuralEngine: ObservableObject, MLProcessing {
    /// Published ML model output data
    @Published public private(set) var outputData: [Float] = []
    
    /// Published beat detection data
    @Published public private(set) var beatDetected: Bool = false
    
    /// Published emotional content analysis
    @Published public private(set) var emotionalContent: EmotionalContent = .neutral
    
    /// Published pattern recognition data
    @Published public private(set) var patternType: PatternType = .none
    
    /// Published beat confidence (0-1)
    @Published public private(set) var beatConfidence: Float = 0.0
    
    /// Processing queue for ML operations
    private let processingQueue = DispatchQueue(label: "com.audiobloom.neuralprocessing", qos: .userInteractive)
    
    /// Beat detector subsystem
    private var beatDetector: BeatDetector
    
    /// Emotional content analyzer
    private var emotionalAnalyzer: EmotionalAnalyzer
    
    /// Pattern recognition system
    private var patternRecognizer: PatternRecognizer
    
    /// Flag indicating if the ML models are ready
    private var areModelsReady = false
    
    /// Audio history buffer for pattern analysis
    private var audioHistoryBuffer: AudioHistoryBuffer
    
    /// Audio data subscriber
    private var audioDataSubscription: AnyCancellable?
    
    /// Beat detection threshold
    private var beatThreshold: Float = 0.5
    
    /// Neural Engine configuration
    private var configuration: NeuralEngineConfiguration
    
    /// Initializes a new NeuralEngine
    public init(configuration: NeuralEngineConfiguration = NeuralEngineConfiguration()) {
        self.configuration = configuration
        self.beatDetector = BeatDetector(sensitivity: configuration.beatSensitivity)
        self.emotionalAnalyzer = EmotionalAnalyzer()
        self.patternRecognizer = PatternRecognizer()
        self.audioHistoryBuffer = AudioHistoryBuffer(
            duration: configuration.patternHistoryDuration,
            sampleRate: Float(AudioBloomCore.Constants.defaultSampleRate),
            fftSize: AudioBloomCore.Constants.defaultFFTSize
        )
    }
    
    /// Subscribes to audio data updates from an AudioDataPublisher
    public func subscribeToAudioData(_ publisher: AudioDataPublisher) {
        audioDataSubscription = publisher.publisher
            .sink { [weak self] (frequencyData, levels) in
                Task {
                    await self?.processAudioData(frequencyData)
                }
            }
    }
    
    /// Prepares the ML models for processing
    public func prepareMLModel() {
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Initialize and prepare all ML components
            self.beatDetector.prepare()
            self.emotionalAnalyzer.prepare()
            self.patternRecognizer.prepare()
            
            // Update the ready state on the main thread
            Task { @MainActor in
                self.areModelsReady = true
                print("Neural Engine models ready")
            }
        }
    }
    
    /// Processes audio data through the Neural Engine
    public func processAudioData(_ audioData: [Float]) async {
        guard areModelsReady else { return }
        
        // Add to history buffer
        audioHistoryBuffer.addFrame(audioData)
        
        // Process with each subsystem
        let beatResult = await beatDetector.detectBeat(in: audioData)
        let emotionResult = await emotionalAnalyzer.analyzeEmotion(in: audioData)
        let patternResult = await patternRecognizer.recognizePattern(in: audioHistoryBuffer.getBuffer())
        
        // Create neural output features vector for visualization
        let neuralOutput = createOutputVector(
            beatResult: beatResult,
            emotionResult: emotionResult,
            patternResult: patternResult
        )
        
        // Update published properties on the main thread
        await MainActor.run {
            self.beatDetected = beatResult.isDetected
            self.beatConfidence = beatResult.confidence
            self.emotionalContent = emotionResult.emotion
            self.patternType = patternResult.pattern
            self.outputData = neuralOutput
        }
    }
    
    /// Creates a combined output vector from all neural subsystems
    private func createOutputVector(
        beatResult: BeatDetectionResult,
        emotionResult: EmotionalAnalysisResult,
        patternResult: PatternRecognitionResult
    ) -> [Float] {
        // Create a feature vector combining insights from all three subsystems
        // This will be used by the visualizer to enhance the visuals
        
        var output = [Float](repeating: 0, count: 32)
        
        // Beat features (0-7)
        output[0] = beatResult.isDetected ? 1.0 : 0.0
        output[1] = beatResult.confidence
        output[2] = beatResult.intensity
        output[3] = beatResult.periodicity
        
        // Emotional content features (8-15)
        output[8] = emotionResult.energy
        output[9] = emotionResult.pleasantness
        output[10] = emotionResult.complexity
        output[11] = emotionResult.density
        
        // Fill in emotion type one-hot encoding
        let emotionIndex = 12 + (emotionResult.emotion.rawValue % 4)
        output[emotionIndex] = 1.0
        
        // Pattern features (16-31)
        output[16] = patternResult.strength
        output[17] = patternResult.consistency
        output[18] = patternResult.complexity
        output[19] = patternResult.repetition
        
        // Fill in pattern type one-hot encoding
        let patternIndex = 20 + (patternResult.pattern.rawValue % 12)
        if patternResult.pattern != .none {
            output[patternIndex] = 1.0
        }
        
        return output
    }
    
    /// Updates the neural engine configuration
    public func updateConfiguration(_ configuration: NeuralEngineConfiguration) {
        self.configuration = configuration
        self.beatDetector.sensitivity = configuration.beatSensitivity
        self.audioHistoryBuffer.resize(
            duration: configuration.patternHistoryDuration,
            sampleRate: Float(AudioBloomCore.Constants.defaultSampleRate)
        )
    }
}

/// Configuration for the Neural Engine
@available(macOS 15.0, *)
public struct NeuralEngineConfiguration: Sendable {
    /// Beat detection sensitivity (0.0-1.0)
    public var beatSensitivity: Float = 0.65
    
    /// Pattern history duration in seconds
    public var patternHistoryDuration: Float = 5.0
    
    /// Emotional analysis sensitivity (0.0-1.0)
    public var emotionalSensitivity: Float = 0.5
    
    /// Pattern recognition sensitivity (0.0-1.0)
    public var patternSensitivity: Float = 0.7
    
    /// Default initializer
    public init() {}
    
    /// Custom initializer
    public init(
        beatSensitivity: Float = 0.65,
        patternHistoryDuration: Float = 5.0,
        emotionalSensitivity: Float = 0.5,
        patternSensitivity: Float = 0.7
    ) {
        self.beatSensitivity = beatSensitivity
        self.patternHistoryDuration = patternHistoryDuration
        self.emotionalSensitivity = emotionalSensitivity
        self.patternSensitivity = patternSensitivity
    }
}

// MARK: - Emotional Content Types

/// Types of emotional content in audio
@available(macOS 15.0, *)
public enum EmotionalContent: Int, CaseIterable {
    case energetic = 0
    case calm = 1
    case tense = 2
    case relaxed = 3
    case neutral = 4
    
    /// User-friendly description
    public var description: String {
        switch self {
        case .energetic: return "Energetic"
        case .calm: return "Calm"
        case .tense: return "Tense"
        case .relaxed: return "Relaxed"
        case .neutral: return "Neutral"
        }
    }
}

// MARK: - Pattern Types

/// Types of patterns in audio
@available(macOS 15.0, *)
public enum PatternType: Int, CaseIterable {
    case none = 0
    case buildUp = 1
    case breakdown = 2
    case steady = 3
    case crescendo = 4
    case decrescendo = 5
    case rhythmic = 6
    case melodic = 7
    case harmonic = 8
    case dissonant = 9
    case repetitive = 10
    case random = 11
    case complex = 12
    
    /// User-friendly description
    public var description: String {
        switch self {
        case .none: return "None"
        case .buildUp: return "Build Up"
        case .breakdown: return "Breakdown"
        case .steady: return "Steady"
        case .crescendo: return "Crescendo"
        case .decrescendo: return "Decrescendo"
        case .rhythmic: return "Rhythmic"
        case .melodic: return "Melodic"
        case .harmonic: return "Harmonic"
        case .dissonant: return "Dissonant"
        case .repetitive: return "Repetitive"
        case .random: return "Random"
        case .complex: return "Complex"
        }
    }
}

// MARK: - Beat Detector

/// Subsystem for detecting beats in audio data
@available(macOS 15.0, *)
class BeatDetector {
    /// Beat detection sensitivity (0.0-1.0)
    var sensitivity: Float
    
    /// Energy history for beat detection
    private var energyHistory: [Float] = []
    
    /// History window size
    private let historySize = 43 // ~1 second at 60fps with some buffer
    
    /// Dynamic threshold for beat detection
    private var dynamicThreshold: Float = 0.0
    
    /// Last beat time
    private var lastBeatTime: TimeInterval = 0
    
    /// Minimum time between beats in seconds
    private let minBeatInterval: TimeInterval = 0.2 // 300 BPM max
    
    /// Initialize with sensitivity
    init(sensitivity: Float) {
        self.sensitivity = sensitivity
        self.energyHistory = [Float](repeating: 0, count: historySize)
    }
    
    /// Prepare the beat detector
    func prepare() {
        // Reset energy history
        energyHistory = [Float](repeating: 0, count: historySize)
        dynamicThreshold = 0.0
        lastBeatTime = 0
    }
    
    /// Detect beats in the audio data
    func detectBeat(in audioData: [Float]) async -> BeatDetectionResult {
        // Calculate energy in the bass/low-mid range (first ~quarter of the spectrum)
        let bassRange = 0..<min(audioData.count / 4, audioData.count)
        var energy: Float = 0
        
        for i in bassRange {
            energy += audioData[i] * audioData[i] // Square for energy
        }
        energy = energy / Float(bassRange.count)
        
        // Update energy history
        energyHistory.removeFirst()
        energyHistory.append(energy)
        
        // Calculate the local average (excluding the current energy)
        let localHistory = energyHistory.dropLast()
        let localAverage = localHistory.reduce(0, +) / Float(localHistory.count)
        
        // Update dynamic threshold with smoothing
        dynamicThreshold = dynamicThreshold * 0.9 + localAverage * 0.1
        
        // Apply sensitivity to threshold
        let adjustedThreshold = dynamicThreshold * (1.5 - sensitivity)
        
        // Check for beat based on current energy exceeding the threshold
        let currentTime = Date().timeIntervalSince1970
        let timeSinceLastBeat = currentTime - lastBeatTime
        
        var isDetected = false
        var confidence: Float = 0.0
        
        if energy > adjustedThreshold && timeSinceLastBeat >= minBeatInterval {
            // Calculate confidence based on how much the energy exceeds the threshold
            confidence = min((energy - adjustedThreshold) / adjustedThreshold, 1.0)
            
            if confidence > 0.3 { // Minimum confidence threshold
                isDetected = true
                lastBeatTime = currentTime
            }
        }
        
        // Calculate additional beat metrics
        let intensity = min(energy / (dynamicThreshold + 0.01), 2.0) // Scale to reasonable range
        
        // Detect periodicity (regular beats)
        let periodicity = detectPeriodicity()
        
        return BeatDetectionResult(
            isDetected: isDetected,
            confidence: confidence,
            intensity: intensity,
            periodicity: periodicity
        )
    }
    
    /// Detect periodicity in the energy history
    private func detectPeriodicity() -> Float {
        // Simple autocorrelation for detecting periodicity
        // This is a simplified version - a real implementation would use more sophisticated algorithms
        
        guard energyHistory.count > 10 else { return 0 }
        
        var maxCorrelation: Float = 0
        let correlationRange = 3..<min(20, energyHistory.count / 2)
        
        for lag in correlationRange {
            var correlation: Float = 0
            var count = 0
            
            for i in 0..<(energyHistory.count - lag) {
                correlation += energyHistory[i] * energyHistory[i + lag]
                count += 1
            }
            
            correlation /= Float(count)
            
            if correlation > maxCorrelation {
                maxCorrelation = correlation
            }
        }
        
        // Normalize
        let baselineEnergy = energyHistory.reduce(0, +) / Float(energyHistory.count)
        let normalizedCorrelation = maxCorrelation / (baselineEnergy * baselineEnergy + 0.0001)
        
        return min(normalizedCorrelation, 1.0)
    }
}

/// Result of beat detection
@available(macOS 15.0, *)
struct BeatDetectionResult: Sendable {
    /// Whether a beat was detected
    let isDetected: Bool
    
    /// Confidence level (0-1)
    let confidence: Float
    
    /// Intensity of the beat (relative to recent history)
    let intensity: Float
    
    /// Periodicity - how regular the beats are (0-1)
    let periodicity: Float
}

// MARK: - Emotional Analyzer

/// Subsystem for analyzing emotional content in audio
@available(macOS 15.0, *)
class EmotionalAnalyzer {
    /// Emotion classifier model
    private var emotionModel: EmotionClassifierModel?
    
    /// Running average of frequency data for stability
    private var runningAverage: [Float] = []
    
    /// Initialize the emotion analyzer
    init() {
        runningAverage = [Float](repeating: 0, count: 32)
    }
    
/// Prepare the emotion analyzer
func prepare() {
    // In a production app, we would load a CoreML model
    // For now, we'll use a simulated model that extracts features
    emotionModel = EmotionClassifierModel()
}

/// Analyze emotional content in audio data
func analyzeEmotion(in audioData: [Float]) async -> EmotionalAnalysisResult {
    // Detect emotional features from frequency distribution
    
    // Update running average for stability (simple low-pass filter)
    for i in 0..<min(audioData.count, runningAverage.count) {
        runningAverage[i] = runningAverage[i] * 0.7 + audioData[i] * 0.3
    }
    
    // Extract spectral features
    let spectralCentroid = calculateSpectralCentroid(from: runningAverage)
    let spectralFlatness = calculateSpectralFlatness(from: runningAverage)
    let spectralRolloff = calculateSpectralRolloff(from: runningAverage)
    let spectralFlux = calculateSpectralFlux(from: runningAverage, and: audioData)
    
    // Categorize into spectral parameters
    let energy = calculateEnergy(from: audioData)
    
    // Calculate pleasantness (high for harmonic content, low for noise)
    let pleasantness = 1.0 - spectralFlatness
    
    // Calculate complexity (high for varied frequency content)
    let complexity = spectralRolloff * spectralFlux
    
    // Calculate density (how "full" the spectrum is)
    let density = calculateDensity(from: audioData)
    
    // Determine the primary emotion using our model
    let emotion = determineEmotion(
        energy: energy,
        pleasantness: pleasantness,
        complexity: complexity,
        density: density
    )
    
    return EmotionalAnalysisResult(
        emotion: emotion,
        energy: energy,
        pleasantness: pleasantness,
        complexity: complexity,
        density: density
    )
}

/// Calculate spectral centroid (brightness)
private func calculateSpectralCentroid(from spectrum: [Float]) -> Float {
    var numerator: Float = 0
    var denominator: Float = 0
    
    for i in 0..<spectrum.count {
        let frequency = Float(i) / Float(spectrum.count - 1)
        numerator += frequency * spectrum[i]
        denominator += spectrum[i]
    }
    
    guard denominator > 0 else { return 0.5 }
    return numerator / denominator
}

/// Calculate spectral flatness (noise vs. tonal)
private func calculateSpectralFlatness(from spectrum: [Float]) -> Float {
    // Avoid log(0) errors
    let nonZeroSpectrum = spectrum.map { max($0, 1e-10) }
    
    let geometricMean = exp(nonZeroSpectrum.map { log($0) }.reduce(0, +) / Float(nonZeroSpectrum.count))
    let arithmeticMean = nonZeroSpectrum.reduce(0, +) / Float(nonZeroSpectrum.count)
    
    guard arithmeticMean > 0 else { return 0 }
    return geometricMean / arithmeticMean
}

/// Calculate spectral rolloff (frequency below which X% of the spectrum is concentrated)
private func calculateSpectralRolloff(from spectrum: [Float], percentile: Float = 0.85) -> Float {
    let totalEnergy = spectrum.reduce(0, +)
    let rolloffThreshold = totalEnergy * percentile
    
    var cumulativeEnergy: Float = 0
    for i in 0..<spectrum.count {
        cumulativeEnergy += spectrum[i]
        if cumulativeEnergy >= rolloffThreshold {
            return Float(i) / Float(spectrum.count - 1)
        }
    }
    
    return 1.0
}

/// Calculate spectral flux (change in spectrum over time)
private func calculateSpectralFlux(from currentSpectrum: [Float], and previousSpectrum: [Float]) -> Float {
    var sum: Float = 0
    let count = min(currentSpectrum.count, previousSpectrum.count)
    
    for i in 0..<count {
        let diff = currentSpectrum[i] - previousSpectrum[i]
        sum += diff * diff
    }
    
    return sqrt(sum / Float(count))
}

/// Calculate energy from the spectrum
private func calculateEnergy(from spectrum: [Float]) -> Float {
    let sum = spectrum.reduce(0) { $0 + ($1 * $1) }
    return min(sqrt(sum / Float(spectrum.count)) * 3.0, 1.0) // Scale for 0-1 range
}

/// Calculate spectral density
private func calculateDensity(from spectrum: [Float]) -> Float {
    let nonZeroCount = spectrum.filter { $0 > 0.1 }.count
    return Float(nonZeroCount) / Float(spectrum.count)
}

/// Determine the emotional content based on spectral features
private func determineEmotion(
    energy: Float,
    pleasantness: Float,
    complexity: Float,
    density: Float
) -> EmotionalContent {
    // Map features to emotion types based on Russell's circumplex model of affect
    // Energy corresponds to arousal (vertical axis)
    // Pleasantness corresponds to valence (horizontal axis)
    
    // Simple mapping based on the quadrants of the circumplex model
    if energy > 0.6 {
        if pleasantness > 0.5 {
            return .energetic    // High energy, high pleasantness
        } else {
            return .tense        // High energy, low pleasantness
        }
    } else {
        if pleasantness > 0.5 {
            return .relaxed      // Low energy, high pleasantness
        } else {
            return .calm         // Low energy, low pleasantness
        }
    }
    
    // Default case if the above logic doesn't trigger
    return .neutral
}
}

/// Simulated emotion classifier model
@available(macOS 15.0, *)
private class EmotionClassifierModel {
    /// Initialize with default parameters
    init() {
        // In a real implementation, this would load a Core ML model
    }
}

/// Result of emotional analysis
@available(macOS 15.0, *)
struct EmotionalAnalysisResult: Sendable {
    /// Detected emotion
    let emotion: EmotionalContent
    
    /// Energy level (0-1)
    let energy: Float
    
    /// Pleasantness level (0-1)
    let pleasantness: Float
    
    /// Complexity level (0-1)
    let complexity: Float
    
    /// Density level (0-1)
    let density: Float
}

// MARK: - Pattern Recognizer

/// Subsystem for recognizing patterns in audio sequences
@available(macOS 15.0, *)
class PatternRecognizer {
    /// Pattern detection threshold
    private var threshold: Float = 0.3
    
    /// Feature memory for pattern detection
    private var featureMemory: [[Float]] = []
    
    /// History window length
    private let historyLength = 120 // ~2 seconds at 60fps
    
    /// Feature vector size
    private let featureSize = 12
    
    /// Initialize the pattern recognizer
    init() {
        featureMemory = [[Float]](repeating: [Float](repeating: 0, count: featureSize), count: historyLength)
    }
    
    /// Prepare the pattern recognizer
    func prepare() {
        // Reset feature memory
        featureMemory = [[Float]](repeating: [Float](repeating: 0, count: featureSize), count: historyLength)
    }
    
    /// Recognize patterns in the audio buffer
    func recognizePattern(in audioBuffer: AudioBuffer) async -> PatternRecognitionResult {
        // Extract a feature vector from the audio buffer
        let currentFeatures = extractFeatures(from: audioBuffer)
        
        // Update feature memory
        featureMemory.removeFirst()
        featureMemory.append(currentFeatures)
        
        // Analyze for different pattern types
        let (pattern, strength) = detectPatternType()
        let consistency = calculateConsistency()
        let complexity = calculateComplexity()
        let repetition = detectRepetition()
        
        return PatternRecognitionResult(
            pattern: pattern,
            strength: strength,
            consistency: consistency,
            complexity: complexity,
            repetition: repetition
        )
    }
    
    /// Extract features from the audio buffer
    private func extractFeatures(from audioBuffer: AudioBuffer) -> [Float] {
        var features = [Float](repeating: 0, count: featureSize)
        
        // These would be more sophisticated in a real implementation
        // For now, we'll use simple statistical measures
        
        // Use the most recent frame for instantaneous features
        if let latestFrame = audioBuffer.frames.last {
            // Simple feature extraction - divide spectrum into bins
            let binSize = latestFrame.count / (featureSize - 2)
            
            for i in 0..<(featureSize - 2) {
                let startIdx = i * binSize
                let endIdx = min(startIdx + binSize, latestFrame.count)
                let binRange = startIdx..<endIdx
                
                // Average energy in this frequency bin
                let binEnergy = binRange.map { latestFrame[$0] }.reduce(0, +) / Float(binRange.count)
                features[i] = binEnergy
            }
            
            // Add additional features
            features[featureSize - 2] = latestFrame.reduce(0, +) / Float(latestFrame.count) // Average energy
            features[featureSize - 1] = latestFrame.max() ?? 0 // Peak energy
        }
        
        return features
    }
    
    /// Detect the type of pattern in the audio
    private func detectPatternType() -> (PatternType, Float) {
        // Calculate trend over the past several frames
        let trendLength = min(featureMemory.count, 30) // Look at ~0.5 second
        guard trendLength > 5 else { return (.none, 0) }
        
        let recentFeatures = Array(featureMemory.suffix(trendLength))
        
        // Calculate average features at start and end of window
        let startAvg = averageFeatures(recentFeatures.prefix(5))
        let endAvg = averageFeatures(recentFeatures.suffix(5))
        
        // Calculate overall energy change
        let startEnergy = startAvg.reduce(0, +)
        let endEnergy = endAvg.reduce(0, +)
        let energyChange = (endEnergy - startEnergy) / max(startEnergy, 0.001)
        
        // Calculate spectrum shape change
        var shapeChange: Float = 0
        for i in 0..<min(startAvg.count, endAvg.count) {
            let normalizedStart = startAvg[i] / max(startEnergy, 0.001)
            let normalizedEnd = endAvg[i] / max(endEnergy, 0.001)
            shapeChange += abs(normalizedEnd - normalizedStart)
        }
        shapeChange /= Float(startAvg.count)
        
        // Detect rhythmic patterns
        let isRhythmic = detectRhythmicPattern()
        
        // Detect melodic patterns
        let isMelodic = detectMelodicPattern()
        
        // Determine pattern type based on energy and shape changes
        var patternType: PatternType = .none
        var strength: Float = 0
        
        if isRhythmic > 0.7 {
            patternType = .rhythmic
            strength = isRhythmic
        } else if isMelodic > 0.7 {
            patternType = .melodic
            strength = isMelodic
        } else if energyChange > 0.3 {
            // Energy increasing significantly
            if shapeChange > 0.2 {
                patternType = .buildUp
            } else {
                patternType = .crescendo
            }
            strength = min(abs(energyChange), 1.0)
        } else if energyChange < -0.3 {
            // Energy decreasing significantly
            if shapeChange > 0.2 {
                patternType = .breakdown
            } else {
                patternType = .decrescendo
            }
            strength = min(abs(energyChange), 1.0)
        } else if shapeChange < 0.1 && abs(energyChange) < 0.2 {
            // Minimal changes in both energy and spectrum shape
            patternType = .steady
            strength = 1.0 - shapeChange
        } else if shapeChange > 0.4 {
            // Significant spectrum shape changes
            patternType = .complex
            strength = min(shapeChange, 1.0)
        }
        
        return (patternType, strength)
    }
    
    /// Calculate the average of multiple feature vectors
    private func averageFeatures(_ features: ArraySlice<[Float]>) -> [Float] {
        guard !features.isEmpty, let firstFeature = features.first else {
            return [Float](repeating: 0, count: featureSize)
        }
        
        var result = [Float](repeating: 0, count: firstFeature.count)
        
        for feature in features {
            for i in 0..<min(result.count, feature.count) {
                result[i] += feature[i]
            }
        }
        
        for i in 0..<result.count {
            result[i] /= Float(features.count)
        }
        
        return result
    }
    
    /// Detect rhythmic patterns in the audio
    private func detectRhythmicPattern() -> Float {
        // In a real implementation, we would analyze for regular energy fluctuations
        // This would use autocorrelation or other signal processing techniques
        
        // For this example, we'll use a simplified approach
        guard featureMemory.count > 10 else { return 0 }
        
        // Extract energy values from each frame
        let energyValues = featureMemory.map { $0.last ?? 0 }
        
        // Calculate autocorrelation to find periodicity
        var maxCorrelation: Float = 0
        let correlationRange = 3..<min(20, energyValues.count / 2)
        
        for lag in correlationRange {
            var correlation: Float = 0
            var count = 0
            
            for i in 0..<(energyValues.count - lag) {
                correlation += energyValues[i] * energyValues[i + lag]
                count += 1
            }
            
            correlation /= Float(count)
            
            if correlation > maxCorrelation {
                maxCorrelation = correlation
            }
        }
        
        // Normalize
        let baselineEnergy = energyValues.reduce(0, +) / Float(energyValues.count)
        let normalizedCorrelation = maxCorrelation / (baselineEnergy * baselineEnergy + 0.0001)
        
        return min(normalizedCorrelation, 1

