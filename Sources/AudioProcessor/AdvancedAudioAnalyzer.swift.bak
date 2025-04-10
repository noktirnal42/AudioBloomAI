//
// AdvancedAudioAnalyzer.swift
// Advanced audio analysis system for AudioBloomAI
//

import Foundation
import Accelerate
import AVFoundation
import Combine
import CoreML
import AudioBloomCore
import AudioBloomCore.Audio
import Metal

/// Advanced audio analysis system that provides sophisticated audio feature extraction, 
/// beat pattern recognition, harmonic analysis, and mood detection
@available(macOS 15.0, *)
public class AdvancedAudioAnalyzer {
    // MARK: - Types
    
    /// Detected beat pattern type
    public enum BeatPatternType: String, CaseIterable, Identifiable {
        case regular = "Regular"
        case syncopated = "Syncopated"
        case complex = "Complex"
        case irregular = "Irregular"
        case sparse = "Sparse"
        case dense = "Dense"
        case polyrhythmic = "Polyrhythmic"
        case unknown = "Unknown"
        
        public var id: String { self.rawValue }
    }
    
    /// Detected music genre
    public enum GenreType: String, CaseIterable, Identifiable {
        case rock = "Rock"
        case electronic = "Electronic"
        case classical = "Classical"
        case jazz = "Jazz"
        case hiphop = "Hip Hop"
        case pop = "Pop"
        case ambient = "Ambient"
        case folk = "Folk"
        case metal = "Metal"
        case unknown = "Unknown"
        
        public var id: String { self.rawValue }
    }
    
    /// Detected mood type
    public enum MoodType: String, CaseIterable, Identifiable {
        case energetic = "Energetic"
        case relaxed = "Relaxed"
        case happy = "Happy"
        case melancholic = "Melancholic"
        case tense = "Tense"
        case calm = "Calm"
        case aggressive = "Aggressive"
        case peaceful = "Peaceful"
        case unknown = "Unknown"
        
        public var id: String { self.rawValue }
    }
    
    /// Detected harmony type (key, scale, etc.)
    public enum HarmonyType: String, CaseIterable, Identifiable {
        case major = "Major"
        case minor = "Minor"
        case diminished = "Diminished"
        case augmented = "Augmented"
        case suspended = "Suspended"
        case unknown = "Unknown"
        
        public var id: String { self.rawValue }
    }
    
    /// Detected chord type
    public enum ChordType: String, CaseIterable, Identifiable {
        case major = "Major"
        case minor = "Minor"
        case diminished = "Diminished"
        case augmented = "Augmented"
        case dominant7 = "Dominant 7th"
        case major7 = "Major 7th"
        case minor7 = "Minor 7th"
        case unknown = "Unknown"
        
        public var id: String { self.rawValue }
    }
    
    /// Information about a detected beat pattern
    public struct BeatPatternInfo {
        /// Beat pattern type
        public let patternType: BeatPatternType
        
        /// Tempo in beats per minute
        public let tempo: Double
        
        /// Beat strength (0.0-1.0)
        public let strength: Double
        
        /// Stability of rhythm (0.0-1.0)
        public let stability: Double
        
        /// Confidence in the detection (0.0-1.0)
        public let confidence: Double
        
        /// Timing offset relative to a reference (in seconds)
        public let offset: Double
        
        /// Beat interval in seconds
        public let interval: Double
        
        /// Beat subdivision type (e.g., 1/4, 1/8, etc.)
        public let subdivision: String
        
        /// Beat timestamps in seconds
        public let beatTimestamps: [Double]
    }
    
    /// Information about harmonic analysis
    public struct HarmonicInfo {
        /// Detected musical key (e.g., C, D#, etc.)
        public let key: String
        
        /// Detected scale type
        public let scaleType: HarmonyType
        
        /// Detected chord progression
        public let chordProgression: [String]
        
        /// Current chord
        public let currentChord: String
        
        /// Chord type
        public let chordType: ChordType
        
        /// Dominant frequencies in Hz
        public let dominantFrequencies: [Double]
        
        /// Harmonic tension level (0.0-1.0)
        public let harmonicTension: Double
        
        /// Key stability (0.0-1.0)
        public let keyStability: Double
        
        /// Confidence in the detection (0.0-1.0)
        public let confidence: Double
    }
    
    /// Information about genre classification
    public struct GenreInfo {
        /// Detected primary genre
        public let primaryGenre: GenreType
        
        /// Genre probabilities for all genres
        public let genreProbabilities: [GenreType: Double]
        
        /// Secondary genre influences
        public let secondaryGenres: [GenreType]
        
        /// Confidence in the classification (0.0-1.0)
        public let confidence: Double
        
        /// Genre stability over time (0.0-1.0)
        public let stability: Double
    }
    
    /// Information about detected mood
    public struct MoodInfo {
        /// Primary detected mood
        public let primaryMood: MoodType
        
        /// Energy level (0.0-1.0)
        public let energy: Double
        
        /// Positivity level (-1.0 to 1.0)
        public let valence: Double
        
        /// Emotional intensity (0.0-1.0)
        public let intensity: Double
        
        /// Mood stability over time (0.0-1.0)
        public let stability: Double
        
        /// Complexity of emotional content (0.0-1.0)
        public let complexity: Double
        
        /// Confidence in the detection (0.0-1.0)
        public let confidence: Double
    }
    
    /// Configuration options for advanced audio analysis
    public struct AdvancedAnalysisConfig {
        /// Enable beat pattern recognition
        public var enableBeatPatternRecognition: Bool = true
        
        /// Enable harmonic analysis
        public var enableHarmonicAnalysis: Bool = true
        
        /// Enable genre classification
        public var enableGenreClassification: Bool = true
        
        /// Enable mood detection
        public var enableMoodDetection: Bool = true
        
        /// Enable GPU acceleration
        public var useGPU: Bool = true
        
        /// History time window for pattern recognition (in seconds)
        public var historyDuration: Double = 10.0
        
        /// Analysis update interval (in seconds)
        public var updateInterval: Double = 0.1
        
        /// Confidence threshold for detection (0.0-1.0)
        public var confidenceThreshold: Double = 0.6
        
        /// Model paths for ML-based features
        public var modelPaths: [String: URL] = [:]
        
        /// Default configuration
        public static var standard: AdvancedAnalysisConfig {
            return AdvancedAnalysisConfig()
        }
        
        /// High accuracy configuration (more resource intensive)
        public static var highAccuracy: AdvancedAnalysisConfig {
            var config = AdvancedAnalysisConfig()
            config.historyDuration = 15.0
            config.updateInterval = 0.05
            config.confidenceThreshold = 0.7
            return config
        }
        
        /// Lightweight configuration (less resource intensive)
        public static var lightweight: AdvancedAnalysisConfig {
            var config = AdvancedAnalysisConfig()
            config.historyDuration = 5.0
            config.updateInterval = 0.2
            config.confidenceThreshold = 0.5
            return config
        }
    }
    
    // MARK: - Beat Pattern Recognition
    
    /// Beat detection history
    private struct BeatHistory {
        /// Beat onset times (in seconds)
        var onsetTimes: [Double] = []
        
        /// Beat intervals (in seconds)
        var intervals: [Double] = []
        
        /// Beat strengths (0.0-1.0)
        var strengths: [Double] = []
        
        /// Analysis start time
        var startTime: Double = 0
        
        /// Add a beat to history
        mutating func addBeat(timestamp: Double, strength: Double) {
            if onsetTimes.isEmpty {
                startTime = timestamp
                onsetTimes.append(timestamp)
                strengths.append(strength)
            } else if let lastBeat = onsetTimes.last {
                let interval = timestamp - lastBeat
                
                // Only add if reasonable interval (avoid duplicates)
                if interval > 0.1 {
                    onsetTimes.append(timestamp)
                    intervals.append(interval)
                    strengths.append(strength)
                }
            }
            
            // Trim history if needed
            trimHistory()
        }
        
        /// Trim history to configured window
        mutating func trimHistory(maxDuration: Double = 10.0) {
            guard let firstBeat = onsetTimes.first, let lastBeat = onsetTimes.last else { return }
            
            let totalDuration = lastBeat - firstBeat
            if totalDuration > maxDuration && onsetTimes.count > 4 {
                // Find index to trim
                var trimIndex = 0
                while trimIndex < onsetTimes.count && (lastBeat - onsetTimes[trimIndex]) > maxDuration {
                    trimIndex += 1
                }
                
                // Trim arrays
                if trimIndex > 0 {
                    onsetTimes.removeFirst(trimIndex)
                    strengths.removeFirst(trimIndex)
                    
                    if trimIndex <= intervals.count {
                        intervals.removeFirst(trimIndex)
                    }
                }
            }
        }
        
        /// Clear history
        mutating func clear() {
            onsetTimes.removeAll()
            intervals.removeAll()
            strengths.removeAll()
            startTime = 0
        }
        
        /// Calculate average interval
        func averageInterval() -> Double? {
            guard !intervals.isEmpty else { return nil }
            return intervals.reduce(0.0, +) / Double(intervals.count)
        }
        
        /// Calculate tempo in BPM
        func calculateTempo() -> Double? {
            guard let avgInterval = averageInterval(), avgInterval > 0 else { return nil }
            return 60.0 / avgInterval
        }
        
        /// Calculate interval variance (for stability measurement)
        func calculateVariance() -> Double {
            guard intervals.count > 1 else { return 0 }
            
            let mean = intervals.reduce(0, +) / Double(intervals.count)
            let variance = intervals.reduce(0.0) { sum, interval in
                let diff = interval - mean
                return sum + (diff * diff)
            } / Double(intervals.count)
            
            return variance
        }
        
        /// Calculate beat pattern stability
        func calculateStability() -> Double {
            guard !intervals.isEmpty else { return 0 }
            
            let variance = calculateVariance()
            let mean = intervals.reduce(0, +) / Double(intervals.count)
            
            // Calculate coefficient of variation (CV)
            let cv = sqrt(variance) / mean
            
            // Map CV to stability (lower CV = higher stability)
            // CV ~ 0     = perfect stability = 1.0
            // CV >= 0.5  = low stability    = 0.0
            return max(0.0, min(1.0, 1.0 - (cv * 2.0)))
        }
    }
    
    // MARK: - Properties
    
    /// The audio feature extractor providing base features
    public private(set) var featureExtractor: AudioFeatureExtractor
    
    /// Configuration options
    private var config: AdvancedAnalysisConfig
    
    /// Metal compute engine for GPU processing
    private var metalCompute: MetalComputeCore?
    
    /// Beat pattern recognition state
    private var beatHistory = BeatHistory()
    private var currentBeatPattern: BeatPatternInfo?
    private var lastBeatTimestamp: Double = 0
    private var beatPatternPublisher = PassthroughSubject<BeatPatternInfo, Never>()
    
    /// Harmonic analysis state
    private var currentHarmonic: HarmonicInfo?
    private var noteProbabilities: [String: Double] = [:]
    private var harmonicHistory: [HarmonicInfo] = []
    private var harmonicPublisher = PassthroughSubject<HarmonicInfo, Never>()
    
    /// Genre classification state
    private var currentGenre: GenreInfo?
    private var genreFeatureHistory: [[Double]] = []
    private var genrePublisher = PassthroughSubject<GenreInfo, Never>()
    
    /// Mood detection state
    private var currentMood: MoodInfo?
    private var moodHistory: [MoodInfo] = []
    private var moodPublisher = PassthroughSubject<MoodInfo, Never>()
    
    /// Machine learning models
    private var genreClassifier: MLModel?
    private var moodClassifier: MLModel?
    
    /// Processing queue
    private let processingQueue = DispatchQueue(label: "com.audiobloom.advanced-analysis", qos: .userInteractive)
    
    /// Analysis timers
    private var analysisTimer: Timer?
    
    /// Feature history for analysis
    private var featureHistory: [AudioFeatureType: [AudioFeatures]] = [:]
    
    // MARK: - Initialization
    
    /// Initialize with feature extractor and configuration
    /// - Parameters:
    ///   - featureExtractor: The audio feature extractor to use
    ///   - config: Analysis configuration options
    public init(featureExtractor: AudioFeatureExtractor, config: AdvancedAnalysisConfig = .standard) {
        self.featureExtractor = featureExtractor
        self.config = config
        
        // Setup GPU processing if requested
        if config.useGPU {
            setupGPUProcessing()
        }
        
        // Load ML models if paths provided
        loadModels()
        
        // Configure feature extractor to provide needed features
        configureFeatureExtractor()
    }
    
    /// Setup GPU processing with Metal
    private func setupGPUProcessing() {
        do {
            metalCompute = try MetalComputeCore(maxConcurrentOperations: 2)
        } catch {
            print("Warning: Could not initialize Metal compute: \(error.localizedDescription)")
        }
    }
    
    /// Load machine learning models
    private func loadModels() {
        // Load genre classifier if path provided
        if let genreModelURL = config.modelPaths["genre"] {
            do {
                genreClassifier = try MLModel(contentsOf: genreModelURL)
            } catch {
                print("Warning: Could not load genre classifier model: \(error.localizedDescription)")
            }
        }
        
        // Load mood classifier if path provided
        if let moodModelURL = config.modelPaths["mood"] {
            do {
                moodClassifier = try MLModel(contentsOf: moodModelURL)
            } catch {
                print("Warning: Could not load mood classifier model: \(error.localizedDescription)")
            }
        }
    }
    
    /// Configure the feature extractor to provide needed features
    private func configureFeatureExtractor() {
        // Determine required features based on enabled analysis types
        var requiredFeatures: Set<AudioFeatureType> = []
        
        // Beat pattern recognition requires tempo, onset, and beat features
        if config.enableBeatPatternRecognition {
            requiredFeatures.insert(.beat)
            requiredFeatures.insert(.onset)
            requiredFeatures.insert(.tempo)
            requiredFeatures.insert(.zeroCrossingRate)
            requiredFeatures.insert(.rmsEnergy)
        }
        
        // Harmonic analysis requires spectral and pitch features
        if config.enableHarmonicAnalysis {
            requiredFeatures.insert(.fftMagnitude)
            requiredFeatures.insert(.pitch)
            requiredFeatures.insert(.spectralCentroid)
            requiredFeatures.insert(.spectralFlatness)
            requiredFeatures.insert(.chroma)
        }
        
        // Genre and mood detection require a broad set of features
        if config.enableGenreClassification || config.enableMoodDetection {
            requiredFeatures.insert(.mfcc)
            requiredFeatures.insert(.spectralCentroid)
            requiredFeatures.insert(.spectralRolloff)
            requiredFeatures.insert(.spectralFlux)
            requiredFeatures.insert(.zeroCrossingRate)
            requiredFeatures.insert(.rmsEnergy)
            requiredFeatures.insert(.frequencyBandEnergy)
        }
        
        // Configure feature extractor if needed
        do {
            // Check if we need to update active features
            let currentFeatures = featureExtractor.activeFeatures
            if currentFeatures != requiredFeatures {
                try featureExtractor.setActiveFeatures(requiredFeatures)
                print("Audio feature extractor configured with \(requiredFeatures.count) features")
            }
        } catch {
            print("Warning: Could not configure feature extractor: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Public Control
    
    /// Start advanced audio analysis
    public func start() {
        // Start feature extractor if needed
        if !featureExtractor.isRunning {
            do {
                try featureExtractor.start()
            } catch {
                print("Error starting feature extractor: \(error.localizedDescription)")
                return
            }
        }
        
        // Setup feature subscription
        setupFeatureSubscription()
        
        // Start periodic analysis
        startAnalysisTimer()
    }
    
    /// Stop advanced audio analysis
    public func stop() {
        // Stop analysis timer
        stopAnalysisTimer()
        
        // Reset internal state
        resetAnalysisState()
    }
    
    /// Setup subscription to feature extractor
    private func setupFeatureSubscription() {
        // Ensure feature extractor delegate is set to self
        featureExtractor.delegate = self
    }
    
    /// Start periodic analysis timer
    private func startAnalysisTimer() {
        // Stop existing timer if any
        stopAnalysisTimer()
        
        // Create new timer on main thread
        DispatchQueue.main.async {
            self.analysisTimer = Timer.scheduledTimer(
                timeInterval: self.config.updateInterval,
                target: self,
                selector: #selector(self.performAnalysis),
                userInfo: nil,
                repeats: true
            )
        }
    }
    
    /// Stop analysis timer
    private func stopAnalysisTimer() {
        analysisTimer?.invalidate()
        analysisTimer = nil
    }
    
    /// Reset internal analysis state
    private func resetAnalysisState() {
        // Clear beat history
        beatHistory.clear()
        currentBeatPattern = nil
        lastBeatTimestamp = 0
        
        // Clear harmonic analysis state
        currentHarmonic = nil
        noteProbabilities.removeAll()
        harmonicHistory.removeAll()
        
        // Clear genre classification state
        currentGenre = nil
        genreFeatureHistory.removeAll()
        
        // Clear mood detection state
        currentMood = nil
        moodHistory.removeAll()
        
        // Clear feature history
        featureHistory.removeAll()
    }
    
    /// Perform periodic analysis
    @objc private func performAnalysis() {
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Process accumulated features
            self.processFeatureHistory()
        }
    }
    
    // MARK: - Feature Processing
    
    /// Process accumulated feature history
    private func processFeatureHistory() {
        // Analyze beat patterns if enabled
        if config.enableBeatPatternRecognition {
            analyzeBeatPattern()
        }
        
        // Analyze harmonics if enabled
        if config.enableHarmonicAnalysis {
            analyzeHarmonics()
        }
        
        // Analyze genre if enabled
        if config.enableGenreClassification {
            analyzeGenre()
        }
        
        // Analyze mood if enabled
        if config.enableMoodDetection {
            analyzeMood()
        }
    }
    
    // MARK: - Beat Pattern Analysis
    
    /// Analyze beat pattern from collected features
    private func analyzeBeatPattern() {
        // Process beat and onset features if available
        if let beatFeatures = getLatestFeatures(ofType: .beat, count: 10),
           !beatFeatures.isEmpty {
            
            // Extract timestamps and values
            for feature in beatFeatures {
                // Skip if no values
                guard let value = feature.values.first else { continue }
                
                // Check if beat detected
                let isBeatDetected = value > 0.5
                
                if isBeatDetected {
                    // Add to beat history
                    let strength = value
                    beatHistory.addBeat(timestamp: feature.timestamp, strength: Double(strength))
                    lastBeatTimestamp = feature.timestamp
                }
            }
            
            // Update beat pattern if we have enough history
            if beatHistory.onsetTimes.count >= 4 {
                updateBeatPattern()
            }
        }
    }
    
    /// Update beat pattern analysis
    private func updateBeatPattern() {
        // Ensure we have enough history
        guard beatHistory.intervals.count >= 3 else { return }
        
        // Calculate basic tempo stats
        guard let tempo = beatHistory.calculateTempo(),
              let interval = beatHistory.averageInterval() else {
            return
        }
        
        // Calculate stability
        let stability = beatHistory.calculateStability()
        
        // Calculate average strength
        let strength = beatHistory.strengths.reduce(0.0, +) / Double(beatHistory.strengths.count)
        
        // Determine beat pattern type
        let patternType = determineBeatPatternType(
            tempo: tempo,
            strength: strength,
            stability: stability,
            intervals: beatHistory.intervals
        )
        
        // Calculate confidence based on history size and stability
        let historyConfidence = min(1.0, Double(beatHistory.onsetTimes.count) / 16.0)
        let stabilityFactor = stability
        let confidence = (historyConfidence + stabilityFactor) / 2.0
        
        // Determine subdivision
        let subdivision = determineSubdivision(tempo: tempo, intervals: beatHistory.intervals)
        
        // Only update if confidence exceeds threshold
        if confidence >= config.confidenceThreshold {
            // Create beat pattern info
            let beatPatternInfo = BeatPatternInfo(
                patternType: patternType,
                tempo: tempo,
                strength: strength,
                stability: stability,
                confidence: confidence,
                offset: 0.0, // Calculate timing offset if needed
                interval: interval,
                subdivision: subdivision,
                beatTimestamps: beatHistory.onsetTimes
            )
            
            // Update current pattern and publish
            if currentBeatPattern == nil || confidence > currentBeatPattern!.confidence || 
               patternType != currentBeatPattern!.patternType {
                currentBeatPattern = beatPatternInfo
                beatPatternPublisher.send(beatPatternInfo)
            }
        }
    }
    
    /// Determine beat pattern type based on tempo and stability
    private func determineBeatPatternType(
        tempo: Double,
        strength: Double,
        stability: Double,
        intervals: [Double]
    ) -> BeatPatternType {
        // Simple algorithm to determine pattern type
        
        // Check for very unstable pattern
        if stability < 0.3 {
            return .irregular
        }
        
        // Check for very stable pattern
        if stability > 0.8 {
            // Check for syncopation
            if hasSyncopation(intervals: intervals) {
                return .syncopated
            }
            
            return .regular
        }
        
        // Check for sparse or dense patterns
        if tempo < 70 {
            return .sparse
        } else if tempo > 160 {
            return .dense
        }
        
        // Check for complex patterns
        if hasPolyrhythm(intervals: intervals) {
            return .polyrhythmic
        }
        
        if hasComplexPattern(intervals: intervals) {
            return .complex
        }
        
        // Default to regular
        return .regular
    }
    
    /// Determine if intervals show syncopation
    private func hasSyncopation(intervals: [Double]) -> Bool {
        guard intervals.count >= 4 else { return false }
        
        // Calculate average and threshold
        let avg = intervals.reduce(0, +) / Double(intervals.count)
        let threshold = avg * 0.25
        
        // Look for offset patterns (short-long patterns)
        var hasShortLongPattern = false
        for i in 0..<intervals.count-1 {
            let ratio = intervals[i] / intervals[i+1]
            if ratio < 0.75 || ratio > 1.33 {
                hasShortLongPattern = true
                break
            }
        }
        
        return hasShortLongPattern
    }
    
    /// Determine if intervals suggest polyrhythm
    private func hasPolyrhythm(intervals: [Double]) -> Bool {
        guard intervals.count >= 6 else { return false }
        
        // Look for repeating patterns of different lengths
        // This is a simplified detection - a real implementation would
        // use more sophisticated pattern recognition
        
        // Check for 3 against 2 polyrhythm
        let avg = intervals.reduce(0, +) / Double(intervals.count)
        
        var pattern3Count = 0
        var pattern2Count = 0
        
        for interval in intervals {
            let ratio = interval / avg
            if abs(ratio - 2.0/3.0) < 0.1 || abs(ratio - 4.0/3.0) < 0.1 {
                pattern3Count += 1
            } else if abs(ratio - 0.5) < 0.1 || abs(ratio - 1.0) < 0.1 || abs(ratio - 1.5) < 0.1 {
                pattern2Count += 1
        }
        
        // Check if we have a sufficient mix of patterns
        return pattern3Count >= 2 && pattern2Count >= 2
    }
    
    /// Determine if intervals show complex pattern
    private func hasComplexPattern(intervals: [Double]) -> Bool {
        guard intervals.count >= 5 else { return false }
        
        // Calculate average interval
        let avg = intervals.reduce(0, +) / Double(intervals.count)
        
        // Count interval variations
        var variationCount = 0
        for interval in intervals {
            // Check how far this interval is from the average
            let deviation = abs(interval - avg) / avg
            if deviation > 0.2 { // More than 20% different from average
                variationCount += 1
            }
        }
        
        // If more than 30% of intervals have significant deviation, it's complex
        let complexityThreshold = Double(intervals.count) * 0.3
        return Double(variationCount) >= complexityThreshold
    }
    
    /// Determine beat subdivision type
    private func determineSubdivision(tempo: Double, intervals: [Double]) -> String {
        // Calculate the division based on tempo and average interval
        
        // Common meter is 4/4, which gives us a reference point
        let quarterNoteInterval = 60.0 / tempo // Quarter note duration in seconds
        let avgInterval = intervals.reduce(0, +) / Double(intervals.count)
        
        // Calculate ratio of average interval to quarter note
        let ratio = avgInterval / quarterNoteInterval
        
        // Determine subdivision based on ratio
        if ratio > 1.9 && ratio < 2.1 {
            return "1/2" // Half note
        } else if ratio > 0.9 && ratio < 1.1 {
            return "1/4" // Quarter note
        } else if ratio > 0.4 && ratio < 0.6 {
            return "1/8" // Eighth note
        } else if ratio > 0.24 && ratio < 0.35 {
            return "1/16" // Sixteenth note
        } else if ratio > 0.15 && ratio < 0.23 {
            return "1/32" // Thirty-second note
        } else if ratio > 2.9 && ratio < 4.1 {
            return "1/1" // Whole note
        } else {
            return "Complex" // Non-standard subdivision
        }
    }
    
    // MARK: - Harmonic Analysis
    
    /// Analyze harmonic content from collected features
    private func analyzeHarmonics() {
        // Process pitch and chroma features for harmonic analysis
        if let pitchFeatures = getLatestFeatures(ofType: .pitch, count: 5),
           let chromaFeatures = getLatestFeatures(ofType: .chroma, count: 5),
           !pitchFeatures.isEmpty && !chromaFeatures.isEmpty {
            
            // Update note probabilities from chroma features
            updateNoteProbabilities(chromaFeatures: chromaFeatures)
            
            // Detect current key and scale
            let (key, scaleType, keyConfidence) = detectKey(from: noteProbabilities)
            
            // Detect current chord
            let (chord, chordType, chordConfidence) = detectChord(from: noteProbabilities)
            
            // Extract dominant frequencies from pitch features
            var dominantFrequencies: [Double] = []
            for feature in pitchFeatures {
                if let frequency = feature.values.first,
                   let confidence = feature.metadata["confidence"] as? Float,
                   confidence > Float(config.confidenceThreshold) {
                    dominantFrequencies.append(Double(frequency))
                }
            }
            
            // Calculate harmonic stability and tension
            let (keyStability, harmonicTension) = calculateHarmonicStability(
                noteProbabilities: noteProbabilities,
                key: key,
                chord: chord
            )
            
            // Update chord progression if key remains stable
            var chordProgression: [String] = []
            if keyStability > 0.7, let currentHarmonic = currentHarmonic {
                chordProgression = currentHarmonic.chordProgression
                
                // Only add new chord if it's different from last one
                if chordProgression.isEmpty || chordProgression.last != chord {
                    // Keep last 8 chords only
                    if chordProgression.count >= 8 {
                        chordProgression.removeFirst()
                    }
                    chordProgression.append(chord)
                }
            } else if keyStability > 0.7 {
                chordProgression = [chord]
            }
            
            // Calculate overall confidence
            let confidence = (keyConfidence + chordConfidence) / 2.0
            
            // Only update if confidence exceeds threshold
            if confidence >= config.confidenceThreshold {
                // Create harmonic info
                let harmonicInfo = HarmonicInfo(
                    key: key,
                    scaleType: scaleType,
                    chordProgression: chordProgression,
                    currentChord: chord,
                    chordType: chordType,
                    dominantFrequencies: dominantFrequencies,
                    harmonicTension: harmonicTension,
                    keyStability: keyStability,
                    confidence: confidence
                )
                
                // Update current harmonic and publish
                currentHarmonic = harmonicInfo
                harmonicPublisher.send(harmonicInfo)
                
                // Add to history
                harmonicHistory.append(harmonicInfo)
                if harmonicHistory.count > 20 {
                    harmonicHistory.removeFirst()
                }
            }
        }
    }
    
    /// Update note probabilities from chroma features
    private func updateNoteProbabilities(chromaFeatures: [AudioFeatures]) {
        // Initialize if needed
        if noteProbabilities.isEmpty {
            let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            for note in noteNames {
                noteProbabilities[note] = 0.0
            }
        }
        
        // Apply exponential decay to existing probabilities
        let decayFactor = 0.8
        for (note, probability) in noteProbabilities {
            noteProbabilities[note] = probability * decayFactor
        }
        
        // Update with new chroma features
        for feature in chromaFeatures {
            let values = feature.values
            if values.count == 12 {
                let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                
                for (i, value) in values.enumerated() {
                    if i < noteNames.count {
                        let note = noteNames[i]
                        noteProbabilities[note] = noteProbabilities[note]! + Double(value)
                    }
                }
            }
        }
        
        // Normalize probabilities
        let sum = noteProbabilities.values.reduce(0.0, +)
        if sum > 0 {
            for (note, probability) in noteProbabilities {
                noteProbabilities[note] = probability / sum
            }
        }
    }
    
    /// Detect musical key from note probabilities
    private func detectKey(from noteProbabilities: [String: Double]) -> (String, HarmonyType, Double) {
        // Major and minor key profiles (Krumhansl-Schmuckler key-finding algorithm)
        let majorProfile: [Double] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        let minorProfile: [Double] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        let noteOrder = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        
        // Calculate correlation with each possible key
        var bestKey = ""
        var bestScale = HarmonyType.major
        var highestCorrelation = -1.0
        
        for startNote in 0..<12 {
            // Test major key
            var majorCorrelation = 0.0
            var minorCorrelation = 0.0
            
            for i in 0..<12 {
                let noteIndex = (startNote + i) % 12
                let note = noteOrder[noteIndex]
                let probability = noteProbabilities[note] ?? 0.0
                
                majorCorrelation += probability * majorProfile[i]
                minorCorrelation += probability * minorProfile[i]
            }
            
            // Update if better correlation found
            if majorCorrelation > highestCorrelation {
                highestCorrelation = majorCorrelation
                bestKey = noteOrder[startNote]
                bestScale = .major
            }
            
            if minorCorrelation > highestCorrelation {
                highestCorrelation = minorCorrelation
                bestKey = noteOrder[startNote]
                bestScale = .minor
            }
        }
        
        // Add "m" suffix for minor keys in display
        let displayKey = bestKey + (bestScale == .minor ? "m" : "")
        
        // Calculate confidence based on correlation strength
        let confidence = min(1.0, max(0.0, highestCorrelation / 10.0))
        
        return (displayKey, bestScale, confidence)
    }
    
    /// Detect chord from note probabilities
    private func detectChord(from noteProbabilities: [String: Double]) -> (String, ChordType, Double) {
        // Define chord templates
        let chordTemplates: [ChordType: [Int]] = [
            .major: [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], // Major (root, major third, perfect fifth)
            .minor: [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], // Minor (root, minor third, perfect fifth)
            .diminished: [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], // Diminished (root, minor third, diminished fifth)
            .augmented: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], // Augmented (root, major third, augmented fifth)
            .dominant7: [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], // Dominant 7th (root, major third, perfect fifth, minor seventh)
            .major7: [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], // Major 7th (root, major third, perfect fifth, major seventh)
            .minor7: [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]  // Minor 7th (root, minor third, perfect fifth, minor seventh)
        ]
        
        let noteOrder = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        
        // Prepare note vector from probabilities
        var noteVector: [Double] = Array(repeating: 0.0, count: 12)
        for (note, probability) in noteProbabilities {
            if let index = noteOrder.firstIndex(of: note) {
                noteVector[index] = probability
            }
        }
        
        // Find best matching chord
        var bestRoot = 0
        var bestChordType = ChordType.unknown
        var bestScore = -1.0
        
        for rootNote in 0..<12 {
            for (chordType, template) in chordTemplates {
                var score = 0.0
                
                // Calculate match score for this chord template at this root
                for i in 0..<12 {
                    let noteIndex = (rootNote + i) % 12
                    let templateValue = Double(template[i])
