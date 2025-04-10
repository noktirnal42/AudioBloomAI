import Foundation
import Combine
import CoreML
import Numerics
import Logging
import AudioBloomCore

/// Errors that can occur during output transformation
public enum OutputTransformationError: Error {
    case invalidInputData
    case unsupportedOutputFormat
    case transformationFailed(String)
    case invalidConfiguration
    
    var localizedDescription: String {
        switch self {
        case .invalidInputData:
            return "Input data format is invalid"
        case .unsupportedOutputFormat:
            return "The requested output format is not supported"
        case .transformationFailed(let message):
            return "Transformation failed: \(message)"
        case .invalidConfiguration:
            return "Invalid transformer configuration"
        }
    }
}

/// Output format for visualization data
public enum VisualizationFormat {
    /// Linear frequency spectrum (simple bar graph)
    case linearSpectrum
    /// Logarithmic frequency spectrum (emphasizes lower frequencies)
    case logSpectrum
    /// Mel-scale frequency spectrum (perceptual)
    case melSpectrum
    /// Circular visualization (radar/polar plot)
    case circularSpectrum
    /// Beat-synchronized pattern
    case beatPattern
    /// Waveform visualization
    case waveform
    /// 3D visualization data
    case spatial3D
    /// Custom format with specific parameters
    case custom(parameters: [String: Any])
}

/// Types of audio features that can be transformed
public enum AudioFeatureType {
    /// Frequency spectrum data
    case frequencySpectrum
    /// Beat and tempo data
    case rhythmicFeatures
    /// Tonal and harmonic features
    case harmonicFeatures
    /// Combined feature set
    case combined
}

/// Configuration for output transformation
public struct TransformationConfiguration {
    /// The visualization format to use
    public var format: VisualizationFormat = .linearSpectrum
    
    /// The number of output data points
    public var outputSize: Int = 64
    
    /// Whether to normalize output to 0-1 range
    public var normalize: Bool = true
    
    /// Amount of smoothing to apply (0-1)
    public var smoothingFactor: Float = 0.3
    
    /// Whether to use logarithmic scaling
    public var useLogScale: Bool = false
    
    /// Custom transformation parameters
    public var parameters: [String: Any] = [:]
    
    /// Create a default configuration
    public static func defaultConfiguration() -> TransformationConfiguration {
        return TransformationConfiguration()
    }
    
    /// Create a spectrum visualization configuration
    public static func spectrumConfiguration(size: Int = 64, smoothing: Float = 0.3) -> TransformationConfiguration {
        var config = TransformationConfiguration()
        config.format = .linearSpectrum
        config.outputSize = size
        config.smoothingFactor = smoothing
        return config
    }
    
    /// Create a beat visualization configuration
    public static func beatConfiguration() -> TransformationConfiguration {
        var config = TransformationConfiguration()
        config.format = .beatPattern
        config.parameters["beatSensitivity"] = 0.7
        config.parameters["beatDecay"] = 0.8
        return config
    }
}

/// Transformed output data for visualization
public struct VisualizationData {
    /// The primary output data array
    public let values: [Float]
    
    /// Secondary or auxiliary data values
    public let auxiliaryValues: [Float]?
    
    /// Metadata about the transformation
    public let metadata: [String: Any]
    
    /// The timestamp of the data
    public let timestamp: TimeInterval
    
    /// Whether this data represents a significant event (like a beat)
    public let isSignificantEvent: Bool
    
    /// Creates visualization data
    public init(
        values: [Float],
        auxiliaryValues: [Float]? = nil,
        metadata: [String: Any] = [:],
        timestamp: TimeInterval = CACurrentMediaTime(),
        isSignificantEvent: Bool = false
    ) {
        self.values = values
        self.auxiliaryValues = auxiliaryValues
        self.metadata = metadata
        self.timestamp = timestamp
        self.isSignificantEvent = isSignificantEvent
    }
    
    /// Returns a subset of the visualization data
    public func subset(range: Range<Int>) -> VisualizationData {
        let subsetValues = Array(values[range])
        let subsetAuxiliary = auxiliaryValues.map { Array($0[range]) }
        
        return VisualizationData(
            values: subsetValues,
            auxiliaryValues: subsetAuxiliary,
            metadata: metadata,
            timestamp: timestamp,
            isSignificantEvent: isSignificantEvent
        )
    }
}

/// Protocol for objects that can receive visualization data
public protocol VisualizationDataReceiver {
    /// Called when new visualization data is available
    func didReceiveVisualizationData(_ data: VisualizationData, forFeature feature: AudioFeatureType)
}

/// Transforms ML model outputs into visualization-ready data
public final class OutputTransformer {
    /// Logger for this class
    private let logger = Logger(label: "com.audiobloom.outputtransformer")
    
    /// Current configuration
    private var configuration: TransformationConfiguration
    
    /// Previously processed data for smoothing
    private var previousOutputs: [AudioFeatureType: [Float]] = [:]
    
    /// High-performance data buffer for frequency processing
    private let frequencyBuffer: UnsafeMutableBufferPointer<Float>
    
    /// Subject for publishing transformed frequency data
    private let frequencyDataSubject = PassthroughSubject<VisualizationData, Never>()
    
    /// Subject for publishing transformed beat data
    private let beatDataSubject = PassthroughSubject<VisualizationData, Never>()
    
    /// Publisher for frequency visualization data
    public var frequencyDataPublisher: AnyPublisher<VisualizationData, Never> {
        frequencyDataSubject.eraseToAnyPublisher()
    }
    
    /// Publisher for beat visualization data
    public var beatDataPublisher: AnyPublisher<VisualizationData, Never> {
        beatDataSubject.eraseToAnyPublisher()
    }
    
    /// Delegate to receive visualization data
    public weak var delegate: VisualizationDataReceiver?
    
    /// Whether to use high-performance transformation
    public var useHighPerformanceMode: Bool = true
    
    /// Creates a new output transformer with the specified configuration
    /// - Parameter configuration: The transformation configuration
    public init(configuration: TransformationConfiguration = .defaultConfiguration()) {
        self.configuration = configuration
        
        // Allocate a reusable buffer for high-performance processing
        // This avoids repeated allocations during real-time processing
        self.frequencyBuffer = UnsafeMutableBufferPointer.allocate(capacity: 1024)
        frequencyBuffer.initialize(repeating: 0.0)
        
        logger.info("OutputTransformer initialized with format: \(configuration.format)")
    }
    
    deinit {
        // Free the allocated buffer
        frequencyBuffer.deallocate()
    }
    
    /// Updates the transformation configuration
    /// - Parameter configuration: The new configuration
    public func updateConfiguration(_ configuration: TransformationConfiguration) {
        self.configuration = configuration
        logger.debug("Updated transformation configuration")
        
        // Clear previous outputs as they may no longer be compatible
        previousOutputs.removeAll()
    }
    
    /// Transforms audio features into visualization data
    /// - Parameters:
    ///   - features: The audio features to transform
    ///   - featureType: The type of features being transformed
    /// - Returns: Transformed visualization data
    /// - Throws: OutputTransformationError if transformation fails
    public func transform(
        features: AudioFeatures,
        type featureType: AudioFeatureType
    ) throws -> VisualizationData {
        switch featureType {
        case .frequencySpectrum:
            return try transformFrequencySpectrum(features.frequencySpectrum)
        case .rhythmicFeatures:
            return try transformRhythmicFeatures(
                tempo: features.tempo,
                beatConfidence: features.beatConfidence,
                beatDetected: features.beatDetected
            )
        case .harmonicFeatures:
            // For demonstration, we'll use a simulated harmonic transformation
            return try transformSimulatedHarmonicFeatures(from: features)
        case .combined:
            // Combine multiple feature transformations
            return try transformCombinedFeatures(features)
        }
    }
    
    /// Transforms raw frequency spectrum data
    /// - Parameter spectrum: The frequency spectrum data
    /// - Returns: Visualization data for the frequency spectrum
    /// - Throws: OutputTransformationError if transformation fails
    public func transformFrequencySpectrum(_ spectrum: [Float]) throws -> VisualizationData {
        guard !spectrum.isEmpty else {
            throw OutputTransformationError.invalidInputData
        }
        
        // Get configured output size or use input size if larger
        let outputSize = min(configuration.outputSize, spectrum.count)
        
        // High-performance path for real-time processing
        if useHighPerformanceMode {
            return try highPerformanceTransformFrequencySpectrum(
                spectrum,
                outputSize: outputSize
            )
        }
        
        // Standard path (more flexible but slightly slower)
        var outputValues = resizeArray(spectrum, newSize: outputSize)
        
        // Apply log scaling if configured
        if configuration.useLogScale {
            outputValues = applyLogScaling(outputValues)
        }
        
        // Normalize if configured
        if configuration.normalize {
            outputValues = normalizeArray(outputValues)
        }
        
        // Apply smoothing if configured
        if configuration.smoothingFactor > 0 {
            outputValues = applySmoothing(
                outputValues,
                featureType: .frequencySpectrum,
                factor: configuration.smoothingFactor
            )
        }
        
        // Create metadata
        let metadata: [String: Any] = [
            "format": configuration.format,
            "sampleCount": spectrum.count,
            "minValue": outputValues.min() ?? 0,
            "maxValue": outputValues.max() ?? 0,
            "processedAt": Date()
        ]
        
        // Create and return visualization data
        let visualizationData = VisualizationData(
            values: outputValues,
            auxiliaryValues: nil,
            metadata: metadata,
            timestamp: CACurrentMediaTime(),
            isSignificantEvent: false
        )
        
        // Publish the data to subscribers
        frequencyDataSubject.send(visualizationData)
        
        // Notify delegate
        delegate?.didReceiveVisualizationData(visualizationData, forFeature: .frequencySpectrum)
        
        return visualizationData
    }
    
    /// High-performance transformation for frequency spectrum data
    /// - Parameters:
    ///   - spectrum: The frequency spectrum data
    ///   - outputSize: The desired output size
    /// - Returns: Visualization data for the frequency spectrum
    /// - Throws: OutputTransformationError if transformation fails
    private func highPerformanceTransformFrequencySpectrum(
        _ spectrum: [Float],
        outputSize: Int
    ) throws -> VisualizationData {
        guard outputSize <= frequencyBuffer.count else {
            throw OutputTransformationError.invalidConfiguration
        }
        
        // Get previous values for smoothing
        let previousValues = previousOutputs[.frequencySpectrum] ?? Array(repeating: 0, count: outputSize)
        
        // Resize spectrum to output size
        let resizeFactor = Float(spectrum.count) / Float(outputSize)
        var minValue: Float = 0
        var maxValue: Float = 0
        
        // Direct buffer manipulation for performance
        for i in 0..<outputSize {
            let sourceIndex = Int(Float(i) * resizeFactor)
            var value = spectrum[min(sourceIndex, spectrum.count - 1)]
            
            // Apply log scaling if configured
            if configuration.useLogScale && value > 0 {
                value = log10(value * 9 + 1)
            }
            
            // Apply smoothing if configured
            if configuration.smoothingFactor > 0 && i < previousValues.count {
                value = value * (1 - configuration.smoothingFactor) + previousValues[i] * configuration.smoothingFactor
            }
            
            frequencyBuffer[i] = value
            
            // Track min/max for normalization
            if i == 0 || value < minValue { minValue = value }
            if i == 0 || value > maxValue { maxValue = value }
        }
        
        // Normalize if configured
        if configuration.normalize && maxValue > minValue {
            let range = maxValue - minValue
            for i in 0..<outputSize {
                frequencyBuffer[i] = (frequencyBuffer[i] - minValue) / range
            }
        }
        
        // Convert back to array for return
        let outputValues = Array(frequencyBuffer[0..<outputSize])
        
        // Store for next smoothing operation
        previousOutputs[.frequencySpectrum] = outputValues
        
        // Create metadata
        let metadata: [String: Any] = [
            "format": configuration.format,
            "sampleCount": spectrum.count,
            "minValue": minValue,
            "maxValue": maxValue,
            "processedAt": Date()
        ]
        
        // Create and return visualization data
        return VisualizationData(
            values: outputValues,
            auxiliaryValues: nil,
            metadata: metadata,
            timestamp: CACurrentMediaTime(),
            isSignificantEvent: false
        )
    }
    
    /// Transforms rhythmic features (tempo, beats) into visualization data
    /// - Parameters:
    ///   - tempo: The detected tempo in BPM
    ///   - beatConfidence: Confidence level of beat detection (0-1)
    ///   - beatDetected: Whether a beat was detected
    /// - Returns: Visualization data for rhythmic features
    /// - Throws: OutputTransformationError if transformation fails
    public func transformRhythmicFeatures(
        tempo: Float,
        beatConfidence: Float,
        beatDetected: Bool
    ) throws -> VisualizationData {
        // Normalize tempo to a reasonable range (40-200 BPM is common for music)
        let normalizedTempo = max(0, min(1, (tempo - 40) / 160))
        
        // Create a beat pattern based on tempo and confidence
        var beatPattern = [Float](repeating: 0, count: configuration.outputSize)
        
        // Generate a beat visualization pattern
        if beatDetected {
            // Create a decay pattern from the beat
            for i in 0..<configuration.outputSize {
                let normalizedPosition = Float(i) / Float(configuration.outputSize)
                let decayFactor = max(0, 1 - normalizedPosition)
                beatPattern[i] = beatConfidence * decayFactor
            }
        } else {
            // Use previous pattern with decay if no new beat
            if let previousPattern = previousOutputs[.rhythmicFeatures] {
                let beatDecay = configuration.parameters["beatDecay"] as? Float ?? 0.8
                for i in 0..<min(configuration.outputSize, previousPattern.count) {
                    beatPattern[i] = previousPattern[i] * beatDecay
                }
            }
        }
        
        // Create auxiliary data with tempo information
        let auxiliaryData = [normalizedTempo]
        
        // Store for next time
        previousOutputs[.rhythmicFeatures] = beatPattern
        
        // Create metadata
        let metadata: [String: Any] = [
            "format": configuration.format,
            "tempo": tempo,
            "beatConfidence": beatConfidence,
            "beatDetected": beatDetected,
            "processedAt": Date()
        ]
        
        // Create visualization data
        let visualizationData = VisualizationData(
            values: beatPattern,
            auxiliaryValues: auxiliaryData,
            metadata: metadata,
            timestamp: CACurrentMediaTime(),
            isSignificantEvent: beatDetected
        )
        
        // Publish to subscribers
        beatDataSubject.send(visualizationData)
        
        // Notify delegate
        delegate?.didReceiveVisualizationData(visualizationData, forFeature: .rhythmicFeatures)
        
        return visualizationData
    }
    
    /// Transforms simulated harmonic features into visualization data
    /// - Parameter features: The audio features to transform
    /// - Returns: Visualization data for the harmonic features
    /// - Throws: OutputTransformationError if transformation fails
    public func transformSimulatedHarmonicFeatures(from features: AudioFeatures) throws -> VisualizationData {
        guard !features.frequencySpectrum.isEmpty else {
            throw OutputTransformationError.invalidInputData
        }
        
        // In a real implementation, this would extract harmonic features
        // For this example, we'll simulate harmonic content based on frequency data
        
        // Create simulated harmonic values
        var harmonicValues = [Float](repeating: 0, count: configuration.outputSize)
        
        // Use frequency data to simulate harmonic content
        if !features.frequencySpectrum.isEmpty {
            let spectrum = features.frequencySpectrum
            
            // Simple algorithm to extract "harmonic" content
            // In a real implementation, this would use actual harmonic analysis
            for i in 0..<min(configuration.outputSize, spectrum.count / 4) {
                // Look at specific frequency bands that might contain harmonic information
                let baseIndex = i * 4
                let maxHarmonic = min(baseIndex + 3, spectrum.count - 1)
                
                // Calculate harmonic relationship (simplified)
                var harmonicValue: Float = 0
                for j in 0...3 {
                    let index = baseIndex + j
                    if index < spectrum.count {
                        // Weight higher harmonics differently
                        harmonicValue += spectrum[index] * (1.0 - Float(j) * 0.2)
                    }
                }
                
                harmonicValues[i] = harmonicValue / 4.0
            }
        }
        
        // Normalize if configured
        if configuration.normalize {
            harmonicValues = normalizeArray(harmonicValues)
        }
        
        // Apply smoothing if configured
        if configuration.smoothingFactor > 0 {
            harmonicValues = applySmoothing(
                harmonicValues,
                featureType: .harmonicFeatures,
                factor: configuration.smoothingFactor
            )
        }
        
        // Create metadata
        let metadata: [String: Any] = [
            "format": configuration.format,
            "type": "harmonicFeatures",
            "processedAt": Date()
        ]
        
        // Create and return visualization data
        return VisualizationData(
            values: harmonicValues,
            auxiliaryValues: nil,
            metadata: metadata,
            timestamp: CACurrentMediaTime(),
            isSignificantEvent: false
        )
    }
    
    /// Transforms combined audio features into unified visualization data
    /// - Parameter features: The audio features to transform
    /// - Returns: Visualization data combining multiple feature types
    /// - Throws: OutputTransformationError if transformation fails
    public func transformCombinedFeatures(_ features: AudioFeatures) throws -> VisualizationData {
        // Create a combined visualization from multiple feature types
        // For example, we might want frequency data modulated by beat information
        
        // Get the frequency data
        let frequencyData = try transformFrequencySpectrum(features.frequencySpectrum)
        
        // Get rhythmic data
        let rhythmicData = try transformRhythmicFeatures(
            tempo: features.tempo,
            beatConfidence: features.beatConfidence,
            beatDetected: features.beatDetected
        )
        
        // Combine the data in a meaningful way
        var combinedValues = [Float](repeating: 0, count: configuration.outputSize)
        
        // Example: Modulate frequency data with beat information
        let beatEmphasis = features.beatDetected ? 1.5 : 1.0
        let tempoFactor = features.tempo / 120.0 // Normalize around typical 120BPM
        
        for i in 0..<min(configuration.outputSize, frequencyData.values.count) {
            // Apply beat and tempo effects to frequency data
            let frequencyValue = frequencyData.values[i]
            
            // Modulate by beat (simple example)
            var value = frequencyValue * Float(beatEmphasis)
            
            // Apply tempo-based effect (simple example)
            if i % 2 == 0 {
                value *= Float(max(0.8, min(1.2, tempoFactor)))
            }
            
            combinedValues[i] = value
        }
        
        // Normalize the combined output
        combinedValues = normalizeArray(combinedValues)
        
        // Combine metadata from both feature types
        var combinedMetadata = frequencyData.metadata
        for (key, value) in rhythmicData.metadata {
            combinedMetadata["rhythmic_\(key)"] = value
        }
        combinedMetadata["combinedType"] = "frequency+rhythm"
        
        // Create and return combined visualization data
        return VisualizationData(
            values: combinedValues,
            auxiliaryValues: rhythmicData.values,
            metadata: combinedMetadata,
            timestamp: CACurrentMediaTime(),
            isSignificantEvent: rhythmicData.isSignificantEvent
        )
    }
    
    // MARK: - Utility Methods
    
    /// Resizes an array to a new size using linear interpolation
    /// - Parameters:
    ///   - array: The array to resize
    ///   - newSize: The new size
    /// - Returns: The resized array
    private func resizeArray(_ array: [Float], newSize: Int) -> [Float] {
        guard !array.isEmpty, newSize > 0 else { return [] }
        
        // If sizes match, return as is
        if array.count == newSize {
            return array
        }
        
        var result = [Float](repeating: 0, count: newSize)
        let scaleFactor = Float(array.count - 1) / Float(max(1, newSize - 1))
        
        for i in 0..<newSize {
            let exactIndex = Float(i) * scaleFactor
            let index = Int(exactIndex)
            let nextIndex = min(index + 1, array.count - 1)
            let fraction = exactIndex - Float(index)
            
            // Linear interpolation
            result[i] = array[index] * (1 - fraction) + array[nextIndex] * fraction
        }
        
        return result
    }
    
    /// Normalizes an array to the range 0-1
    /// - Parameter array: The array to normalize
    /// - Returns: The normalized array
    private func normalizeArray(_ array: [Float]) -> [Float] {
        guard !array.isEmpty else { return [] }
        
        // Find min and max values
        var minValue = array[0]
        var maxValue = array[0]
        
        for value in array {
            if value < minValue { minValue = value }
            if value > maxValue { maxValue = value }
        }
        
        // If range is zero, return uniform array
        if maxValue == minValue {
            return array.map { _ in 0.5 }
        }
        
        // Normalize to 0-1 range
        let range = maxValue - minValue
        return array.map { ($0 - minValue) / range }
    }
    
    /// Applies logarithmic scaling to an array
    /// - Parameter array: The array to scale
    /// - Returns: The log-scaled array
    private func applyLogScaling(_ array: [Float]) -> [Float] {
        guard !array.isEmpty else { return [] }
        
        // Apply log scaling: log10(value * 9 + 1)
        // This maps 0 -> 0 and 1 -> 1, with a logarithmic curve in between
        return array.map { value in
            value > 0 ? log10(value * 9 + 1) : 0
        }
    }
    
    /// Applies smoothing to an array using previous values
    /// - Parameters:
    ///   - array: The array to smooth
    ///   - featureType: The feature type (for retrieving previous values)
    ///   - factor: The smoothing factor (0-1)
    /// - Returns: The smoothed array
    private func applySmoothing(_ array: [Float], featureType: AudioFeatureType, factor: Float) -> [Float] {
        guard !array.isEmpty, factor > 0 else { return array }
        
        // Get previous values for this feature type
        guard let previousValues = previousOutputs[featureType], !previousValues.isEmpty else {
            // Store current values for next time
            previousOutputs[featureType] = array
            return array
        }
        
        // Apply exponential smoothing
        var result = [Float](repeating: 0, count: array.count)
        let previousCount = previousValues.count
        
        for i in 0..<array.count {
            if i < previousCount {
                // Use previous value for smoothing
                result[i] = array[i] * (1 - factor) + previousValues[i] * factor
            } else {
                // No previous value available
                result[i] = array[i]
            }
        }
        
        // Store current output for next time
        previousOutputs[featureType] = result
        
        return result
    }
}
