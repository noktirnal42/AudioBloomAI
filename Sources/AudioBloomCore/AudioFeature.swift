// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency

import Foundation

@available(macOS 15.0, *)
public struct AudioFeature: Identifiable, Hashable, Sendable {
    public let id = UUID()
    public let type: AudioFeatureType
    public let value: Double
    public let timestamp: Date
    
    public init(type: AudioFeatureType, value: Double, timestamp: Date = Date()) {
        self.type = type
        self.value = value
        self.timestamp = timestamp
    }
    
    // Helper to create a batch of features
    public static func batch(features: [AudioFeature]) -> [AudioFeature] {
        return features
    }
}

@available(macOS 15.0, *)
public enum AudioFeatureType: String, CaseIterable, Hashable, Sendable {
    case dominantFrequency = "Dominant Frequency"
    case beatStrength = "Beat Strength"
    case tempoEstimate = "Tempo Estimate"
    case spectralCentroid = "Spectral Centroid"
    case spectralFlux = "Spectral Flux"
    
    // Add more feature types as needed
    
    // Display units for each feature type
    public var unit: String {
        switch self {
        case .dominantFrequency:
            return "Hz"
        case .beatStrength:
            return "%"
        case .tempoEstimate:
            return "BPM"
        case .spectralCentroid:
            return "Hz"
        case .spectralFlux:
            return ""
        }
    }
    
    // Description for each feature type
    public var description: String {
        switch self {
        case .dominantFrequency:
            return "The most prominent frequency in the audio signal"
        case .beatStrength:
            return "Intensity of detected beats"
        case .tempoEstimate:
            return "Estimated tempo of the audio"
        case .spectralCentroid:
            return "Weighted mean of the frequencies present in the signal"
        case .spectralFlux:
            return "Measure of how quickly the power spectrum is changing"
        }
    }
}

