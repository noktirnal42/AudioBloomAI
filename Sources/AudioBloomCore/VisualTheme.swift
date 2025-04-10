// Swift 6 optimized implementation
// Requires macOS 15.0 or later
// Updated for modern concurrency
//
// VisualTheme.swift
// Visual theme definitions for AudioBloomAI
//

import Foundation
import SwiftUI

public extension AudioBloomCore {
    /// Visual theme options for the application
    @available(macOS 15.0, *)
    enum VisualTheme: String, CaseIterable, Identifiable, Codable, Equatable, Sendable {
        /// Classic blue/purple theme
        case classic = "Classic"
        
        /// Vibrant neon colors
        case neon = "Neon"
        
        /// Monochromatic grayscale theme
        case monochrome = "Monochrome"
        
        /// Deep space cosmic theme
        case cosmic = "Cosmic"
        
        /// Nature-inspired green theme
        case nature = "Nature"
        
        /// Dark minimal theme
        case minimal = "Minimal"
        
        /// Ocean depths theme
        case ocean = "Ocean"
        
        /// Fire and ember theme
        case fire = "Fire"
        
        /// Unique identifier for SwiftUI integration
        public var id: String { self.rawValue }
        
        /// Returns color parameters for the theme
        public var colorParameters: [String: Any] {
            switch self {
            case .classic:
                return [
                    "primaryColor": [0.0, 0.5, 1.0, 1.0],
                    "secondaryColor": [1.0, 0.0, 0.5, 1.0],
                    "backgroundColor": [0.0, 0.0, 0.1, 1.0],
                    "accentColor": [0.2, 0.8, 1.0, 1.0],
                    "textColor": [1.0, 1.0, 1.0, 1.0],
                    "visualizationStyle": "gradient",
                    "particleCount": 1000
                ]
            case .neon:
                return [
                    "primaryColor": [0.0, 1.0, 0.8, 1.0],
                    "secondaryColor": [1.0, 0.0, 1.0, 1.0],
                    "backgroundColor": [0.05, 0.0, 0.05, 1.0],
                    "accentColor": [1.0, 0.8, 0.0, 1.0],
                    "textColor": [0.9, 1.0, 0.9, 1.0],
                    "visualizationStyle": "glow",
                    "particleCount": 1500
                ]
            case .monochrome:
                return [
                    "primaryColor": [0.9, 0.9, 0.9, 1.0],
                    "secondaryColor": [0.6, 0.6, 0.6, 1.0],
                    "backgroundColor": [0.1, 0.1, 0.1, 1.0],
                    "accentColor": [0.8, 0.8, 0.8, 1.0],
                    "textColor": [1.0, 1.0, 1.0, 1.0],
                    "visualizationStyle": "minimal",
                    "particleCount": 800
                ]
            case .cosmic:
                return [
                    "primaryColor": [0.5, 0.0, 0.8, 1.0],
                    "secondaryColor": [0.0, 0.8, 0.8, 1.0],
                    "backgroundColor": [0.0, 0.0, 0.2, 1.0],
                    "accentColor": [0.8, 0.4, 0.1, 1.0],
                    "textColor": [0.9, 0.8, 1.0, 1.0],
                    "visualizationStyle": "particle",
                    "particleCount": 2000
                ]
            case .nature:
                return [
                    "primaryColor": [0.2, 0.8, 0.2, 1.0],
                    "secondaryColor": [0.8, 0.9, 0.3, 1.0],
                    "backgroundColor": [0.05, 0.1, 0.05, 1.0],
                    "accentColor": [0.0, 0.6, 0.3, 1.0],
                    "textColor": [0.9, 1.0, 0.9, 1.0],
                    "visualizationStyle": "organic",
                    "particleCount": 1200
                ]
            case .minimal:
                return [
                    "primaryColor": [0.8, 0.8, 0.8, 1.0],
                    "secondaryColor": [0.6, 0.6, 0.6, 1.0],
                    "backgroundColor": [0.0, 0.0, 0.0, 1.0],
                    "accentColor": [0.7, 0.7, 0.7, 1.0],
                    "textColor": [1.0, 1.0, 1.0, 1.0],
                    "visualizationStyle": "line",
                    "particleCount": 500
                ]
            case .ocean:
                return [
                    "primaryColor": [0.0, 0.4, 0.8, 1.0],
                    "secondaryColor": [0.0, 0.6, 0.5, 1.0],
                    "backgroundColor": [0.0, 0.1, 0.2, 1.0],
                    "accentColor": [0.0, 0.8, 0.8, 1.0],
                    "textColor": [0.8, 0.9, 1.0, 1.0],
                    "visualizationStyle": "wave",
                    "particleCount": 1500
                ]
            case .fire:
                return [
                    "primaryColor": [1.0, 0.4, 0.0, 1.0],
                    "secondaryColor": [1.0, 0.8, 0.0, 1.0],
                    "backgroundColor": [0.1, 0.0, 0.0, 1.0],
                    "accentColor": [0.9, 0.2, 0.0, 1.0],
                    "textColor": [1.0, 0.9, 0.8, 1.0],
                    "visualizationStyle": "flame",
                    "particleCount": 1800
                ]
            }
        }
        
        /// Returns SwiftUI colors for the theme
        public var swiftUIColors: ThemeColors {
            let params = colorParameters
            
            // Extract RGBA components for primary color
            let primaryRGBA = params["primaryColor"] as? [CGFloat] ?? [0, 0, 0, 1]
            let secondaryRGBA = params["secondaryColor"] as? [CGFloat] ?? [0, 0, 0, 1]
            let backgroundRGBA = params["backgroundColor"] as? [CGFloat] ?? [0, 0, 0, 1]
            let accentRGBA = params["accentColor"] as? [CGFloat] ?? [0, 0, 0, 1]
            let textRGBA = params["textColor"] as? [CGFloat] ?? [1, 1, 1, 1]
            
            return ThemeColors(
                primary: Color(
                    red: primaryRGBA[0],
                    green: primaryRGBA[1],
                    blue: primaryRGBA[2],
                    opacity: primaryRGBA[3]
                ),
                secondary: Color(
                    red: secondaryRGBA[0],
                    green: secondaryRGBA[1],
                    blue: secondaryRGBA[2],
                    opacity: secondaryRGBA[3]
                ),
                background: Color(
                    red: backgroundRGBA[0],
                    green: backgroundRGBA[1],
                    blue: backgroundRGBA[2],
                    opacity: backgroundRGBA[3]
                ),
                accent: Color(
                    red: accentRGBA[0],
                    green: accentRGBA[1],
                    blue: accentRGBA[2],
                    opacity: accentRGBA[3]
                ),
                text: Color(
                    red: textRGBA[0],
                    green: textRGBA[1],
                    blue: textRGBA[2],
                    opacity: textRGBA[3]
                )
            )
        }
        
        /// Returns a descriptive name for the theme
        public var displayName: String {
            rawValue
        }
        
        /// Returns a description of the theme
        public var description: String {
            switch self {
            case .classic:
                return "A timeless blue and purple theme"
            case .neon:
                return "Vibrant neon colors for high energy visualizations"
            case .monochrome:
                return "Clean black and white aesthetic"
            case .cosmic:
                return "Deep space-inspired colors and effects"
            case .nature:
                return "Organic green tones inspired by nature"
            case .minimal:
                return "Minimalist design with subtle elements"
            case .ocean:
                return "Deep blues of the ocean depths"
            case .fire:
                return "Warm reds and oranges of fire and ember"
            }
        }
        
        /// Returns the visualization style for the theme
        public var visualizationStyle: String {
            colorParameters["visualizationStyle"] as? String ?? "gradient"
        }
        
        /// Recommended particle count for particle-based visualizations
        public var particleCount: Int {
            colorParameters["particleCount"] as? Int ?? 1000
        }
        
        /// Returns a gradient for use in the UI
        public var gradient: Gradient {
            let colors = swiftUIColors
            return Gradient(colors: [colors.primary, colors.secondary])
        }
        
        /// Returns a linear gradient for use in the UI
        public var linearGradient: LinearGradient {
            LinearGradient(
                gradient: gradient,
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        }
    }
    
    /// SwiftUI color collection for a theme
    @available(macOS 15.0, *)
    struct ThemeColors: Sendable: Equatable {
        /// Primary color
        public let primary: Color
        
        /// Secondary color 
        public let secondary: Color
        
        /// Background color
        public let background: Color
        
        /// Accent color
        public let accent: Color
        
        /// Text color
        public let text: Color
    }
}

