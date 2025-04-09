// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.
// Optimized for macOS 15+ and Xcode 16+

import PackageDescription

let package = Package(
    name: "AudioBloomAI",
    platforms: [
        .macOS(.v15) // Specific requirement for macOS 15+
    ],
    products: [
        // Main executable application
        .executable(
            name: "AudioBloomApp",
            targets: ["AudioBloomApp"]
        ),
        .library(
            name: "AudioBloomCore",
            targets: ["AudioBloomCore"]
        ),
        .library(
            name: "AudioProcessor",
            targets: ["AudioProcessor"]
        ),
        .library(
            name: "Visualizer", 
            targets: ["Visualizer"]
        ),
        .library(
            name: "MLEngine", 
            targets: ["MLEngine"]
        ),
        .library(
            name: "AudioBloomUI", 
            targets: ["AudioBloomUI"]
        )
    dependencies: [
        // External dependencies
        .package(url: "https://github.com/apple/swift-algorithms", from: "1.2.1"),
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.3"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.6.3"),
        .package(url: "https://github.com/pointfreeco/swift-composable-architecture", from: "1.19.0"),
        .package(url: "https://github.com/AudioKit/AudioKit", from: "5.6.5")
    ],
    targets: [
        // Main application target
        .executableTarget(
            name: "AudioBloomApp",
            dependencies: [
                "AudioBloomCore",
                "AudioProcessor",
                "Visualizer",
                "MLEngine",
                "AudioBloomUI"
            ],
            exclude: ["README.md"],
            resources: [
                .process("Resources")
            ]
        ),
        
        // Core shared functionality
        .target(
            name: "AudioBloomCore",
            dependencies: [
                .product(name: "Algorithms", package: "swift-algorithms"),
                .product(name: "Numerics", package: "swift-numerics"),
                .product(name: "Logging", package: "swift-log")
            ],
            exclude: ["README.md"]
        ),
        
        // Audio processing module
        .target(
            name: "AudioProcessor",
            dependencies: [
                "AudioBloomCore",
                .product(name: "Numerics", package: "swift-numerics"),
                .product(name: "AudioKit", package: "AudioKit")
            ],
            exclude: ["README.md"],
            swiftSettings: [
                // Any swift settings would go here
            ],
            linkerSettings: [
                .linkedFramework("AVFoundation", .when(platforms: [.macOS])),
                .linkedFramework("CoreAudio", .when(platforms: [.macOS])),
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
            ]
        ),
        
        // Visualization module
        .target(
            name: "Visualizer",
            dependencies: [
                "AudioBloomCore",
                .product(name: "Algorithms", package: "swift-algorithms"),
                .product(name: "Numerics", package: "swift-numerics")
            ],
            exclude: ["README.md"], // Removed MetalRenderer.swift from exclusions
            resources: [
                .process("Resources/Shaders")
            ],
            cSettings: [
                .unsafeFlags(["-fmodules"], .when(platforms: [.macOS]))
            ],
            swiftSettings: [
                // Any swift settings would go here
            ],
            linkerSettings: [
                .linkedFramework("Metal", .when(platforms: [.macOS])),
                .linkedFramework("MetalKit", .when(platforms: [.macOS])),
                .linkedFramework("CoreImage", .when(platforms: [.macOS])),
                .linkedFramework("QuartzCore", .when(platforms: [.macOS]))
            ]
        ),
        
        // Neural Engine integration module
        .target(
            name: "MLEngine",
            dependencies: [
                "AudioBloomCore",
                .product(name: "Numerics", package: "swift-numerics"),
                .product(name: "Logging", package: "swift-log")
            ],
            exclude: ["README.md"],
            resources: [
                .process("Resources/Models")
            ],
            cSettings: [
                .unsafeFlags(["-fmodules"], .when(platforms: [.macOS]))
            ],
            swiftSettings: [
                .define("ENABLE_NEURAL_ENGINE")
            ],
            linkerSettings: [
                .linkedFramework("CoreML", .when(platforms: [.macOS])),
                .linkedFramework("Accelerate", .when(platforms: [.macOS])),
                .linkedFramework("CreateML", .when(platforms: [.macOS])),
                .linkedFramework("SoundAnalysis", .when(platforms: [.macOS]))
            ]
        ),
        
        // UI Components module
        .target(
            name: "AudioBloomUI",
            dependencies: [
                "AudioBloomCore",
                "MLEngine",
                "Visualizer",
                .product(name: "ComposableArchitecture", package: "swift-composable-architecture")
            ],
            exclude: ["README.md"], // Removed PresetControlsView.swift from exclusions
            swiftSettings: [
                // Any swift settings would go here
            ],
            linkerSettings: [
                .linkedFramework("SwiftUI", .when(platforms: [.macOS])),
                .linkedFramework("Combine", .when(platforms: [.macOS]))
            ]
        ),
        
        // Test targets
        .testTarget(
            name: "AudioBloomTests",
            dependencies: ["AudioBloomCore"],
            path: "Tests/AudioBloomTests",
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "AudioProcessorTests",
            dependencies: ["AudioProcessor"],
            path: "Tests/AudioProcessorTests",
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "VisualizerTests",
            dependencies: ["Visualizer"],
            path: "Tests/VisualizerTests",
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "MLEngineTests",
            dependencies: ["MLEngine"],
            path: "Tests/MLEngineTests",
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "AudioBloomUITests",
            dependencies: ["AudioBloomUI"],
            path: "Tests/AudioBloomUITests"
        )
    ],
    swiftLanguageModes: [.version("6")]
)
