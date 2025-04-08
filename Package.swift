// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "AudioBloom",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        // Main application
        .executable(
            name: "AudioBloom",
            targets: ["AudioBloomApp"]
        ),
        // Libraries that can be reused
        .library(
            name: "AudioBloomCore",
            targets: ["AudioBloomCore"]
        ),
    ],
    dependencies: [
        // External dependencies can be added here as needed
    ],
    targets: [
        // Main application target
        .executableTarget(
            name: "AudioBloomApp",
            dependencies: [
                "AudioBloomCore",
                "CoreAudio",
                "Visualizer",
                "MLEngine"
            ],
            resources: [
                .process("Resources")
            ]
        ),
        
        // Core shared functionality
        .target(
            name: "AudioBloomCore",
            dependencies: [
                "CoreAudio",
                "Visualizer",
                "MLEngine"
            ]
        ),
        
        // Audio processing module
        .target(
            name: "CoreAudio",
            dependencies: []
        ),
        
        // Metal-based visualization module
        .target(
            name: "Visualizer",
            dependencies: [],
            resources: [
                .process("Resources/Shaders")
            ]
        ),
        
        // Neural Engine integration module
        .target(
            name: "MLEngine",
            dependencies: []
        ),
        
        // Test targets
        .testTarget(
            name: "AudioBloomTests",
            dependencies: ["AudioBloomCore"]
        ),
        .testTarget(
            name: "CoreAudioTests",
            dependencies: ["CoreAudio"]
        ),
        .testTarget(
            name: "VisualizerTests",
            dependencies: ["Visualizer"]
        ),
        .testTarget(
            name: "MLEngineTests",
            dependencies: ["MLEngine"]
        ),
    ]
)

