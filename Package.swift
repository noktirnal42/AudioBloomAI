// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "AudioBloomAI",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .executable(name: "AudioBloomApp", targets: ["AudioBloomApp"]),
        .library(name: "AudioBloomCore", targets: ["AudioBloomCore"]),
        .library(name: "AudioProcessor", targets: ["AudioProcessor"]),
        .library(name: "Visualizer", targets: ["Visualizer"]),
        .library(name: "MLEngine", targets: ["MLEngine"]),
        .library(name: "AudioBloomUI", targets: ["AudioBloomUI"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-algorithms", from: "1.2.1"),
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.3"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.6.3"),
        .package(url: "https://github.com/pointfreeco/swift-composable-architecture", from: "1.19.0"),
        .package(url: "https://github.com/AudioKit/AudioKit", from: "5.6.5")
    ],
    targets: [
        .executableTarget(
            name: "AudioBloomApp",
            dependencies: [
                "AudioBloomCore",
                "AudioProcessor",
                "Visualizer",
                "MLEngine",
                "AudioBloomUI"
            ],
            swiftSettings: [
                .enableUpcomingFeature("BareSlashRegexLiterals"),
                .enableUpcomingFeature("ConciseMagicFile"),
                .enableUpcomingFeature("ExistentialAny"),
                .enableUpcomingFeature("ForwardTrailingClosures"),
                .enableUpcomingFeature("ImportObjcForwardDeclarations")
            ]
        )
    ],
    swiftLanguageVersions: [.v6]
)
