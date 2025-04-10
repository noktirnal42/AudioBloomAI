// Test organization
import XCTest
@testable import AudioProcessor

@available(macOS 15.0, *)
final class AudioPipelineTestSuite {
    static func runAllTests() {
        AudioPipelineStateTests().runTests()
        AudioPipelineConnectionTests().runTests()
        AudioPipelineValidationTests().runTests()
        AudioPipelineErrorTests().runTests()
    }
}
