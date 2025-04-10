//
//  AudioBloomUITests.swift
//  AudioBloomAI
//
//  Created on: 2025-04-09
//

import XCTest
import SwiftUI
import AudioBloomUI

final class AudioBloomUITests: XCTestCase {
    
    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
        continueAfterFailure = false
    }
    
    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    func testUIComponentsInitialization() throws {
        // Basic test to verify UI components can be initialized
        let _ = PresetControlsView()
        XCTAssertTrue(true, "PresetControlsView initialized successfully")
    }
    
    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        measure {
            // Put the code whose performance you want to measure here
            let _ = PresetControlsView()
        }
    }
}

