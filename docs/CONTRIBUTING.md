# Contributing to AudioBloomAI

Thank you for your interest in contributing to AudioBloomAI! This document provides guidelines for both human developers and AI assistants to effectively contribute to the project.

## Table of Contents

- [Introduction](#introduction)
- [Human and AI Collaboration](#human-and-ai-collaboration)
- [Development Environment Setup](#development-environment-setup)
- [Contribution Workflow](#contribution-workflow)
- [AI-Ready Workflow](#ai-ready-workflow)
- [Issue Templates](#issue-templates)
- [Project Organization](#project-organization)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Guidelines](#documentation-guidelines)
- [Review Process](#review-process)

## Introduction

AudioBloomAI is a cutting-edge audio processing application that leverages Swift 6.0, Metal compute capabilities, and machine learning to provide innovative audio visualization and transformation tools. We welcome contributions from both human developers and AI assistants to help improve and expand the project.

## Human and AI Collaboration

We've designed our workflow to optimize collaboration between human developers and AI assistants:

- **For Human Contributors**: Focus on high-level design, code review, creative aspects, and ensuring technical correctness.
- **For AI Assistants**: Assist with implementation details, documentation, testing, and repetitive tasks based on structured issue templates.

All issues in our project have been designed to be "AI-ready" with clear instructions, structured formats, and explicit requirements to facilitate effective AI assistance.

## Development Environment Setup

### Requirements

- macOS 15.0 or higher
- Xcode 16.0 or higher
- Swift 6.0
- Metal-capable hardware
- Git and GitHub CLI

### Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone git@github.com:YOUR_USERNAME/AudioBloomAI.git
   cd AudioBloomAI
   ```
3. Set up the development environment:
   ```bash
   ./scripts/setup.sh
   ```
4. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Contribution Workflow

1. **Find or Create an Issue**: Browse existing issues or create a new one using our templates.
2. **Assign Yourself**: Comment on the issue to express interest and get assigned.
3. **Implement Changes**: Make your changes following our code standards.
4. **Test Your Changes**: Ensure all tests pass and add new tests as needed.
5. **Create a Pull Request**: Submit your changes for review.
6. **Address Review Feedback**: Make necessary adjustments based on reviewer comments.
7. **Merge**: Once approved, your changes will be merged.

## AI-Ready Workflow

Our project is optimized for AI assistant contribution through:

### AI-Ready Labels

All issues with the `ai-ready` label contain structured information designed for AI processing:

- Clear, step-by-step AI agent instructions
- Complexity scores to indicate difficulty
- Implementation order and dependencies
- Technical requirements
- Testing requirements
- Definition of Done criteria

### AI Agent Fields in Project Board

Our project board includes specialized fields for AI agents:

- **AI Agent Instructions**: Step-by-step tasks for AI to follow
- **Complexity Score**: Rating from 1-5 indicating task complexity
- **Implementation Status**: Current status in the AI workflow

### AI Contribution Process

1. AI assistants should prioritize issues with the `ai-ready` label
2. Follow the structured AI Agent Instructions in each issue
3. Generate implementations that strictly adhere to technical requirements
4. Provide detailed documentation for all implementations
5. Generate comprehensive tests according to testing requirements
6. Update the Implementation Status as progress is made

## Issue Templates

We provide several issue templates to ensure consistency:

### Feature Request Template
Use for proposing new features or enhancements

### Bug Report Template
Use for reporting bugs with reproduction steps

### Documentation Template (AI-Ready)
Use for documentation tasks optimized for AI processing

When creating issues, please:
- Use the appropriate template
- Fill in all required fields
- Add relevant labels
- Assign to the appropriate milestone
- Include AI Agent Instructions for AI-ready tasks

## Project Organization

Our project is organized into several components:

- **AudioProcessor**: Core audio processing and analysis
- **MLEngine**: Machine learning components
- **AudioBloomCore**: Shared functionality and utilities
- **Visualizer**: Visualization rendering and effects
- **AudioBloomUI**: User interface components
- **AudioBloomApp**: Main application

Each component has a dedicated label for issues and should follow the established architecture patterns.

## Code Standards

### Swift Style Guide

- Follow Swift API Design Guidelines
- Use Swift 6.0 features appropriately
- Implement proper error handling
- Use Swift's strong type system effectively
- Document all public APIs using DocC comments

### Commit Guidelines

- Use conventional commit messages (feat, fix, docs, style, refactor, test, chore)
- Reference issue numbers in commit messages
- Keep commits focused on single changes
- Write clear, descriptive commit messages

## Testing Requirements

All contributions must include appropriate tests:

- **Unit Tests**: Required for all new functions/methods
- **Integration Tests**: Required for component interactions
- **Performance Tests**: Required for performance-critical code
- **UI Tests**: Required for UI components

Test coverage should aim for minimum 80% coverage of new code.

## Documentation Guidelines

Documentation is critical for both humans and AI assistants:

- All public APIs must have DocC documentation
- Complex algorithms should include explanatory comments
- Include usage examples for key components
- Update README.md and wiki pages as needed
- Follow the AI-ready documentation format for AI-consumable docs

## Review Process

Pull requests will be reviewed by project maintainers:

1. Automated checks (CI) must pass
2. Code review by at least one maintainer
3. Documentation review
4. Testing review

For AI-assisted contributions, human maintainers will provide final verification.

---

Thank you for contributing to AudioBloomAI! Your efforts help make this project better for everyone.
