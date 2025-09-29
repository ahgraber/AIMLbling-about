---
mode: agent
description: Review and improve unit tests
tools: [createFile, createDirectory, editFiles, search, runCommands, runTasks, usages, think, problems, changes, testFailure, fetch, githubRepo, runTests, context7, getPythonEnvironmentInfo, getPythonExecutableCommand, configurePythonEnvironment]
---

# Instructions

Revise and expand unit tests, critically reviewing the provided source code and its existing unit tests.

1. Infer the intended behavior and purpose of the source code. If necessary, ask questions to clarify.
2. Evaluate whether the unit tests accurately validate this **intended behavior**, rather than simply confirming the current implementation.
3. Identify any gaps in test coverage, such as missing edge cases, incorrect assertions, or lack of negative tests.
4. If applicable, recommend improvements to the source code itself to enhance clarity, testability, or adherence to best practices.
5. Develop a clear, step-by-step plan. Break down the changes into manageable, incremental steps with explanation and rationale. Display those steps in a simple todo list using standard markdown format. Make sure you wrap the todo list in triple backticks so that it is formatted correctly.
6. Implement the changes incrementally. Make small, testable code changes.

Provide a clear summary of your changes and reasoning.

DO NOT MAKE CHANGES TO SOURCE CODE. You can only make changes to the unit tests.

## Testing Guidelines

### Testing Framework & Structure

- Use pytest as the primary testing framework.
- Structure tests using the Arrange-Act-Assert (AAA) pattern.
- Create separate test classes for different components or behaviors

### Test Naming & Organization

- Use descriptive names for test functions and test cases. Test method names should follow: test\_<method>_<scenario>_\<expected_result>
- Use markers to categorize tests for selective running.

### Test Coverage & Scenarios

- Write comprehensive unit tests for all critical functions, classes, and methods.
- Test both positive and negative test cases, including edge cases, boundary conditions, and error scenarios.
- Test that proper error handling is in place and proper exceptions are thrown.

### Test Isolation & Dependencies

- Mock external dependencies and I/O operations
- Utilize mocks and stubs to isolate the unit under test and avoid external dependencies when testing.
- Avoid testing implementation details; focus on behavior and contracts

### Test Data & Fixtures

- Utilize pytest fixtures to set up test environments and share test data.
- Use pytest.mark.parametrize to test different inputs and expected outcomes.
- Use meaningful test data that represents realistic scenarios
- Use fixtures to avoid logging inside test functions.
