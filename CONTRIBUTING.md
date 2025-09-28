# Contributing to EigenWave

Thank you for your interest in contributing to EigenWave! This document provides guidelines and information about contributing to the project.

## Development Process

1. **Fork the repository** and create your branch from `main`
2. **Write tests** for any new functionality
3. **Ensure all tests pass** before submitting a PR
4. **Update documentation** as needed
5. **Submit a pull request**

## Pull Request Process

### Before Submitting

- Ensure your code follows the existing style
- Add tests for new functionality
- Update documentation if you're changing behavior
- Run tests locally: `cmake --build build && ctest --test-dir build`

### Automated Checks

When you submit a pull request, the following automated checks will run:

1. **Build and Test Matrix**: Tests on multiple platforms (Ubuntu, macOS, Windows) with different compilers
2. **Code Quality**: Static analysis with cppcheck and clang-tidy
3. **Code Coverage**: Ensures tests adequately cover the code

All checks must pass before merging.

## Code Style Guidelines

- Use 4 spaces for indentation (no tabs)
- Follow the existing naming conventions:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Template parameters: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
- Keep lines under 100 characters when reasonable
- Use trailing return types only when necessary
- Prefer `const` correctness

## Testing

- All new features must have corresponding tests
- Tests use Google Test framework
- Place tests in `tests/` directory
- Name test files as `test_<feature>.cpp`

## Building Locally

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build

# Run tests
ctest --test-dir build --verbose

# Run specific test
./build/tests/test_tensor
```

## Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when applicable

## Questions?

Feel free to open an issue for any questions about contributing!