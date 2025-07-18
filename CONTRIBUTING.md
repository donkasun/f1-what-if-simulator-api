# Contributing to F1 What-If Simulator API

Thank you for your interest in contributing to the F1 What-If Simulator API! This document provides guidelines and information for contributors.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** for your changes
4. **Make your changes** following the guidelines below
5. **Test your changes** thoroughly
6. **Submit a pull request**

## 📋 Development Setup

### Prerequisites

- Python 3.11+
- Git
- Virtual environment (recommended)

### Local Development

```bash
# Clone your fork
git clone https://github.com/your-username/f1-what-if-simulator-api.git
cd f1-what-if-simulator-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp env.example .env
# Edit .env as needed

# Run the application
python -m app.main
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_simulation_service.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Follow the existing test structure in `tests/`
- Use descriptive test names
- Mock external dependencies
- Test both success and error cases

### Test Structure

```
tests/
├── test_api/           # API endpoint tests
├── test_services/      # Service layer tests
├── test_external/      # External API client tests
└── conftest.py         # Pytest configuration
```

## 📝 Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: Use `ruff` for automatic sorting
- **Type hints**: Required for all function parameters and return values

### Code Quality Tools

```bash
# Format code
black app/ tests/

# Lint code
ruff check app/ tests/

# Type checking
mypy app/

# Run all quality checks
black app/ tests/ && ruff check app/ tests/ && mypy app/
```

### Pre-commit Hooks

Consider setting up pre-commit hooks to automatically run quality checks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

## 🏗️ Architecture Guidelines

### Adding New Features

1. **API Layer** (`app/api/v1/`):
   - Add Pydantic schemas in `schemas.py`
   - Add endpoints in `endpoints.py`
   - Keep endpoints lean - delegate to services

2. **Service Layer** (`app/services/`):
   - Add business logic in service classes
   - Raise custom exceptions for business errors
   - Use dependency injection for external clients

3. **External Layer** (`app/external/`):
   - Add new API clients as needed
   - Implement caching where appropriate
   - Handle external API errors gracefully

### Error Handling

- Use custom exceptions from `app.core.exceptions`
- Log errors with appropriate context
- Return structured error responses
- Never expose internal implementation details

### Logging

- Use structured logging with `structlog`
- Include relevant context in log messages
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Log external API calls and responses

## 🔧 Configuration

### Environment Variables

- Add new configuration options to `app.core.config.Settings`
- Provide sensible defaults
- Document new variables in `env.example`
- Use type hints for all settings

### Adding Dependencies

- Add production dependencies to `requirements.txt`
- Add development dependencies to `requirements.txt` (commented)
- Consider the impact on Docker image size
- Document why the dependency is needed

## 📚 Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Follow Google docstring format
- Include type hints in docstrings
- Document exceptions that may be raised

### API Documentation

- Update OpenAPI schemas when adding new endpoints
- Provide meaningful examples in Pydantic models
- Document error responses
- Keep documentation in sync with implementation

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected behavior** vs actual behavior
4. **Environment details** (OS, Python version, etc.)
5. **Error messages** and stack traces
6. **Minimal example** if possible

## 💡 Feature Requests

When requesting features, please include:

1. **Clear description** of the feature
2. **Use case** and motivation
3. **Proposed implementation** approach
4. **Impact** on existing functionality
5. **Alternative solutions** considered

## 🔄 Pull Request Process

1. **Create a feature branch** from `main`
2. **Make focused commits** with clear messages
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Run quality checks** locally
6. **Submit PR** with clear description
7. **Address review feedback** promptly

### PR Guidelines

- **Title**: Clear, descriptive title
- **Description**: Explain what and why (not how)
- **Related issues**: Link to relevant issues
- **Breaking changes**: Document any breaking changes
- **Testing**: Describe how to test the changes

## 🏷️ Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 📞 Communication

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Code reviews**: Be constructive and respectful
- **Questions**: Ask in issues or discussions

## 🎯 Contribution Areas

We welcome contributions in these areas:

- **Bug fixes** and improvements
- **New API endpoints** and features
- **Performance optimizations**
- **Test coverage** improvements
- **Documentation** updates
- **Code quality** improvements
- **Security** enhancements

## 🙏 Recognition

Contributors will be recognized in:

- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

Thank you for contributing to the F1 What-If Simulator API! 🏎️ 