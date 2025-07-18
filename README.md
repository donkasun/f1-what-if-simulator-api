# F1 What-If Simulator API

A high-performance, production-ready FastAPI backend for Formula 1 simulation and analysis. This API provides comprehensive F1 data access, machine learning-powered lap time predictions, and robust error handling.

## 🏎️ Features

- **FastAPI-based REST API** with automatic OpenAPI documentation
- **Machine Learning Integration** for lap time predictions
- **OpenF1 API Integration** for real F1 data
- **Structured Logging** with JSON output
- **Comprehensive Error Handling** with custom business exceptions
- **Async HTTP Client** with connection pooling and caching
- **Production-ready Docker** setup with multi-stage builds
- **Type Safety** with 100% type hint coverage
- **CORS Support** for frontend integration

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd f1-what-if-simulator-api
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python -m app.main
   ```

The API will be available at `http://localhost:8000`

### Docker Deployment

```bash
# Build the image
docker build -t f1-what-if-simulator-api .

# Run the container
docker run -p 8000:8000 f1-what-if-simulator-api
```

## 📚 API Documentation

Once the server is running, you can access:

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔧 API Endpoints

### Health Check
- `GET /api/v1/health` - Service health status

### Drivers
- `GET /api/v1/drivers?season=2024` - Get all drivers for a season

### Tracks
- `GET /api/v1/tracks?season=2024` - Get all tracks for a season

### Simulations
- `POST /api/v1/simulate` - Run a what-if simulation
- `GET /api/v1/simulation/{simulation_id}` - Get simulation results

## 🏗️ Architecture

The application follows a clean, layered architecture:

```
app/
├── main.py              # FastAPI app, middleware, exception handlers
├── api/                 # API layer (endpoints, schemas)
│   └── v1/
│       ├── endpoints.py # Lean endpoint functions
│       └── schemas.py   # Pydantic models
├── services/            # Business logic layer
│   └── simulation_service.py
├── core/                # Configuration and utilities
│   ├── config.py        # Environment-based settings
│   ├── exceptions.py    # Custom business exceptions
│   └── logging_config.py
├── models/              # ML model management
│   └── model_loader.py
└── external/            # External API clients
    └── openf1_client.py
```

## 🔒 Security & Error Handling

- **Input Validation**: All requests validated with Pydantic models
- **Custom Exceptions**: Business-specific error handling
- **Structured Logging**: JSON-formatted logs with request tracking
- **CORS Configuration**: Configurable allowed origins
- **Rate Limiting**: Built-in protection against abuse

## 📊 Machine Learning

The API includes a machine learning component for lap time predictions:

- **Model Loading**: Automatic model loading with fallback to dummy model
- **Feature Engineering**: Historical data processing and feature extraction
- **Prediction Pipeline**: End-to-end prediction workflow
- **Confidence Scoring**: Quality assessment of predictions

## 🧪 Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_simulation_service.py
```

## 📝 Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking

```bash
# Format code
black app/ tests/

# Lint code
ruff check app/ tests/

# Type check
mypy app/
```

### Adding New Endpoints

1. Add schema to `app/api/v1/schemas.py`
2. Add endpoint to `app/api/v1/endpoints.py`
3. Add business logic to `app/services/`
4. Add tests in `tests/`

## 🌐 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Debug mode | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FORMAT` | Log format | `json` |
| `OPENF1_API_URL` | OpenF1 API URL | `https://api.openf1.org` |
| `OPENF1_API_TIMEOUT` | API timeout (seconds) | `30` |
| `MODEL_PATH` | ML model path | `app/models/lap_time_predictor.joblib` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run code quality checks
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the logs for detailed error information

## 🔮 Roadmap

- [ ] Database integration for persistent storage
- [ ] Real-time WebSocket support
- [ ] Advanced ML model training pipeline
- [ ] Performance monitoring and metrics
- [ ] Authentication and authorization
- [ ] Rate limiting and API quotas