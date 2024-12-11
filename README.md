
# Data Extraction and NLP

This project is designed to extract and process data using Natural Language Processing (NLP) techniques. The application is built using Python and deployable in a Docker container, allowing you to run it in any environment that supports Docker.

## Features
- Data extraction from text-based sources
- NLP processing with various libraries like pandas, numpy, and more
- Multi-architecture Docker image support (Linux/amd64, Linux/arm64)
- Integration with GitHub Container Registry for easy distribution

## Prerequisites
To run this project locally or deploy it in a containerized environment, you need the following prerequisites:
- Python 3.11 or higher
- Docker (for containerization)
- Git (for version control)
- pip (for Python package installation)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/cayanide/Data-Extraction-and-NLP.git
cd Data-Extraction-and-NLP
```

### 2. Install dependencies
Create a virtual environment and install required dependencies.
```bash
python3 -m venv .venv
source .venv/bin/activate  # For macOS/Linux
.venv\Scripts\activate  # For Windows

pip install -r requirements.txt
```

### 3. Run the application
To run the application locally:
```bash
python main.py
```

### 4. Build and run using Docker (optional)
Alternatively, you can build and run the application using Docker for a more streamlined experience.

#### Build the Docker image:
```bash
docker build -t my-nlp-app .
```

#### Run the Docker container:
```bash
docker run -p 5000:5000 my-nlp-app
```

## Docker Setup
This project includes a `Dockerfile` for containerizing the application. You can build the Docker image with multi-architecture support using the following command:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/cayanide/my-nlp-app:latest --push .
```


Screenshots:

![image](https://github.com/user-attachments/assets/c0ea1a68-339a-4680-8dae-4036b5b1d03c)


![image](https://github.com/user-attachments/assets/eeca723b-694c-4cb4-b458-9f38df6088f2)

<img width="1163" alt="image" src="https://github.com/user-attachments/assets/854ab131-274d-4167-b2e6-bb187aebf703" />


This will build the image for both `amd64` and `arm64` platforms and push it to GitHub Container Registry.

## Contributing
We welcome contributions to improve this project. To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- This project uses various NLP techniques and libraries like `pandas`, `numpy`, etc.
- Thanks to the contributors and maintainers of the libraries that made this project possible.

