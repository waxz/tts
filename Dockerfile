# Use Python 3.10 slim image for a balance of size and compatibility
FROM python:3.10-slim

# Install system dependencies
# libsndfile1 and ffmpeg are often required for audio processing (scipy/numpy/onnx)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
# Hugging Face Spaces strictly require running as non-root (ID 1000)
RUN useradd -m -u 1000 user

# Switch to the "user" context
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy requirements first to leverage Docker cache
COPY --chown=user requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . $HOME/app


# Expose the port that Hugging Face expects
EXPOSE 7860

# Start the application
# We map host to 0.0.0.0 and port to 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
