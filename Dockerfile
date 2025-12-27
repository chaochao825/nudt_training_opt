# Use Python 3.8 image as base
FROM python:3.8

# Set working directory
WORKDIR /project

# Copy requirements
COPY requirements.txt ./

# Install PyTorch and dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . ./

# Create necessary directories
RUN mkdir -p /input /output /project/cfgs

# Default environment variables
ENV INPUT_DIR=/input
ENV OUTPUT_DIR=/output
# The following are lowercase as per requirement
ENV model=resnet50
ENV dataset=CIFAR-10
ENV epochs=2
ENV batch_size=32
ENV lr=0.001
ENV process=train

# Run main script
CMD ["python", "main.py"]
