# Base image: official ROS2 Humble
FROM ros:humble

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and basic tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install ROS2 packages
RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    && rm -rf /var/lib/apt/lists/*

# Set default workdir inside container
WORKDIR /workspace

# Source ROS setup automatically in bash
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Default command
CMD ["bash"]
