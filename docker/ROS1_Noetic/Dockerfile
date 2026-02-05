# Base image: official ROS1 Noetic
FROM ros:noetic

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

# Install ROS1 packages
RUN apt-get update && apt-get install -y \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    && rm -rf /var/lib/apt/lists/*

# Set default workdir inside container
WORKDIR /workspace

# Source ROS setup automatically in bash
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Default command
CMD ["bash"]

# Set user arguments (being username, echo $UID, and id -g)
# NOTE: These should be overwritten when running docker build!
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

# Create a non-root user with matching UID/GID
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Add a custom .bashrc with warm colors and fixed hostname
RUN echo 'export PS1="\\[\\e[1;31m\\]\\u@robotdataprocess_ros1_container\\[\\e[0m\\]:\\[\\e[1;33m\\]\\w\\[\\e[0m\\]\\$ "' > /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

# Set default user
USER ${USERNAME}
WORKDIR /home/${USERNAME}
