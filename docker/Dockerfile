FROM --platform=linux/arm64 ros:jazzy-ros-base

WORKDIR /ros2_ws/

RUN printf '#!/bin/bash\nsource /opt/ros/jazzy/setup.bash\nif [ -f /ros2_ws/install/setup.bash ]; then\n  source /ros2_ws/install/setup.bash\nfi\nexec "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]