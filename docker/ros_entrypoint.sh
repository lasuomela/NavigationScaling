#!/bin/bash

# Check if the earthrovers_deployment ros package is installed, and build it
if [ ! -d "/opt/NavigationScaling/earthrovers/deployment/src/install" ];
then
    echo "Building earthrovers deployment package"
    cd /opt/NavigationScaling/earthrovers/deployment/src
    colcon build
    cd /opt/NavigationScaling/
fi

# Add earthrovers_deployment setup.bash to the bashrc
echo "source /opt/NavigationScaling/earthrovers/deployment/src/install/setup.bash" >> /root/.bashrc
cd /opt/NavigationScaling/earthrovers/deployment/src/earthrovers_deployment/earthrovers_deployment
exec "$@"