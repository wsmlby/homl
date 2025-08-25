# BUILD CLI from source

Build the CLI from source by following these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/homl-dev/homl.git
   cd homl/cli
   ```
2. create a venv

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. run the build command

    ```
    cd cli
    bash build.sh
    ```

3. The CLI binary will be at `dist/homl`


# Adding support for other platforms

1. Make modification to the following functions to add support for other platforms inside [install_utils](homl_cli/utils/install_utils.py):

    1. detect_platform: to support detecting the platform correctly
    2. get_platform_config: to return the correct image, and add correct hardware resource assignments for docker.
    3. install: add platform-specific installation steps

2. When running with locally build image, use HOML_DOCKER_IMAGE_OVERRIDE environment variable to specify the image when running the `homl server install` command.
