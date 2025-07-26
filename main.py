import modal

evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        ["build-essential", "cmake", "ninja-build",
            "libcudnn8", "libcudnn8-dev", "git", "gcc", "g++"],
    )
    .env({
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
    })
    .run_commands("git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && cd evo2 && pip install .")
    .run_commands("pip uninstall -y transformer-engine transformer_engine")
    .run_commands("pip install 'transformer_engine[pytorch]==1.13' --no-build-isolation")
    .pip_install_from_requirements("requirements.txt")

)

app = modal.App("variant-analysis-evo2", image=evo2_image)


@app.function(gpu="H100", volumes={mount_path: volume})
def run_brca1_analysis():
    print("Running BRCA1 variant analysis with EVO2...")


@app.function()
def brca1_example():
    print("Running BRCA1 variant analysis with EVO2...")


@app.local_entrypoint()
def main():
    brca1_example.local()
