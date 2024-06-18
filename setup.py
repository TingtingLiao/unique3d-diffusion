from setuptools import setup, find_packages

setup(
    name='unique3d_diffusion',
    version='0.0.1',
    description='Unique3d Diffusion Models', 
    packages=find_packages(),
    package_data={"unique3d_diffusion": ["configs/*.yaml"]} ,
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'omegaconf',
        'einops',
        'huggingface_hub',
        "transformers",
        "open-clip-torch",
    ],
)