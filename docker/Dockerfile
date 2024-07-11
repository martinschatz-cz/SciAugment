# Use the official Python base image
FROM python:3.9

# jupyter-lab with SciAugment
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes

# Copy the requirements file
COPY requirements.txt /tmp/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# update pip
RUN pip install --no-cache-dir --upgrade pip

# Install the packages from requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install SciAugment package from GitHub
RUN pip install git+https://github.com/martinschatz-cz/SciAugment.git

# Expose the port JupyterLab runs on
EXPOSE 8888

# Create a directory for the Jupyter notebooks
RUN mkdir /notebooks

# Copy the requirements file
COPY /notebooks/SciAugment_test.ipynb /notebooks/

# Set the working directory
WORKDIR /notebooks

# Start JupyterLab without a token
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
