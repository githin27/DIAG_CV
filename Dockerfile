# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /workspace_container

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
# Install Jupyter Notebook
RUN pip install jupyterlab
RUN apt-get update -y

# install required packages for X forwarding
RUN apt-get install -y libgl1-mesa-glx

RUN pip install -r requirements.txt

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter","lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]