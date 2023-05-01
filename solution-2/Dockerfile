# Start with an anaconda/pytorch image
FROM continuumio/anaconda3:2022.05-py38_0

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary packages
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
RUN pip install -r requirements.txt

# Expose port 5000 for the Flask app
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]