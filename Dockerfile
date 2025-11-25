# Use the official AWS base image for Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the function code
COPY app.py .

# Set the CMD to your handler (app.lambda_handler)
CMD ["app.lambda_handler"]