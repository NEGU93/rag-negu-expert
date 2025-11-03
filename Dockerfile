FROM python:3.10

WORKDIR /home/user/app

# Create user with UID 1000 (matches HF Spaces)
RUN useradd -m -u 1000 user

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Playwright cache directory and install browsers
RUN playwright install-deps
ENV PLAYWRIGHT_BROWSERS_PATH=/home/user/app/playwright-browsers
RUN playwright install

COPY . .

# Set proper ownership
RUN chmod -R 755 /home/user/app/playwright-browsers
RUN chown -R user:user /home/user/app

ENV PYTHONPATH=/home/user/app

# Switch to non-root user
USER user

CMD ["python", "src/app.py"]
