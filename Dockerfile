FROM python:3.10

WORKDIR /home/user/app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libxcursor1 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcb-dri3-0 \
    libxshmfence1 \
    fonts-liberation \
    libappindicator3-1 \
    libnss3-tools \
    libxss1 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install

# Copy the rest of the application
COPY . .

CMD ["python", "src/app.py"]