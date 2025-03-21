# Production Deployment Guide

This guide provides instructions for deploying the Knowledge Graph application in a production environment.

## Recommended Production Setup

### Option 1: Gunicorn + Nginx

1. **Install Gunicorn**:
   ```
   pip install gunicorn
   ```

2. **Create a systemd service file** (`/etc/systemd/system/knowledgegraph.service`):
   ```ini
   [Unit]
   Description=Knowledge Graph Application
   After=network.target

   [Service]
   User=yourusername
   WorkingDirectory=/path/to/KAG/knowledgeGraph
   Environment="PATH=/path/to/your/venv/bin"
   Environment="USE_PROCESS_MANAGER=true"
   Environment="RESTART_TRIGGER_FILE=/path/to/KAG/knowledgeGraph/.restart_trigger"
   ExecStart=/path/to/your/venv/bin/gunicorn --workers 4 --bind 127.0.0.1:8080 app:app
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. **Configure Nginx as a reverse proxy** (`/etc/nginx/sites-available/knowledgegraph`):
   ```nginx
   server {
       listen 80;
       server_name your_domain.com;

       location / {
           proxy_pass http://127.0.0.1:8080;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           
           # Health check handling
           proxy_intercept_errors on;
           error_page 502 503 504 = @health_check;
       }
       
       location @health_check {
           # Return a failed health check for monitoring systems
           return 503;
       }
       
       location /health {
           proxy_pass http://127.0.0.1:8080/health;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           
           # Add health check to access log
           access_log /var/log/nginx/health_check.log;
       }
   }
   ```

4. **Enable and start the services**:
   ```
   sudo systemctl enable knowledgegraph
   sudo systemctl start knowledgegraph
   sudo systemctl enable nginx
   sudo systemctl start nginx
   ```

### Option 2: Docker Deployment

1. **Create a Dockerfile**:
   ```
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   ENV USE_PROCESS_MANAGER=true
   ENV RESTART_TRIGGER_FILE=/app/.restart_trigger
   
   EXPOSE 8080
   
   HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
     CMD curl -f http://localhost:8080/health || exit 1
   
   CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:8080", "app:app"]
   ```

2. **Build and run the Docker container**:
   ```
   docker build -t knowledgegraph .
   docker run -p 80:8080 -v /path/for/data:/app/processed_files knowledgegraph
   ```

## Monitoring and Maintenance

### Health Checks

The application provides a `/health` endpoint that returns the current health status. You can configure monitoring tools like Prometheus, Grafana, or a simple cron job to check this endpoint regularly.

### System Status

The `/status` endpoint provides detailed metrics about the application, including:
- Server uptime
- Process resource usage (CPU, memory)
- Graph statistics
- Active requests

### Restarting the Application

The application provides several ways to restart:

1. **Web Interface**: Use the "Restart Server" button in the UI
2. **API Endpoint**: Make a POST request to `/restart` 
3. **Process Manager**: If using a process manager like systemd, you can use:
   ```
   sudo systemctl restart knowledgegraph
   ```
4. **File Trigger**: Touch the restart trigger file:
   ```
   touch /path/to/RESTART_TRIGGER_FILE
   ```

### Backup Recommendations

1. Regularly backup the `processed_files` directory
2. Consider implementing automated backups:

```bash
#!/bin/bash
# Example backup script
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/path/to/backups"
APP_DIR="/path/to/KAG/knowledgeGraph"

# Create backup of processed files
tar -czf "$BACKUP_DIR/knowledge_graph_data_$TIMESTAMP.tar.gz" -C "$APP_DIR" processed_files

# Rotate backups (keep last 7 days)
find "$BACKUP_DIR" -name "knowledge_graph_data_*.tar.gz" -mtime +7 -delete
```

### Log Management

The application logs to both console and `app.log`. For production, consider:

1. Using a log rotation solution like logrotate
2. Sending logs to a centralized logging system like ELK stack or Graylog
