# Video Object Tracker

Real-time multi-object tracking in video using YOLOv8 with FastAPI deployment.

## Features

- Real-time webcam object detection
- Pre-trained YOLOv8n COCO model (80 classes)
- Professional web interface
- One-click deployment to cloud platforms

## Quick Start

### Local Deployment
```bash
pip install -r requirements.txt
python main.py
```
Access at http://localhost:8000

### Docker Deployment
```bash
docker build -t video-tracker .
docker run -p 8000:8000 video-tracker
```

### Cloud Deployment (Render.com)
1. Push code to GitHub
2. Connect repo to Render.com
3. Auto-deploys with included Dockerfile

## Project Structure

```
video-tracker/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker compose
├── DEPLOYMENT.md          # Deployment guide
├── templates/             # HTML templates
│   └── index.html        # Live demo interface
├── static/               # Static files
└── .gitignore            # Git ignore file
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## License

This project uses YOLOv8 from Ultralytics, licensed under AGPL-3.0.
