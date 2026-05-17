Dablyou Media Tracker
> Tunisia's first real-time media attention analytics ecosystem — measuring engagement across TV, radio, and outdoor advertising.
Dablyou Media Tracker helps advertisers, media owners, and brands move from estimated exposure to measurable attention across every major media channel in Tunisia.
---
The Problem: 

Traditional media advertising in Tunisia still lacks reliable and real-time attention measurement. Advertisers know where their campaigns are displayed or broadcasted, but they lack accurate answers to questions such as:
Who actually watched a TV campaign?
Who actively listened to a radio campaign?
Who really noticed an outdoor billboard?
Which age group or gender engaged the most?
Which channel, station, or billboard location performed best?
How did the campaign perform in real time?
Current methods rely on delayed reports, manual estimations, and incomplete audience data — creating a gap between media exposure and real audience attention.
> Advertisers pay for exposure. They need measurable attention.
---
Platform Components
The platform is composed of three integrated solutions:
1. تبّع و أربح — Mobile App
A gamified mobile application that transforms TV watching and radio listening into an interactive quiz-based experience.
Users can:
create an account with age and gender (used for anonymous audience segmentation)
choose between TV and Radio mode
select the channel or station they are currently watching or listening to
receive live quiz questions related to what is on screen or on air
answer within a limited time window
earn points for correct answers
convert points into discount coupons
The app is designed for the Tunisian market and uses Tunisian dialect written in Arabic as its primary interface language.
---
2. Web Dashboard
A real-time analytics dashboard that centralizes engagement data collected from the mobile app and outdoor advertising panels.
Designed for internal teams, advertisers, media owners, and partners. It visualizes:
engaged TV viewers and radio listeners via the app
quiz exposure, response rates, and correct answer rates
points distributed and coupons generated
TV and radio ad attention performance
outdoor billboard performance via PanelIQ
audience segmentation by age and gender
> The platform measures the exact number of engaged users inside the app — not claimed national audience totals.
---
3. PanelIQ — Outdoor Advertising Intelligence
An edge AI system that measures the real performance of roadside advertising panels — updated every 5 minutes, privacy-first by design.
PanelIQ uses a small camera installed near each billboard, connected to a Raspberry Pi that processes everything locally. It estimates:
how many vehicles and pedestrians passed the panel
what type of vehicles (sedan, SUV, motorcycle, truck) — used as a socioeconomic proxy
how many people were likely looking at the panel (attentive impressions)
whether the panel is physically intact and visible
the audience profile based on zone-level census data, time of day, and weather
---
PanelIQ — Technical Details
Architecture
```
┌─────────────────────────────────────┐
│         Edge Device (Pi 4)          │
│  YOLOv8n + ByteTrack + Face Blur    │
│  Job A: vehicle counting & attention│
│  Job B: panel integrity (SSIM)      │
└────────────────┬────────────────────┘
                 │ POST every 5 min
                 │ (aggregate counts only, no images)
┌────────────────▼────────────────────┐
│         Backend (FastAPI)           │
│  Scoring engine · SQLite · REST API │
│  Zone data · Weather · Context      │
└────────────────▼────────────────────┘
                 │
┌────────────────▼────────────────────┐
│         Dashboard (HTML/JS)         │
│  Live scores · Vehicle mix charts   │
│  Attention metrics · Alert monitor  │
└─────────────────────────────────────┘
```
Project Structure
```
panel_hackathon/
├── backend/
│   ├── main.py               # FastAPI app, endpoints, SQLite
│   ├── scoring.py            # Footfall estimator, context modulator, CPM
│   ├── verify_pipeline.py    # YOLO model validation endpoint
│   ├── static/
│   │   └── dashboard.css
│   └── templates/
│       └── dashboard.html    # Full dashboard UI
├── edge/
│   ├── run_edge.py           # Entry point — accepts --panel argument
│   ├── jobs/
│   │   ├── job_a.py          # Vehicle detection, tracking, attention scoring
│   │   ├── job_b.py          # Panel integrity monitor (threaded)
│   │   └── tamper_check.py   # Laplacian blur + brightness tamper detection
│   └── vision/
│       ├── detector.py       # YOLOv8n + ByteTrack wrapper
│       ├── blur.py           # On-device face blurring (top 25% of person bbox)
│       ├── attention.py      # Head pose / attention estimation
│       └── ssim_check.py     # SSIM comparison for Job B
├── configs/
│   ├── panel_rois.yaml       # Per-panel ROI, zone type, lat/lng, attention config
│   └── thresholds.yaml       # SSIM, tamper, attention thresholds
├── data/
│   └── panels/
│       ├── panel_001/        # video.mp4 + ref.jpg per panel
│       ├── panel_002/
│       └── panel_003/
├── tools/
│   ├── extract_frames.py     # Extract reference frame from video
│   └── test_all_panels.py    # Run full pipeline test across all panels
├── models/
│   └── yolov8n.pt            # Pre-trained model (runs offline after first download)
├── .env                      # OWM_API_KEY (not committed)
├── requirements.txt
└── README.md
```
Setup
1. Clone and install dependencies
```bash
git clone <your-repo-url>
cd panel_hackathon
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```
2. Download the YOLO model (once)
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
cp ~/.cache/ultralytics/yolov8n.pt models/yolov8n.pt
```
3. Set up environment variables
Create a `.env` file at the project root:
```
OWM_API_KEY=your_openweathermap_key_here
```
Get a free key at openweathermap.org.
4. Add panel videos and reference images
```
data/panels/panel_001/video.mp4
data/panels/panel_001/ref.jpg
```
To extract a reference frame from a video:
```bash
python tools/extract_frames.py --panel panel_001
```
Running the Platform
Start the backend
```bash
uvicorn backend.main:app --reload
```
Open the dashboard at `http://127.0.0.1:8000`
Start the edge pipeline (one process per panel)
```bash
python -m edge.run_edge --panel panel_001
python -m edge.run_edge --panel panel_002
python -m edge.run_edge --panel panel_003
```
Test all panels
```bash
python tools/test_all_panels.py
```
API Endpoints
Method	Endpoint	Description
GET	`/`	Dashboard UI
GET	`/panels`	List all configured panels with status
GET	`/panels/{id}/score`	Live reach score for a panel
GET	`/data?panel_id=`	Vehicle readings and alerts
GET	`/api/verify-video?panel_id=`	Run YOLO model check on panel video
POST	`/ingest/vehicle-mix`	Receive vehicle counts from edge
POST	`/ingest/integrity-alert`	Receive SSIM alert from edge
Scoring Model
```
impressions = vehicles x avg_occupancy x visibility_factor
              x zone_ses_weight x time_multiplier x weather_multiplier
```
Factor	Source
`avg_occupancy`	1.4 (INS Tunisia urban average)
`zone_ses_weight`	INS delegation census data
`time_multiplier`	Rush hour x1.3 · Friday midday x0.6 · Night x0.4
`weather_multiplier`	OpenWeatherMap API · Rain x0.8
`CPM estimate`	`base_cpm x ses_score / 5.0` (base: 8 TND)
Privacy and Compliance
PanelIQ is compliant with Tunisian Law 63-2004 on personal data protection:
All image processing runs locally on the Raspberry Pi
Faces are blurred on-device before any processing
No images or video are ever transmitted to the backend
Only aggregate counts are sent: `{car: 12, person: 5, attentive_impressions: 3, ...}`
No face recognition model is used at any point
---
Requirements
```
fastapi
uvicorn
opencv-python
numpy
ultralytics
scikit-image
requests
pyyaml
pydantic
python-dotenv
```
---
Dablyou Media Tracker · Built for the Tunisian market · PanelIQ edge device: Raspberry Pi 4 + YOLOv8n · Privacy-first by design
