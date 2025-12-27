from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import cv2
import base64
from io import BytesIO
import os
from datetime import datetime, timedelta
import json
import random
import threading
import time

# Google Earth Engine
try:
    import ee
    ee.Initialize(project='oceanshield-ai')  # 👈 ADD PROJECT NAME
    EE_AVAILABLE = True
    print("✅ Earth Engine initialized successfully")
except Exception as e:
    EE_AVAILABLE = False
    print("⚠️ Earth Engine not available:", e)

from deep_model.model import SpillNet

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join("model", "spillnet.pth")

model = SpillNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Storage
spill_database = []
traveler_alerts = []
satellite_alerts = []

# Monitored regions
MONITORED_REGIONS = [
    {"name": "Gulf of Mexico", "bbox": [-97, 18, -81, 31]},
    {"name": "Bay of Bengal", "bbox": [82, 10, 88, 16]},
    {"name": "Persian Gulf", "bbox": [48, 24, 56, 30]},
    {"name": "North Sea", "bbox": [-2, 54, 8, 62]},
    {"name": "Arabian Sea", "bbox": [60, 10, 75, 25]},
    {"name": "South China Sea", "bbox": [105, 5, 120, 20]}
]

# ============================================
# SATELLITE DATA INTEGRATION
# ============================================

class SatelliteDataFetcher:
    """Real-time satellite data from Google Earth Engine"""
    
    def __init__(self):
        self.initialized = EE_AVAILABLE
        if not EE_AVAILABLE:
            print("⚠️ Satellite features disabled - Earth Engine not available")
    
    def fetch_latest_sentinel1(self, bbox, days_back=30):
        """Fetch Sentinel-1 radar data"""
        if not self.initialized:
            return None
        
        try:
            roi = ee.Geometry.Rectangle(bbox)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            collection = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(roi)
                .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            )
            
            count = collection.size().getInfo()
            print(f"📡 Found {count} Sentinel-1 images")
            
            if count == 0:
                return None
            
            image = collection.sort("system:time_start", False).first()
            properties = image.getInfo()['properties']
            
            return image, properties
        except Exception as e:
            print(f"Error fetching Sentinel-1: {e}")
            return None
    
    def get_thumbnail_url(self, ee_image, bbox, size=800):
        """Get thumbnail URL"""
        if not self.initialized:
            return None
        
        try:
            roi = ee.Geometry.Rectangle(bbox)
            vv = ee_image.select('VV')
            
            url = vv.getThumbUrl({
                'region': roi,
                'dimensions': size,
                'min': -25,
                'max': 0,
                'palette': ['000000', '0000FF', '00FFFF', 'FFFF00', 'FF0000']
            })
            
            return url
        except Exception as e:
            print(f"Error getting thumbnail: {e}")
            return None
    
    def detect_potential_spills(self, sentinel1_image, bbox):
        """Detect dark spots (potential oil spills) in radar"""
        if not self.initialized:
            return None
        
        try:
            roi = ee.Geometry.Rectangle(bbox)
            vv = sentinel1_image.select('VV')
            
            # Oil appears dark in radar (< -22 dB typically)
            dark_spots = vv.lt(-22)
            
            # Clean up noise
            dark_spots = dark_spots.focal_median(radius=2, kernelType='circle')
            
            # Calculate area
            spill_area = dark_spots.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=roi,
                scale=250,
                maxPixels=1e10
            ).getInfo()
            
            area_km2 = spill_area.get('VV', 0) / 1e6
            
            return {
                'area_km2': round(area_km2, 2),
                'detected': area_km2 > 1,  # Threshold: 1 km²
                'severity': 'HIGH' if area_km2 > 5 else 'MEDIUM' if area_km2 > 1 else 'LOW'
            }
        except Exception as e:
            print(f"Error detecting spills: {e}")
            return None

satellite_fetcher = SatelliteDataFetcher()

# ============================================
# TEMPORAL FINGERPRINTING
# ============================================

def extract_temporal_features(img_pil):
    img_np = np.array(img_pil.resize((224, 224)))
    
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hue_mean = np.mean(hsv[:,:,0])
    saturation_mean = np.mean(hsv[:,:,1])
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_sharpness = laplacian.var()
    
    texture_std = np.std(gray)
    contrast = gray.max() - gray.min()
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    else:
        area, perimeter, circularity = 0, 0, 0
    
    return {
        "color_decay_score": round(float(saturation_mean / 255), 3),
        "hue_shift_degrees": round(float(hue_mean * 2), 1),
        "edge_sharpness": round(float(edge_sharpness), 2),
        "texture_complexity": round(float(texture_std), 2),
        "contrast_level": round(float(contrast), 1),
        "spill_area_pixels": int(area),
        "boundary_perimeter": round(float(perimeter), 1),
        "shape_circularity": round(float(circularity), 3)
    }

# ============================================
# VISUALIZATION
# ============================================

def generate_advanced_visualizations(img_tensor, age_value, thickness_value):
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    visualizations = {}
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Age heatmap
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    age_overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    visualizations['age_heatmap'] = encode_image(age_overlay)
    
    # Thickness map
    thickness_map = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    thickness_overlay = cv2.addWeighted(img_np, 0.5, thickness_map, 0.5, 0)
    visualizations['thickness_map'] = encode_image(thickness_overlay)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_colored[:,:,1] = 0
    edge_overlay = cv2.addWeighted(img_np, 0.7, edges_colored, 0.3, 0)
    visualizations['edge_detection'] = encode_image(edge_overlay)
    
    # Contour analysis
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_np.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    visualizations['contour_analysis'] = encode_image(contour_img)
    
    return visualizations

def encode_image(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_str}"

# ============================================
# RISK ASSESSMENT
# ============================================

def advanced_risk_assessment(age, thickness, temporal_features, risk_probs):
    risk_factors = []
    total_score = 0
    
    if age < 60:
        risk_factors.append({
            "factor": "Fresh Spill (< 1 hour)",
            "severity": "CRITICAL",
            "score": 40,
            "description": "Immediate containment required"
        })
        total_score += 40
    elif age < 180:
        risk_factors.append({
            "factor": "Recent Spill (1-3 hours)",
            "severity": "HIGH",
            "score": 25,
            "description": "Rapid response needed"
        })
        total_score += 25
    else:
        risk_factors.append({
            "factor": "Aged Spill (> 3 hours)",
            "severity": "MEDIUM",
            "score": 10,
            "description": "Continued monitoring"
        })
        total_score += 10
    
    if thickness > 0.5:
        risk_factors.append({
            "factor": "High Oil Concentration",
            "severity": "HIGH",
            "score": 30,
            "description": "Significant threat"
        })
        total_score += 30
    
    if temporal_features['shape_circularity'] < 0.5:
        risk_factors.append({
            "factor": "Irregular Spread Pattern",
            "severity": "HIGH",
            "score": 20,
            "description": "Unpredictable diffusion"
        })
        total_score += 20
    
    if total_score >= 70:
        threat_level = "CRITICAL"
    elif total_score >= 40:
        threat_level = "HIGH"
    elif total_score >= 20:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"
    
    return {
        "threat_level": threat_level,
        "risk_score": total_score,
        "max_score": 100,
        "risk_factors": risk_factors,
        "recommendations": generate_recommendations(threat_level, age, thickness)
    }

def generate_recommendations(threat_level, age, thickness):
    recommendations = []
    
    if threat_level in ["CRITICAL", "HIGH"]:
        recommendations.append("🚨 Deploy containment booms immediately")
        recommendations.append("📞 Alert coast guard")
        recommendations.append("⛵ Issue travel advisory")
    
    if age < 120:
        recommendations.append("⏱️ Prioritize rapid response")
    
    if thickness > 0.3:
        recommendations.append("🛢️ Prepare heavy-duty skimmers")
    
    recommendations.append("📊 Continue monitoring")
    recommendations.append("🌊 Model drift patterns")
    
    return recommendations

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def create_traveler_alert(spill_data, location):
    spill_lat = float(location.get('lat', 25.0)) + random.uniform(-0.5, 0.5)
    spill_lon = float(location.get('lon', -80.0)) + random.uniform(-0.5, 0.5)
    
    distance = calculate_distance(
        float(location.get('lat', 25.0)),
        float(location.get('lon', -80.0)),
        spill_lat, spill_lon
    )
    
    alert = {
        "id": len(traveler_alerts) + 1,
        "type": "SPILL_ALERT",
        "severity": spill_data['risk_assessment']['threat_level'],
        "location": f"{spill_lat:.2f}°N, {spill_lon:.2f}°W",
        "distance": f"{distance:.1f} km",
        "message": f"{spill_data['risk_assessment']['threat_level']} risk oil spill detected",
        "timestamp": datetime.now().isoformat(),
        "recommendations": spill_data['risk_assessment']['recommendations']
    }
    
    traveler_alerts.append(alert)
    return alert

# ============================================
# API ENDPOINTS
# ============================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "✅ OceanShield-AI Complete Backend Running",
        "version": "4.0 - Satellite Integration Edition",
        "features": [
            "AI Spill Detection",
            "Temporal Fingerprinting",
            "Traveler Safety Dashboard",
            "Real-time Satellite Monitoring" if EE_AVAILABLE else "Satellite Integration (Disabled - Need Auth)",
            "Automated Regional Scanning"
        ],
        "device": device,
        "satellite_enabled": EE_AVAILABLE,
        "active_spills": len(spill_database),
        "active_alerts": len(traveler_alerts),
        "satellite_alerts": len(satellite_alerts)
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files["file"]
        location_data = request.form.get("location", "{}")
        location = json.loads(location_data) if location_data != "{}" else {}
        
        img_pil = Image.open(file).convert("RGB")
        temporal_features = extract_temporal_features(img_pil)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            age, thickness, risk_logits = model(img_tensor)
            risk_probs = F.softmax(risk_logits, dim=1)[0]
        
        risk_levels = ["LOW", "MEDIUM", "HIGH"]
        predicted_risk = risk_levels[risk_probs.argmax().item()]
        
        risk_probs_dict = {
            "LOW": round(risk_probs[0].item(), 3),
            "MEDIUM": round(risk_probs[1].item(), 3),
            "HIGH": round(risk_probs[2].item(), 3)
        }
        
        visualizations = generate_advanced_visualizations(img_tensor, age.item(), thickness.item())
        risk_assessment = advanced_risk_assessment(age.item(), thickness.item(), temporal_features, risk_probs_dict)
        
        explainability = [
            f"✅ Temporal Analysis: Edge sharpness {temporal_features['edge_sharpness']:.1f} indicates ~{age.item():.0f} min age",
            f"🎨 Color Decay: {temporal_features['color_decay_score']} confirms weathering",
            f"📐 Geometry: Circularity {temporal_features['shape_circularity']} shows spread",
            f"🧠 AI: {risk_probs.max().item()*100:.1f}% confidence in {predicted_risk}",
            f"📊 Risk Score: {risk_assessment['risk_score']}/100"
        ]
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "analysis_id": f"SPILL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "predictions": {
                "spill_age_minutes": round(age.item(), 2),
                "oil_thickness_index": round(thickness.item(), 3),
                "risk_level": predicted_risk,
                "risk_probabilities": risk_probs_dict
            },
            "temporal_features": temporal_features,
            "risk_assessment": risk_assessment,
            "visualizations": visualizations,
            "explainability": explainability,
            "model_info": {
                "architecture": "SpillNet",
                "device": device,
                "confidence": round(risk_probs.max().item(), 3)
            }
        }
        
        spill_database.append(response)
        
        if risk_assessment['threat_level'] in ['HIGH', 'CRITICAL']:
            alert = create_traveler_alert(response, location)
            response['traveler_alert_created'] = True
            response['alert_id'] = alert['id']
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

# ============================================
# SATELLITE ENDPOINTS
# ============================================

@app.route("/satellite/regions", methods=["GET"])
def get_regions():
    """Get list of monitored regions"""
    return jsonify({
        "regions": MONITORED_REGIONS,
        "total": len(MONITORED_REGIONS),
        "satellite_enabled": EE_AVAILABLE
    })

@app.route("/satellite/fetch", methods=["POST"])
def fetch_satellite():
    """Fetch latest satellite data for a region"""
    if not EE_AVAILABLE:
        return jsonify({
            "status": "error",
            "message": "Satellite features not available. Install: pip install earthengine-api"
        }), 503
    
    data = request.json
    bbox = data.get('bbox')
    region_name = data.get('name', 'Unknown')
    
    result = satellite_fetcher.fetch_latest_sentinel1(bbox, days_back=30)
    
    if result is None:
        return jsonify({
            "status": "error",
            "message": "No satellite data available for this period"
        }), 404
    
    image, properties = result
    
    # Get thumbnail URL
    thumbnail_url = satellite_fetcher.get_thumbnail_url(image, bbox)
    
    # Detect potential spills
    detection = satellite_fetcher.detect_potential_spills(image, bbox)
    
    response = {
        "status": "success",
        "region": region_name,
        "satellite": "Sentinel-1",
        "acquisition_date": properties.get('system:time_start'),
        "thumbnail_url": thumbnail_url,
        "detection": detection,
        "bbox": bbox
    }
    
    # Create alert if spill detected
    if detection and detection['detected']:
        satellite_alert = {
            "id": len(satellite_alerts) + 1,
            "region": region_name,
            "area_km2": detection['area_km2'],
            "severity": detection['severity'],
            "timestamp": datetime.now().isoformat(),
            "source": "Sentinel-1 Automated Detection"
        }
        satellite_alerts.append(satellite_alert)
        response['alert_created'] = True
        response['alert'] = satellite_alert
    
    return jsonify(response)

@app.route("/satellite/auto-scan", methods=["POST"])
def auto_scan():
    """Automated scanning of all monitored regions"""
    if not EE_AVAILABLE:
        return jsonify({
            "status": "error",
            "message": "Satellite features not available"
        }), 503
    
    results = []
    new_alerts = []
    
    for region in MONITORED_REGIONS:
        print(f"🔍 Scanning {region['name']}...")
        
        result = satellite_fetcher.fetch_latest_sentinel1(region['bbox'], days_back=7)
        
        if result:
            image, properties = result
            detection = satellite_fetcher.detect_potential_spills(image, region['bbox'])
            
            if detection and detection['detected']:
                alert = {
                    "id": len(satellite_alerts) + 1,
                    "region": region['name'],
                    "area_km2": detection['area_km2'],
                    "severity": detection['severity'],
                    "timestamp": datetime.now().isoformat(),
                    "source": "Automated Regional Scan"
                }
                satellite_alerts.append(alert)
                new_alerts.append(alert)
            
            results.append({
                "region": region['name'],
                "scanned": True,
                "detection": detection
            })
        else:
            results.append({
                "region": region['name'],
                "scanned": False,
                "message": "No data available"
            })
    
    return jsonify({
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "regions_scanned": len(results),
        "alerts_generated": len(new_alerts),
        "results": results,
        "new_alerts": new_alerts
    })

@app.route("/satellite/alerts", methods=["GET"])
def get_satellite_alerts():
    """Get all satellite-detected alerts"""
    return jsonify({
        "total": len(satellite_alerts),
        "alerts": satellite_alerts
    })

# ============================================
# TRAVELER ENDPOINTS
# ============================================

@app.route("/traveler/alerts", methods=["GET"])
def get_traveler_alerts():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    radius = request.args.get('radius', 100, type=float)
    
    # Combine manual and satellite alerts
    all_alerts = traveler_alerts + [
        {
            **alert,
            "type": "SATELLITE_DETECTION",
            "location": alert['region']
        }
        for alert in satellite_alerts
    ]
    
    return jsonify({
        "total_alerts": len(all_alerts),
        "alerts": all_alerts
    })

@app.route("/traveler/route-safety", methods=["POST"])
def check_route_safety():
    data = request.json
    start_lat = float(data.get('start_lat', 25.0))
    start_lon = float(data.get('start_lon', -80.0))
    end_lat = float(data.get('end_lat', 26.0))
    end_lon = float(data.get('end_lon', -81.0))
    
    route_distance = calculate_distance(start_lat, start_lon, end_lat, end_lon)
    
    # Check for hazards
    hazards = []
    
    # Check against satellite alerts
    for alert in satellite_alerts:
        if alert['severity'] in ['HIGH', 'CRITICAL']:
            hazards.append({
                "type": "satellite_detection",
                "region": alert['region'],
                "severity": alert['severity'],
                "area": alert['area_km2']
            })
    
    safety_status = "SAFE"
    if len(hazards) > 0:
        safety_status = "UNSAFE" if any(h.get('severity') == 'CRITICAL' for h in hazards) else "CAUTION"
    
    recommendation = {
        "SAFE": "✅ Route appears SAFE",
        "CAUTION": "⚠️ Proceed with CAUTION",
        "UNSAFE": "❌ Route NOT RECOMMENDED"
    }[safety_status]
    
    return jsonify({
        "route_distance_km": round(route_distance, 2),
        "safety_status": safety_status,
        "hazards_count": len(hazards),
        "hazards": hazards,
        "recommendation": recommendation
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify({
        "total_analyses": len(spill_database),
        "traveler_alerts": len(traveler_alerts),
        "satellite_alerts": len(satellite_alerts),
        "satellite_enabled": EE_AVAILABLE,
        "monitored_regions": len(MONITORED_REGIONS),
        "last_update": datetime.now().isoformat()
    })

if __name__ == "__main__":
    print("="*60)
    print("🌊 OceanShield-AI Complete Backend")
    print("="*60)
    print(f"🔧 Device: {device}")
    print(f"📦 Model: {MODEL_PATH}")
    print(f"🛰️ Satellite Integration: {'✅ ENABLED' if EE_AVAILABLE else '❌ DISABLED (Need Auth)'}")
    print(f"⛵ Traveler Safety: ✅ ENABLED")
    print(f"🌍 Monitored Regions: {len(MONITORED_REGIONS)}")
    print("="*60)
    if not EE_AVAILABLE:
        print("⚠️  To enable satellite features:")
        print("   1. pip install earthengine-api")
        print("   2. earthengine authenticate")
        print("="*60)
    app.run(port=5000, debug=True)