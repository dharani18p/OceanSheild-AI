import numpy as np
import requests
from datetime import datetime, timedelta

class OilSpillDriftPredictor:
    """
    Predictive drift modeling for oil spill movement
    Uses physics-based formulas + real-time environmental data
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key  # OpenWeatherMap API key
        self.BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
        
    def get_environmental_data(self, lat, lon):
        """Fetch real-time wind and current data"""
        if self.api_key:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(self.BASE_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                wind_speed = data['wind']['speed']  # m/s
                wind_direction = data['wind']['deg']  # degrees
                return wind_speed, wind_direction
        
        # Fallback to simulation if no API key
        return 5.0, 45.0  # Default: 5 m/s wind at 45°
    
    def calculate_drift_vector(self, wind_speed, wind_direction, oil_type="crude"):
        """
        Calculate oil drift based on wind leeway factor
        Leeway factor: oil movement as % of wind speed (typically 3-5%)
        """
        leeway_factors = {
            "crude": 0.03,      # Heavy crude: 3%
            "refined": 0.045,   # Refined oil: 4.5%
            "light": 0.05       # Light oil: 5%
        }
        
        leeway = leeway_factors.get(oil_type, 0.03)
        
        # Convert wind direction to radians
        wind_rad = np.radians(wind_direction)
        
        # Calculate drift velocity (m/s)
        drift_speed = wind_speed * leeway
        
        # Decompose into x, y components
        drift_x = drift_speed * np.sin(wind_rad)  # East-West
        drift_y = drift_speed * np.cos(wind_rad)  # North-South
        
        return drift_x, drift_y, drift_speed
    
    def predict_trajectory(self, start_lat, start_lon, hours=24, 
                          oil_type="crude", timestep_minutes=30):
        """
        Predict spill trajectory over time
        Returns list of (lat, lon, time) coordinates
        """
        # Get environmental data
        wind_speed, wind_direction = self.get_environmental_data(start_lat, start_lon)
        
        # Calculate drift vector
        drift_x, drift_y, drift_speed = self.calculate_drift_vector(
            wind_speed, wind_direction, oil_type
        )
        
        # Initialize trajectory
        trajectory = [(start_lat, start_lon, 0)]
        current_lat, current_lon = start_lat, start_lon
        
        # Simulate movement
        num_steps = int((hours * 60) / timestep_minutes)
        
        for step in range(1, num_steps + 1):
            time_elapsed = step * timestep_minutes
            
            # Convert drift from m/s to degrees (approximate)
            # 1 degree latitude ≈ 111 km
            # 1 degree longitude ≈ 111 km * cos(latitude)
            
            delta_time = timestep_minutes * 60  # Convert to seconds
            
            delta_lat = (drift_y * delta_time) / 111000  # meters to degrees
            delta_lon = (drift_x * delta_time) / (111000 * np.cos(np.radians(current_lat)))
            
            current_lat += delta_lat
            current_lon += delta_lon
            
            trajectory.append((
                round(current_lat, 6),
                round(current_lon, 6),
                time_elapsed
            ))
        
        return {
            "trajectory": trajectory,
            "wind_speed_ms": round(wind_speed, 2),
            "wind_direction_deg": round(wind_direction, 1),
            "drift_speed_ms": round(drift_speed, 4),
            "total_distance_km": round(self._calculate_distance(
                start_lat, start_lon, current_lat, current_lon
            ), 2),
            "prediction_hours": hours
        }
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate great circle distance in km"""
        R = 6371  # Earth radius in km
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
             np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def get_affected_zones(self, trajectory, radius_km=5):
        """
        Identify geographic zones at risk
        """
        affected_zones = []
        
        for lat, lon, time_min in trajectory:
            zone = {
                "lat": lat,
                "lon": lon,
                "time_minutes": time_min,
                "radius_km": radius_km,
                "alert_level": self._calculate_alert_level(time_min)
            }
            affected_zones.append(zone)
        
        return affected_zones
    
    def _calculate_alert_level(self, time_min):
        """Determine alert urgency based on time"""
        if time_min <= 120:  # 0-2 hours
            return "CRITICAL"
        elif time_min <= 360:  # 2-6 hours
            return "HIGH"
        elif time_min <= 720:  # 6-12 hours
            return "MEDIUM"
        else:
            return "LOW"


# Example usage
if __name__ == "__main__":
    predictor = OilSpillDriftPredictor()
    
    # Example: Oil spill off Gulf of Mexico
    start_lat = 28.5
    start_lon = -89.5
    
    prediction = predictor.predict_trajectory(
        start_lat, start_lon, 
        hours=48, 
        oil_type="crude"
    )
    
    print("Drift Prediction Results:")
    print(f"Total Distance: {prediction['total_distance_km']} km")
    print(f"Wind Speed: {prediction['wind_speed_ms']} m/s")
    print(f"\nTrajectory (first 5 points):")
    for point in prediction['trajectory'][:5]:
        print(f"  Time {point[2]}min: ({point[0]}, {point[1]})")