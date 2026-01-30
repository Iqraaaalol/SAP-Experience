"""
WebSocket connection management for crew dashboard.
"""
from typing import List, Dict
from fastapi import WebSocket
from datetime import datetime


class ConnectionManager:
    """Manage WebSocket connections for crew dashboard."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.alerts: List[Dict] = []  # Store alerts for persistence
        self.alert_id_counter = 0

    async def connect(self, websocket: WebSocket):
        """Accept and store a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ Crew dashboard connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"❌ Crew dashboard disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast alert to all connected crew dashboards and store it."""
        # Add unique ID and store the alert
        self.alert_id_counter += 1
        message['id'] = self.alert_id_counter
        message['acknowledged'] = False
        self.alerts.insert(0, message)  # Add to beginning of list
        
        # Broadcast to connected dashboards
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
    
    def get_alerts(self) -> List[Dict]:
        """Get all stored alerts."""
        return self.alerts
    
    def get_pending_alerts(self) -> List[Dict]:
        """Get only unacknowledged alerts."""
        return [a for a in self.alerts if not a.get('acknowledged', False)]
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self.alerts:
            if alert.get('id') == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().isoformat()
                return True
        return False
    
    def clear_acknowledged(self) -> int:
        """Clear all acknowledged alerts. Returns count of cleared alerts."""
        original_count = len(self.alerts)
        self.alerts = [a for a in self.alerts if not a.get('acknowledged', False)]
        return original_count - len(self.alerts)
    
    def clear_all_alerts(self) -> int:
        """Clear all alerts. Returns count of cleared alerts."""
        count = len(self.alerts)
        self.alerts = []
        return count


# Singleton instance
crew_manager = ConnectionManager()
