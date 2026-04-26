import React from "react";
import "./Dashboard.css";

export default function DetectionCard({ event }) {
return ( <div className="detection-card">
{event.image ? (
<img
src={`data:image/jpeg;base64,${event.image}`}
alt="Detected Face"
className="detection-card-img"
/>
) : ( <div className="detection-card-img placeholder">No Image</div>
)} <div className="detection-card-info"> <p><strong>Age:</strong> {event.age}</p> <p><strong>Gender:</strong> {event.gender}</p> <p><strong>Emotion:</strong> {event.emotion}</p> </div> </div>
);
}



