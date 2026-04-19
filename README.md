# Unsupervised Learning for Autonomous Vehicle's (AVs) using Waymo's Open Motion Dataset
An honor's thesis project related to training an autonomous vehicle to see if AI is able to categorize each input (parked cars, pedestrians, moving cars, etc.) in the Waymo Open Motion Dataset based on their physics and see if it can find aggressive vs cautious driving clusters.

## Defined Vehicle Actions

> [!NOTE]
> Not considering external conditions in these cases
> External conditions may include poor weather, driving at night traffic stops, etc. 

| Action | Mathematical Definition | Context |
| :--- | :--- | :--- |
| Safe Cruising | Velocity > 5 m/s, Acceleration bounds [1.5, 2] m/s^2 | Maintain a constant, legal speed; Following a lead vehicle |
| Aggressive Tailgating | Velocity > 5 m/s | Lead vehicle present; High risk of rear-end collision |
| Smooth Stop | Velocity approaching 0 m/s, Braking bounds [-1.5, -3.5] m/s^2 | Approaching intersections, yielding to pedestrians, approaching stops |
| Evasive Hard Brake (Emergency Braking) | Vehicle approaching 0 m/s, Braking < -8 m/s^2 | Sudden obstacle or imminent collision; Reacting to unsafe environment | 
| Safe Intersection Turn | Heading Change > 45 degrees, Velocity < 10 m/s | Confirmed intersection |

