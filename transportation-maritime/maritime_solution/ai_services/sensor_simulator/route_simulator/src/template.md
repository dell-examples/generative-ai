# {vessel_name} - Voyage Report - Generated at: {generation_time}

## **Voyage Overview**

<div style="display: flex; align-items: center;">

  <div style="flex: 1; padding: 20px;">
    <h2>Voyage Details</h2>
    <ul>
      <li><strong>Voyage Number:</strong> {voyage_number}</li>
      <li><strong>Vessel Name:</strong> {vessel_name}</li>
      <li><strong>Captain:</strong> {captain_name}</li>
      <li><strong>Departure Port:</strong> {departure_port}</li>
      <li><strong>Arrival Port:</strong> {arrival_port}</li>
      <li><strong>Departure Date:</strong> {departure_date}</li>
      <li><strong>Arrival Date:</strong> {arrival_date}</li>
      <li><strong>Total Voyage Duration:</strong> {total_duration}</li>
    </ul>
  </div>

  <div style="flex: 1; text-align: center;">
    <img src="http://{ip_address}:5000/static/ship.jpg" alt="Ship Image" style="max-height: 100%; max-width: 100%;">
  </div>

</div>

## **Cargo Handling**

- **Total Containers Loaded:** {total_containers}
- **Container Types:** {container_types}
- **Total Dry Containers:** {total_dry_containers}
- **Total Reefer Containers:** {total_reefer_containers}


## **Power Consumption of Reefers** &nbsp;&nbsp; <span style="background-color: #4285f4; color: #ffffff; padding: 3px 6px; font-size:50%; border-radius: 3px;">AI Generated</span>

{power_consumption}

## **Ventilator Systems for Container Sweating Prevention** &nbsp;&nbsp; <span style="background-color: #4285f4; color: #ffffff; padding: 3px 6px; font-size:50%; border-radius: 3px;">AI Generated</span>

{sweat_info}

## **Maritime worker safety violations** &nbsp;&nbsp; <span style="background-color: #4285f4; color: #ffffff; padding: 3px 6px; font-size:50%; border-radius: 3px;">AI Generated</span>

{incidents}

## **Conclusion** &nbsp;&nbsp; <span style="background-color: #4285f4; color: #ffffff; padding: 3px 6px; font-size:50%; border-radius: 3px;">AI Generated</span>
