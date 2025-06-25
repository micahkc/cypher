model SimpleSatellite
  // Parameters
  parameter Real Ix = 10; // Moment of inertia around X-axis (kg·m²)
  parameter Real Iy = 15; // Moment of inertia around Y-axis (kg·m²)
  parameter Real Iz = 20; // Moment of inertia around Z-axis (kg·m²)
  
  // State variables: angular velocities (rad/s)
  Real wx(start=0);
  Real wy(start=0);
  Real wz(start=0);
  
  // Applied external torques (N·m)
  input Real Tx;
  input Real Ty;
  input Real Tz;
  
  // Equations of motion for rotational dynamics (Euler’s equations)
equation
  der(wx) = (Tx - (Iz - Iy)*wy*wz) / Ix;
  der(wy) = (Ty - (Ix - Iz)*wz*wx) / Iy;
  der(wz) = (Tz - (Iy - Ix)*wx*wy) / Iz;
  
  // You can add integration for orientation angles if needed:
  // Real phi(start=0), theta(start=0), psi(start=0);
  // der(phi) = ... (depends on wx, wy, wz and angle conventions)
  
end SimpleSatellite;

