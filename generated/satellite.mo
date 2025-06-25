model SimpleSatellite
  // Parameters: moments of inertia (kg·m²)
  parameter Real Ix = 10;
  parameter Real Iy = 15;
  parameter Real Iz = 20;

  // Parameters: constant torques applied (N·m)
  parameter Real Tx = 4;
  parameter Real Ty = 0;
  parameter Real Tz = 0;

  // State variables: angular velocities (rad/s)
  Real wx(start=0.1);
  Real wy(start=0.0);
  Real wz(start=0.0);

equation
  // Euler's equations with constant torques as parameters
  der(wx) = (Tx - (Iz - Iy) * wy * wz) / Ix;
  der(wy) = (Ty - (Ix - Iz) * wz * wx) / Iy;
  der(wz) = (Tz - (Iy - Ix) * wx * wy) / Iz;

end SimpleSatellite;
