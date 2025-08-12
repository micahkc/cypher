model Satellite
  // Physical constants
  parameter Real G = 6.67430e-11;
  parameter Real M_earth = 5.972e24;
  parameter Real R_earth = 6.371e6;
  parameter Real m = 1000;

  // Inertia tensor (diagonal)
  parameter Real Ixx = 200.0;
  parameter Real Iyy = 200.0;
  parameter Real Izz = 100.0;

  // Position and velocity vectors
  Real [3] radius// (start = {R_earth + 500000, 0, 0});
  Real [3] v; //(start = {0, 7660, 0}) "Velocity (m/s)";

  // Angular velocity (state)
  Real [3] omega; //(start = {0, 0, 0.01}) "Angular velocity (rad/s)";
  Real [3] tau;

  // Gravitational force vector
  Real [3] Fg;

  // Inertia-weighted angular velocity
  Real [3] Iomega;

  // Output vector (example array usage)
  //Real[2,2] outputVec = {{1, 2}, {3, 4}};  // ✅ Safe nested array literal

  Real r_norm;
equation
  // Translational dynamics
  der(r) = v;
  r_norm = sqrt(radius[1]^2 + radius[2]^2 + radius[3]^2);
  Fg = -G * M_earth * m / r_norm^3 * 1;
  der(v) = Fg / m;

  // Inertia times omega
  Iomega[1] = Ixx * omega[1];
  Iomega[2] = Iyy * omega[2];
  Iomega[3] = Izz * omega[3];

  // Euler’s rotational dynamics
  der(omega[1]) = (tau[1] - (omega[2]*Iomega[3] - omega[3]*Iomega[2])) / Ixx;
  der(omega[2]) = (tau[2] - (omega[3]*Iomega[1] - omega[1]*Iomega[3])) / Iyy;
  der(omega[3]) = (tau[3] - (omega[1]*Iomega[2] - omega[2]*Iomega[1])) / Izz;
end Satellite;
