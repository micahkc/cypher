model pendulum "Simple Pendulum with input torque"
  parameter Real m = 1.0 "Mass";
  parameter Real l = 1.0 "Length of pendulum";
  parameter Real g = 9.81 "Gravitational acceleration";
  Real theta;
  Real omega;
  input Real u;
equation
  der(theta) = omega;
  der(omega) = -(g/l)*sin(theta) + u/(m*l^2);
end pendulum;

