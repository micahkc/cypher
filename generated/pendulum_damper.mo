model pendulum_pid_io "Damped pendulum with PID ref and disturbance as inputs"
  // --- Plant parameters ---
  parameter Real m = 1.0 "Mass";
  parameter Real l = 1.0 "Length of pendulum";
  parameter Real g = 9.81 "Gravitational acceleration";
  parameter Real c = 0.1 "Damping coefficient";

  // --- PID parameters ---
  parameter Real Kp = 5.0 "Proportional gain";
  parameter Real Ki = 1.0 "Integral gain";
  parameter Real Kd = 0.5 "Derivative gain";
  parameter Real N  = 50.0 "Derivative filter pole";

  // --- I/O ---
  input Real theta_ref "Reference angle";
  // (ADDED) reference derivatives for feedforward:
  input Real dtheta_ref "Reference angular rate";
  input Real ddtheta_ref "Reference angular acceleration";
  input Real d = 0 "Additive disturbance torque";

  // --- States ---
  Real theta "Angle";
  Real omega "Angular rate";
  Real xi "Integral of angle error";
  Real omega_f "Filtered angular rate for D term";

  // --- Internal ---
  Real e "Angle error";
  Real u "Control torque";
  // (ADDED) feedforward torque:
  Real u_ff "Feedforward torque from reference";

equation
  // Plant
  der(theta) = omega;
  der(omega) = -(g/l)*sin(theta) - c/(m*l^2)*omega + (u + d)/(m*l^2);

  // PID (derivative on measurement, filtered)
  e = theta_ref - theta;
  der(xi) = e;
  der(omega_f) = N*(omega - omega_f);

  // (ADDED) feedforward from reference dynamics
  u_ff = m*l^2*ddtheta_ref + c*dtheta_ref + m*g*l*sin(theta_ref);

  // Sum feedforward + existing PID correction
  u = u_ff + (Kp*e + Ki*xi - Kd*omega_f);
end pendulum_pid_io;
