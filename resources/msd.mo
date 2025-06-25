model msd "Mass spring damper"
  parameter Real k=1.0;
  parameter Real c=1.0;
  parameter Real m=1.0;
  Real x = 0.0;
  Real v;
  input Real u;
equation
  der(x) = v;
  der(v) = -(k/m)*x - (c/m)*v - (1/m)*u;
end msd;

