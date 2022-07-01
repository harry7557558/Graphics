template<typename vec3, typename Float>
class ColorFunctions {
  // calculations are done in double precision
  static Float clp(double x) {
    return (Float)(x<0.?0.:x>1.?1.:x);
  }
public:
  static vec3 AlpineColors(double t) {
    double r = (.61*t+.15)*t+.27;
    double g = (.5*t+.09)*t+.4;
    double b = (((-2.66*t+4.45)*t-.91)*t-.5)*t+.51;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LakeColors(double t) {
    double r = (-.21*t+.85)*t+.29;
    double g = (-.93*t+1.81)*t+.02;
    double b = ((1.02*t-2.57)*t+1.9)*t+.49;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ArmyColors(double t) {
    double r = (.13*t+.18)*t+.45;
    double g = ((1.1*t-1.17)*t+.29)*t+.58;
    double b = (.65*t-.5)*t+.51;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 MintColors(double t) {
    double r = (-.58*t+1.04)*t+.46;
    double g = (-.37*t+.01)*t+.97;
    double b = (-.55*t+.7)*t+.63;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AtlanticColors(double t) {
    double r = (((-.97*t+1.5)*t-1.4)*t+1.26)*t+.1;
    double g = ((-.85*t-.25)*t+1.48)*t+.12;
    double b = ((-1.25*t+.41)*t+1.33)*t+.13;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 NeonColors(double t) {
    double r = (-.33*t+.44)*t+.69;
    double g = (((-3.39*t+5.26)*t-1.04)*t-1.62)*t+.95;
    double b = (((-3.65*t+7.87)*t-4.42)*t+.7)*t+.27;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AuroraColors(double t) {
    double r = (((6.62*t-12.68)*t+7.58)*t-.87)*t+.25;
    double g = (((2.78*t-9.91)*t+9.)*t-1.95)*t+.31;
    double b = (((-8.72*t+17.42)*t-10.1)*t+2.11)*t+.24;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PearlColors(double t) {
    double r = (((-4.27*t+10.54)*t-7.15)*t+.94)*t+.89;
    double g = (((-4.24*t+10.44)*t-7.65)*t+1.44)*t+.82;
    double b = (((-6.53*t+14.12)*t-9.04)*t+1.66)*t+.76;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AvocadoColors(double t) {
    double r = ((-1.86*t+3.64)*t-.83)*t+.03;
    double g = ((1.04*t-2.34)*t+2.33)*t-.03;
    double b = (.03*t+.17)*t+.01;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PlumColors(double t) {
    double r = ((3.19*t-4.92)*t+2.7)*t-.03;
    double g = (.8*t+.03)*t+.04;
    double b = ((-2.*t+2.23)*t+.17)*t+0.;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BeachColors(double t) {
    double r = (((4.55*t-6.3)*t+1.93)*t+.01)*t+.86;
    double g = (((3.94*t-5.4)*t+1.2)*t+.8)*t+.5;
    double b = (((-4.62*t+7.53)*t-2.6)*t+.42)*t+.25;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RoseColors(double t) {
    double r = (((4.77*t-9.87)*t+5.17)*t+.48)*t+.16;
    double g = (((4.06*t-8.77)*t+4.81)*t-.29)*t+.32;
    double b = (-1.2*t+1.24)*t+.04;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CandyColors(double t) {
    double r = (((2.68*t-3.42)*t-.7)*t+1.75)*t+.38;
    double g = ((-1.41*t+2.18)*t-.12)*t+.2;
    double b = ((-2.02*t+2.86)*t-.32)*t+.35;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SolarColors(double t) {
    double r = ((.88*t-2.29)*t+1.97)*t+.44;
    double g = ((-1.16*t+1.87)*t+.08)*t+.01;
    double b = (.16*t-.03)*t+.01;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CMYKColors(double t) {
    double r = (((-2.95*t+2.91)*t-2.1)*t+2.)*t+.27;
    double g = (((-8.11*t+8.04)*t+1.3)*t-2.05)*t+.82;
    double b = (((-11.92*t+21.96)*t-12.03)*t+1.26)*t+.88;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SouthwestColors(double t) {
    double r = (((-1.65*t+1.99)*t-1.92)*t+1.45)*t+.43;
    double g = (((5.87*t-14.14)*t+9.87)*t-1.34)*t+.33;
    double b = (((16.25*t-30.02)*t+17.36)*t-2.91)*t+.27;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DeepSeaColors(double t) {
    double r = ((3.26*t-3.94)*t+1.38)*t+.13;
    double g = ((-1.4*t+2.65)*t-.35)*t+.02;
    double b = (-.74*t+1.47)*t+.28;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 StarryNightColors(double t) {
    double r = ((-1.92*t+2.68)*t+.07)*t+.11;
    double g = ((-1.39*t+1.18)*t+.84)*t+.15;
    double b = ((-.77*t-.3)*t+1.22)*t+.18;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FallColors(double t) {
    double r = ((-1.21*t+1.97)*t-.06)*t+.29;
    double g = (((-5.*t+9.93)*t-5.2)*t+.67)*t+.38;
    double b = (.44*t-.54)*t+.38;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SunsetColors(double t) {
    double r = (((4.93*t-8.68)*t+2.56)*t+2.25)*t-.02;
    double g = ((-1.09*t+1.85)*t+.22)*t+.03;
    double b = (((-14.8*t+36.43)*t-27.3)*t+6.77)*t-.1;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FruitPunchColors(double t) {
    double r = ((.99*t-.63)*t-.39)*t+1.03;
    double g = (((3.2*t-4.45)*t+.35)*t+.78)*t+.49;
    double b = (((-5.66*t+7.54)*t-1.47)*t+.09)*t-0.;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ThermometerColors(double t) {
    double r = (((3.75*t-9.01)*t+4.86)*t+.78)*t+.15;
    double g = (((5.74*t-10.57)*t+2.57)*t+2.25)*t+.1;
    double b = ((1.63*t-3.87)*t+1.61)*t+.77;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 IslandColors(double t) {
    double r = (((7.05*t-18.07)*t+14.48)*t-3.6)*t+.8;
    double g = (-1.38*t+1.73)*t+.4;
    double b = ((1.42*t-4.68)*t+3.34)*t+.22;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 WatermelonColors(double t) {
    double r = ((-1.21*t+1.21)*t+.78)*t+.13;
    double g = ((-1.96*t+.6)*t+1.6)*t+.11;
    double b = (((-4.28*t+4.77)*t-1.17)*t+.9)*t+.09;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrassTones(double t) {
    double r = (((3.34*t-6.77)*t+1.16)*t+2.35)*t+.09;
    double g = (((3.34*t-6.91)*t+1.73)*t+1.89)*t+.11;
    double b = (((1.86*t-3.96)*t+1.2)*t+.93)*t+.03;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GreenPinkTones(double t) {
    double r = (((11.14*t-29.11)*t+21.17)*t-3.07)*t+.08;
    double g = (((6.54*t-6.8)*t-4.06)*t+4.27)*t+.19;
    double b = (((9.33*t-25.28)*t+18.64)*t-2.6)*t+.11;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrownCyanTones(double t) {
    double r = ((.97*t-3.47)*t+2.52)*t+.3;
    double g = ((-1.01*t-.32)*t+1.78)*t+.19;
    double b = ((-1.33*t+.26)*t+1.77)*t+.07;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PigeonTones(double t) {
    double r = (((-4.68*t+8.84)*t-4.7)*t+1.36)*t+.17;
    double g = (((-3.16*t+6.49)*t-3.97)*t+1.5)*t+.15;
    double b = (((-3.96*t+8.27)*t-5.14)*t+1.65)*t+.19;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CherryTones(double t) {
    double r = ((2.*t-4.02)*t+2.83)*t+.21;
    double g = (1.15*t-.3)*t+.19;
    double b = (1.12*t-.27)*t+.19;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RedBlueTones(double t) {
    double r = (((5.28*t-9.7)*t+3.1)*t+.99)*t+.46;
    double g = (((5.07*t-11.42)*t+5.87)*t+.66)*t+.15;
    double b = ((-2.29*t+1.99)*t+.6)*t+.21;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CoffeeTones(double t) {
    double r = (-.31*t+.85)*t+.43;
    double g = ((.75*t-.86)*t+.79)*t+.34;
    double b = (((-3.64*t+9.28)*t-6.4)*t+1.53)*t+.24;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RustTones(double t) {
    double r = (((3.72*t-7.43)*t+3.71)*t+1.02)*t+.01;
    double g = (-.49*t+.97)*t-.02;
    double b = (.16*t-.32)*t+.2;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FuchsiaTones(double t) {
    double r = (-.64*t+1.55)*t+.07;
    double g = ((-1.11*t+1.97)*t-.04)*t+.13;
    double b = (-.46*t+1.38)*t+.07;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SiennaTones(double t) {
    double r = ((.93*t-2.32)*t+1.86)*t+.44;
    double g = ((-.8*t+.94)*t+.56)*t+.18;
    double b = ((-1.35*t+2.5)*t-.44)*t+.09;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GrayTones(double t) {
    double r = (.39*t+.47)*t+.1;
    double g = (.26*t+.59)*t+.1;
    double b = (.11*t+.68)*t+.11;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ValentineTones(double t) {
    double r = (-.07*t+.56)*t+.51;
    double g = (.45*t+.34)*t+.09;
    double b = (.3*t+.44)*t+.18;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GrayYellowTones(double t) {
    double r = ((-1.57*t+1.79)*t+.52)*t+.18;
    double g = ((-2.08*t+2.16)*t+.42)*t+.22;
    double b = ((-3.17*t+2.8)*t+.3)*t+.31;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkTerrain(double t) {
    double r = ((4.05*t-5.9)*t+2.92)*t-.04;
    double g = (((2.38*t+1.4)*t-5.75)*t+3.)*t+.04;
    double b = (((4.4*t-3.2)*t-1.04)*t+.48)*t+.45;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LightTerrain(double t) {
    double r = ((-1.35*t+1.98)*t-.29)*t+.54;
    double g = ((-1.85*t+3.21)*t-1.25)*t+.78;
    double b = ((-1.86*t+4.13)*t-2.23)*t+.86;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GreenBrownTerrain(double t) {
    double r = (((5.66*t-9.79)*t+4.02)*t+1.13)*t-0.;
    double g = (((6.03*t-7.78)*t+.54)*t+2.27)*t-.02;
    double b = ((5.38*t-7.66)*t+3.32)*t-.05;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SandyTerrain(double t) {
    double r = (((3.48*t-7.65)*t+3.48)*t+.27)*t+.68;
    double g = (((4.39*t-8.69)*t+3.53)*t+.83)*t+.3;
    double b = (-.49*t+.43)*t+.2;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrightBands(double t) {
    double r = (((-16.63*t+28.73)*t-12.13)*t-.11)*t+.98;
    double g = (((6.8*t-14.81)*t+8.23)*t+.17)*t+.23;
    double b = (((-8.13*t+26.66)*t-27.59)*t+9.55)*t-.04;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkBands(double t) {
    double r = (((17.15*t-33.07)*t+20.27)*t-3.87)*t+.68;
    double g = (((-2.97*t+6.52)*t-3.03)*t-.41)*t+.81;
    double b = (((-5.17*t+5.24)*t+2.06)*t-2.81)*t+1.;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Aquamarine(double t) {
    double r = ((2.66*t-3.77)*t+1.1)*t+.67;
    double g = ((1.62*t-2.42)*t+.82)*t+.73;
    double b = ((1.39*t-1.78)*t+.41)*t+.84;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Pastel(double t) {
    double r = ((-1.54*t+.75)*t+.41)*t+.77;
    double g = ((-1.64*t+1.3)*t+.52)*t+.5;
    double b = (((-6.39*t+11.82)*t-4.98)*t-.49)*t+.98;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BlueGreenYellow(double t) {
    double r = (1.4*t-.63)*t+.15;
    double g = (-.82*t+1.69)*t+.01;
    double b = (-.63*t+.52)*t+.42;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Rainbow(double t) {
    double r = ((-4.42*t+7.09)*t-2.3)*t+.46;
    double g = (((1.63*t-4.94)*t+1.89)*t+1.49)*t+.04;
    double b = (((-8.06*t+20.35)*t-17.07)*t+4.45)*t+.45;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkRainbow(double t) {
    double r = (((2.62*t-9.73)*t+9.43)*t-2.02)*t+.33;
    double g = (((6.12*t-15.58)*t+10.93)*t-1.77)*t+.41;
    double b = (((-2.34*t+2.43)*t+1.04)*t-1.62)*t+.65;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 TemperatureMap(double t) {
    double r = (((5.84*t-11.36)*t+4.9)*t+1.31)*t+.16;
    double g = ((-2.4*t+.36)*t+1.84)*t+.28;
    double b = (((11.29*t-19.02)*t+7.83)*t-.76)*t+.95;
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LightTemperatureMap(double t) {
    double r = (((4.11*t-8.72)*t+4.23)*t+1.04)*t+.18;
    double g = (-2.77*t+2.85)*t+.27;
    double b = (((5.78*t-10.5)*t+4.36)*t-.37)*t+.95;
    return vec3(clp(r),clp(g),clp(b));
  }
};
