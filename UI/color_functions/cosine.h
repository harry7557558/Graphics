template<typename vec3, typename Float>
class ColorFunctions {
  static Float clp(Float x) {
    return (Float)(x<0.?0.:x>1.?1.:x);
  }
public:
  static vec3 AlpineColors(Float t) {
    Float r = .34+.56*t+.15*cos(3.47*t+1.97);
    Float g = .26+.75*t+.09*cos(4.56*t+.16);
    Float b = .43+.3*t+.17*cos(4.86*t+1.11);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LakeColors(Float t) {
    Float r = .25+.73*t+.06*cos(3.28*t-1.);
    Float g = .13+.67*t+.25*cos(3.07*t-1.9);
    Float b = .34+.73*t+.3*cos(3.36*t-.98);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ArmyColors(Float t) {
    Float r = .43+.3*t+.02*cos(12.93*t+.41);
    Float g = .43+.43*t+.14*cos(4.13*t+.06);
    Float b = .89+2.69*t+3.54*cos(.85*t+1.68);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 MintColors(Float t) {
    Float r = -.97+.09*t+1.81*cos(.82*t-.67);
    Float g = .7-.49*t+.45*cos(1.32*t-.9);
    Float b = .07+.08*t+.74*cos(1.24*t-.7);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AtlanticColors(Float t) {
    Float r = -.36-4.38*t+7.37*cos(.73*t-1.51);
    Float g = .31-.07*t+.48*cos(2.92*t-1.96);
    Float b = .46-.51*t+.75*cos(2.48*t-2.03);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 NeonColors(Float t) {
    Float r = .77+.07*t+.04*cos(5.93*t+2.73);
    Float g = .83-.79*t+.14*cos(5.94*t+.76);
    Float b = .14+.51*t+.13*cos(5.69*t+.06);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AuroraColors(Float t) {
    Float r = .21+.59*t+.05*cos(10.52*t+1.43);
    Float g = .51-.23*t+.28*cos(5.46*t+2.47);
    Float b = .17+.7*t+.12*cos(7.58*t-.64);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PearlColors(Float t) {
    Float r = .72+.13*t+.2*cos(5.4*t-.18);
    Float g = .78-.07*t+.11*cos(6.57*t-.89);
    Float b = .71+.15*t+.11*cos(7.19*t-.77);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AvocadoColors(Float t) {
    Float r = .65-.2*t+.65*cos(2.76*t+2.92);
    Float g = -.68+2.05*t+.66*cos(2.28*t-.1);
    Float b = -.01+.25*t+.02*cos(5.9*t-1.25);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PlumColors(Float t) {
    Float r = -1.02+2.94*t+1.01*cos(2.77*t+.15);
    Float g = -2.84+17.04*t+30.48*cos(.544*t+1.478);
    Float b = 2.04-4.71*t+3.86*cos(1.48*t-2.12);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BeachColors(Float t) {
    Float r = -40.16+87.82*t+202.13*cos(.4371*t+1.3668);
    Float g = .56+.36*t+.08*cos(7.61*t-2.56);
    Float b = .15+.8*t+.1*cos(6.82*t+.38);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RoseColors(Float t) {
    Float r = .32+.54*t+.17*cos(5.63*t-2.92);
    Float g = .48-.21*t+.17*cos(5.38*t-2.92);
    Float b = .22+0.*t+.16*cos(4.87*t-2.52);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CandyColors(Float t) {
    Float r = .43+.47*t+.21*cos(4.93*t-1.75);
    Float g = 19.47-36.59*t+79.3*cos(.474*t-1.816);
    Float b = .48+.3*t+.17*cos(4.66*t+2.53);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SolarColors(Float t) {
    Float r = -.33+1.62*t+.78*cos(2.09*t-.13);
    Float g = .03+.73*t+.08*cos(5.13*t+1.92);
    Float b = .17-.13*t+.16*cos(2.13*t+3.14);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CMYKColors(Float t) {
    Float r = 21.34-109.66*t+274.8*cos(.4044*t-1.6474);
    Float g = .96-.7*t+.36*cos(6.26*t+1.69);
    Float b = .86-.54*t+.1*cos(9.08*t-1.02);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SouthwestColors(Float t) {
    Float r = 2.15-26.75*t+51.14*cos(.542*t-1.604);
    Float g = .44+.3*t+.17*cos(6.24*t+2.53);
    Float b = .17+.38*t+.12*cos(11.28*t+1.02);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DeepSeaColors(Float t) {
    Float r = -.45+1.77*t+.59*cos(3.48*t+.17);
    Float g = .39+.17*t+.4*cos(2.95*t+2.86);
    Float b = .22+.37*t+.47*cos(1.93*t-1.41);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 StarryNightColors(Float t) {
    Float r = .19+.74*t+.12*cos(5.25*t+2.27);
    Float g = .49-.07*t+.43*cos(2.99*t-2.48);
    Float b = .39-.02*t+.27*cos(3.94*t-2.33);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FallColors(Float t) {
    Float r = .21+.81*t+.04*cos(8.62*t-.26);
    Float g = .26+.4*t+.12*cos(6.28*t+.02);
    Float b = 6.5-7.17*t+12.76*cos(.571*t-2.069);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SunsetColors(Float t) {
    Float r = .14+1.15*t+.27*cos(5.06*t-2.21);
    Float g = -.05+1.07*t+.04*cos(7.88*t+.1);
    Float b = -.07+.73*t+.32*cos(7.07*t-1.2);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FruitPunchColors(Float t) {
    Float r = .89-.02*t+.11*cos(5.77*t-.21);
    Float g = .57-.09*t+.14*cos(5.77*t-2.09);
    Float b = -.04+.68*t+.14*cos(6.99*t+1.02);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ThermometerColors(Float t) {
    Float r = .45+.12*t+.39*cos(4.18*t-2.51);
    Float g = .28+.14*t+.55*cos(4.18*t-1.92);
    Float b = .46+.05*t+.48*cos(3.22*t-.81);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 IslandColors(Float t) {
    Float r = .67+.12*t+.14*cos(7.54*t+1.45);
    Float g = -8.59+8.*t+12.2*cos(.712*t+.746);
    Float b = -35.37+32.87*t+68.16*cos(.508*t+1.022);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 WatermelonColors(Float t) {
    Float r = 8.38-23.7*t+45.9*cos(.542*t-1.752);
    Float g = 6.22-56.24*t+127.22*cos(.4551*t-1.6188);
    Float b = 80.01-209.12*t+639.31*cos(.3292*t-1.696);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrassTones(Float t) {
    Float r = .16+.07*t+.73*cos(3.31*t-1.67);
    Float g = .24+0.*t+.6*cos(3.49*t-1.8);
    Float b = .12-.04*t+.31*cos(3.55*t-1.89);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GreenPinkTones(Float t) {
    Float r = .53-.05*t+.55*cos(5.5*t+2.78);
    Float g = .21+.51*t+.62*cos(4.82*t-1.55);
    Float b = .6-.21*t+.57*cos(5.27*t+2.86);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrownCyanTones(Float t) {
    Float r = .2+.37*t+.47*cos(3.37*t-1.29);
    Float g = -1.14-14.82*t+26.53*cos(.626*t-1.521);
    Float b = .59-3.1*t+3.44*cos(1.41*t-1.72);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PigeonTones(Float t) {
    Float r = .12+.86*t+.06*cos(7.66*t-.48);
    Float g = .13+.84*t+.04*cos(7.9*t-.99);
    Float b = .17+.79*t+.06*cos(7.72*t-.96);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CherryTones(Float t) {
    Float r = -33.95+52.15*t+107.83*cos(.4822*t+1.2484);
    Float g = 11.9-9.82*t+19.47*cos(.588*t-2.214);
    Float b = 13.78-12.63*t+25.21*cos(.562*t-2.139);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RedBlueTones(Float t) {
    Float r = .65-.25*t+.33*cos(4.74*t-2.17);
    Float g = .46+.01*t+.39*cos(4.49*t-2.54);
    Float b = .91-1.3*t+.96*cos(2.62*t-2.37);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CoffeeTones(Float t) {
    Float r = -2.68+4.95*t+5.69*cos(.8*t+1.);
    Float g = .32+.64*t+.04*cos(7.86*t-1.39);
    Float b = .06+.86*t+.2*cos(5.28*t-.16);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RustTones(Float t) {
    Float r = .12+1.02*t+.12*cos(5.86*t-2.93);
    Float g = .06+.47*t+.06*cos(5.84*t-2.93);
    Float b = .17-.16*t+.02*cos(5.85*t+.2);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FuchsiaTones(Float t) {
    Float r = .39-3.14*t+5.33*cos(.83*t-1.63);
    Float g = .03+.91*t+.05*cos(7.86*t-.06);
    Float b = 1.44-4.92*t+8.11*cos(.75*t-1.74);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SiennaTones(Float t) {
    Float r = -.41+1.68*t+.86*cos(2.03*t-.04);
    Float g = .33+.42*t+.15*cos(3.42*t-2.91);
    Float b = .4+.09*t+.34*cos(3.09*t+2.8);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GrayTones(Float t) {
    Float r = .04+.84*t+.04*cos(7.06*t-.08);
    Float g = .06+.84*t+.03*cos(7.43*t-.36);
    Float b = .09+.79*t+.02*cos(9.*t-1.35);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ValentineTones(Float t) {
    Float r = 2.39-3.68*t+5.19*cos(.82*t-1.94);
    Float g = .14+.6*t+.1*cos(4.26*t+1.82);
    Float b = .17+.66*t+.05*cos(5.68*t+1.15);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GrayYellowTones(Float t) {
    Float r = .6-.13*t+.47*cos(2.89*t-2.67);
    Float g = 22.-61.93*t+141.55*cos(.4457*t-1.7253);
    Float b = 40.32-135.25*t+364.05*cos(.3746*t-1.6809);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkTerrain(Float t) {
    Float r = -119.59+246.86*t+786.34*cos(.3139*t+1.4182);
    Float g = -5.32+12.06*t+7.19*cos(1.75*t+.74);
    Float b = -180.09+452.81*t+1665.28*cos(.27247*t+1.4622);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LightTerrain(Float t) {
    Float r = .59+.26*t+.09*cos(5.12*t+2.23);
    Float g = 27.29-45.82*t+94.81*cos(.49*t-1.854);
    Float b = 39.02-51.94*t+114.12*cos(.4623*t-1.9117);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GreenBrownTerrain(Float t) {
    Float r = .12+.95*t+.12*cos(6.76*t-2.89);
    Float g = .12+.88*t+.18*cos(6.8*t-2.34);
    Float b = -205.56+434.26*t+1593.1*cos(.27279*t+1.44143);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SandyTerrain(Float t) {
    Float r = .9-.54*t+.32*cos(4.28*t-2.37);
    Float g = .48+.07*t+.27*cos(4.7*t-2.32);
    Float b = .26-.03*t+.06*cos(5.68*t-2.62);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrightBands(Float t) {
    Float r = .63+.13*t+.31*cos(7.39*t-.03);
    Float g = .52+.31*t+.29*cos(5.46*t-3.02);
    Float b = -577.52+1052.6*t+4334.34*cos(.24336*t+1.43714);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkBands(Float t) {
    Float r = .52+.31*t+.15*cos(11.59*t+.78);
    Float g = .62+.21*t+.26*cos(4.3*t+.76);
    Float b = 1.08-.98*t+.35*cos(5.11*t+1.76);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Aquamarine(Float t) {
    Float r = -.3+1.96*t+1.04*cos(2.57*t+.36);
    Float g = .26+.95*t+.47*cos(2.85*t+.15);
    Float b = -.2+2.36*t+1.51*cos(1.8*t+.8);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Pastel(Float t) {
    Float r = 5.99-30.91*t+57.32*cos(.549*t-1.662);
    Float g = 1.69-3.63*t+2.89*cos(1.56*t-1.99);
    Float b = .81-.07*t+.22*cos(5.58*t+.59);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BlueGreenYellow(Float t) {
    Float r = 2.08+4.68*t+6.45*cos(.82*t+1.88);
    Float g = -.47+1.41*t+.5*cos(2.07*t-.42);
    Float b = -1.06+1.97*t+1.61*cos(1.48*t+.45);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Rainbow(Float t) {
    Float r = 132.23-245.97*t+755.63*cos(.3275*t-1.7461);
    Float g = .39-1.4*t+1.32*cos(2.39*t-1.84);
    Float b = -142.83+270.69*t+891.31*cos(.3053*t+1.4092);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkRainbow(Float t) {
    Float r = .25+.64*t+.16*cos(7.89*t+1.19);
    Float g = .65-.34*t+.28*cos(5.83*t+2.69);
    Float b = .52-.4*t+.11*cos(6.93*t+.6);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 TemperatureMap(Float t) {
    Float r = .37+.71*t+.26*cos(5.2*t-2.51);
    Float g = .89-2.12*t+1.56*cos(2.48*t-1.96);
    Float b = 1.18-.94*t+.2*cos(8.03*t+2.87);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LightTemperatureMap(Float t) {
    Float r = .38+.62*t+.24*cos(4.9*t-2.61);
    Float g = -5.49+.96*t+6.09*cos(.97*t-.33);
    Float b = 1.11-.73*t+.17*cos(6.07*t-2.74);
    return vec3(clp(r),clp(g),clp(b));
  }
};
