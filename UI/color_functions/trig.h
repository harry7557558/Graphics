template<typename vec3, typename Float>
class ColorFunctions {
  static Float clp(Float x) {
    return (Float)(x<0.?0.:x>1.?1.:x);
  }
public:
  static vec3 AlpineColors(Float x) {
    Float r = .39+.51*x+.18*cos(3.14*x+2.18);
    Float g = .2+.98*x+.2*cos(3.14*x+.66);
    Float b = .73-.01*x+.35*cos(3.14*x+2.16);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LakeColors(Float x) {
    Float r = .25+.74*x+.06*cos(3.14*x-.92);
    Float g = .14+.68*x+.24*cos(3.14*x-1.93);
    Float b = .27+.8*x+.34*cos(3.14*x-.83);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ArmyColors(Float x) {
    Float r = .37+.46*x+.07*cos(3.14*x+.44);
    Float g = .33+.71*x+.27*cos(3.14*x+.44);
    Float b = .41+.32*x+.17*cos(3.14*x+1.14);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 MintColors(Float x) {
    Float r = .47+.44*x+.14*cos(3.14*x-1.64);
    Float g = .99-.38*x+.09*cos(3.14*x-1.67);
    Float b = .64+.14*x+.13*cos(3.14*x-1.61);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AtlanticColors(Float x) {
    Float r = 1.61-2.72*x+1.55*cos(3.14*x-2.94)+.28*cos(6.28*x-1.39);
    Float g = .32+.01*x+.42*cos(3.14*x-2.03);
    Float b = .41-.05*x+.45*cos(3.14*x-2.23);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 NeonColors(Float x) {
    Float r = .83-.15*x+.14*cos(3.14*x-2.51);
    Float g = -.12+1.19*x+.95*cos(3.14*x+.08)+.26*cos(6.28*x+1.26);
    Float b = .53-.19*x+.34*cos(3.14*x+2.89)+.12*cos(6.28*x-.8);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AuroraColors(Float x) {
    Float r = 1.18-.83*x+.83*cos(3.14*x+2.62)+.26*cos(6.28*x-2.45);
    Float g = 1.25-2.02*x+1.01*cos(3.14*x-2.87);
    Float b = .02+.71*x+.26*cos(3.14*x-1.55)+.24*cos(6.28*x+.01);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PearlColors(Float x) {
    Float r = .5+.94*x+.55*cos(3.14*x+.64);
    Float g = .16+1.11*x+.56*cos(3.14*x-.11)+.11*cos(6.28*x+.22);
    Float b = .07+1.27*x+.54*cos(3.14*x-.29)+.18*cos(6.28*x+.31);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AvocadoColors(Float x) {
    Float r = .43+.14*x+.46*cos(3.14*x+2.68);
    Float g = -.25+1.48*x+.3*cos(3.14*x-.69);
    Float b = -.11+.44*x+.1*cos(3.14*x+.06);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PlumColors(Float x) {
    Float r = -.73+2.37*x+.71*cos(3.14*x-.05);
    Float g = -.19+1.27*x+.26*cos(3.14*x+.83);
    Float b = .45-.47*x+.48*cos(3.14*x-2.75);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BeachColors(Float x) {
    Float r = .33+1.52*x+.78*cos(3.14*x+.53)+.15*cos(6.28*x+3.08);
    Float g = .03+1.69*x+.62*cos(3.14*x+.36)+.12*cos(6.28*x+3.09);
    Float b = .54-.06*x+.41*cos(3.14*x-2.94)+.12*cos(6.28*x-.02);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RoseColors(Float x) {
    Float r = .3+.49*x+.12*cos(3.14*x-1.85)+.11*cos(6.28*x+3.07);
    Float g = .43-.48*x+.37*cos(3.14*x-1.98);
    Float b = .11-.07*x+.3*cos(3.14*x-1.72);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CandyColors(Float x) {
    Float r = -.07+1.17*x+.53*cos(3.14*x-.61);
    Float g = .51+.04*x+.31*cos(3.14*x+3.09);
    Float b = .8-.38*x+.45*cos(3.14*x-3.05);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SolarColors(Float x) {
    Float r = .26+.95*x+.31*cos(3.14*x-.87);
    Float g = .27+.29*x+.26*cos(3.14*x+3.01);
    Float b = .06+.02*x+.06*cos(3.14*x+2.45);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CMYKColors(Float x) {
    Float r = 1.55-3.03*x+1.77*cos(3.14*x-2.51)+.22*cos(6.28*x-.72);
    Float g = -1.73+3.26*x+2.22*cos(3.14*x-.53)+.85*cos(6.28*x+1.03)+.05*cos(12.57*x+.11);
    Float b = 1.39-2.95*x+1.53*cos(3.14*x-2.35)+.55*cos(6.28*x-.21)+.06*cos(12.57*x+.46);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SouthwestColors(Float x) {
    Float r = -1.09+2.73*x+1.51*cos(3.14*x-.43)+.34*cos(6.28*x+1.29);
    Float g = .99-.78*x+.53*cos(3.14*x+3.12)+.14*cos(6.28*x+3.14);
    Float b = -.13+2.44*x+1.46*cos(3.14*x+.93)+.51*cos(6.28*x+3.02);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DeepSeaColors(Float x) {
    Float r = -.6+2.14*x+.76*cos(3.14*x+.31);
    Float g = .32+.28*x+.34*cos(3.14*x+2.74);
    Float b = .35+.6*x+.19*cos(3.14*x-1.85);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 StarryNightColors(Float x) {
    Float r = .54-.01*x+.43*cos(3.14*x-3.04);
    Float g = .47+.02*x+.38*cos(3.14*x-2.53);
    Float b = .37-.19*x+.39*cos(3.14*x-2.02);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FallColors(Float x) {
    Float r = .56+.16*x+.27*cos(3.14*x+3.);
    Float g = .27+.39*x+.01*cos(3.14*x+2.95)+.12*cos(6.28*x+.01);
    Float b = .48-.31*x+.14*cos(3.14*x+2.25);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SunsetColors(Float x) {
    Float r = 1.-.81*x+.94*cos(3.14*x-2.85)+.25*cos(6.28*x-1.98);
    Float g = .27+.5*x+.25*cos(3.14*x+2.92);
    Float b = .14-.07*x+.67*cos(3.14*x-2.58)+.58*cos(6.28*x-.86)+.06*cos(12.57*x-1.68);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FruitPunchColors(Float x) {
    Float r = .81+.4*x+.3*cos(3.14*x+.76);
    Float g = .03+.75*x+.48*cos(3.14*x-.44);
    Float b = .74-1.31*x+.97*cos(3.14*x-2.78)+.17*cos(6.28*x-.08);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ThermometerColors(Float x) {
    Float r = .45-.28*x+.63*cos(3.14*x-2.12);
    Float g = -.14+.38*x+.86*cos(3.14*x-1.34);
    Float b = .43+.1*x+.5*cos(3.14*x-.76);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 IslandColors(Float x) {
    Float r = 2.03-2.23*x+1.09*cos(3.14*x+2.86)+.2*cos(6.28*x-2.98);
    Float g = .25+.69*x+.36*cos(3.14*x-1.17);
    Float b = -1.17+2.65*x+1.51*cos(3.14*x-.58)+.21*cos(6.28*x+1.09);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 WatermelonColors(Float x) {
    Float r = .25+.26*x+.47*cos(3.14*x-2.17)+.11*cos(6.28*x+.01);
    Float g = .42-.66*x+.96*cos(3.14*x-2.07)+.12*cos(6.28*x-.03);
    Float b = .81-1.54*x+1.13*cos(3.14*x-2.47)+.17*cos(6.28*x-.05);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrassTones(Float x) {
    Float r = -1.03+2.84*x+1.45*cos(3.14*x-.26)+.36*cos(6.28*x+2.08)+.08*cos(12.57*x+2.19);
    Float g = -.89+2.55*x+1.3*cos(3.14*x-.24)+.34*cos(6.28*x+2.1)+.07*cos(12.57*x+2.17);
    Float b = .07-.07*x+.38*cos(3.14*x-1.7);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GreenPinkTones(Float x) {
    Float r = -3.25+6.9*x+3.37*cos(3.14*x-.17)+1.01*cos(6.28*x+1.7)+.11*cos(12.57*x+1.25);
    Float g = 3.08-6.12*x+3.09*cos(3.14*x-2.85)+.89*cos(6.28*x-1.51)+.08*cos(12.57*x-.96);
    Float b = -3.34+6.89*x+3.41*cos(3.14*x-.22)+.99*cos(6.28*x+1.61)+.1*cos(12.57*x+1.12);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrownCyanTones(Float x) {
    Float r = .11+.45*x+.53*cos(3.14*x-1.15);
    Float g = .44+.01*x+.5*cos(3.14*x-2.04);
    Float b = .38+.11*x+.51*cos(3.14*x-2.18);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PigeonTones(Float x) {
    Float r = .72-.52*x+.67*cos(3.14*x-2.92)+.17*cos(6.28*x-.69);
    Float g = .15+.94*x+.1*cos(3.14*x+1.13);
    Float b = .66-.32*x+.56*cos(3.14*x-2.92)+.16*cos(6.28*x-.85);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CherryTones(Float x) {
    Float r = -.22+1.7*x+.51*cos(3.14*x-.5);
    Float g = .34+.51*x+.31*cos(3.14*x+2.03);
    Float b = .36+.47*x+.31*cos(3.14*x+2.1);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RedBlueTones(Float x) {
    Float r = -.19+1.15*x+.81*cos(3.14*x-.46)+.14*cos(6.28*x+2.33);
    Float g = 1.25-1.86*x+1.09*cos(3.14*x-2.73)+.16*cos(6.28*x-2.13);
    Float b = .74-.7*x+.62*cos(3.14*x-2.54);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CoffeeTones(Float x) {
    Float r = .31+.8*x+.13*cos(3.14*x-.61);
    Float g = .17+1.03*x+.18*cos(3.14*x+.39);
    Float b = -.16+1.66*x+.55*cos(3.14*x+.61);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RustTones(Float x) {
    Float r = .12+.96*x+.06*cos(3.14*x-1.97)+.09*cos(6.28*x-3.09);
    Float g = -.01+.47*x+.12*cos(3.14*x-1.59);
    Float b = .19-.15*x+.04*cos(3.14*x+1.54);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FuchsiaTones(Float x) {
    Float r = .21+.66*x+.19*cos(3.14*x-2.17);
    Float g = .38+.33*x+.26*cos(3.14*x+2.84);
    Float b = .23+.63*x+.16*cos(3.14*x-2.4);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SiennaTones(Float x) {
    Float r = .24+.89*x+.3*cos(3.14*x-.82);
    Float g = .36+.34*x+.19*cos(3.14*x-2.8);
    Float b = .38+.12*x+.32*cos(3.14*x+2.77);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GrayTones(Float x) {
    Float r = .2+.65*x+.13*cos(3.14*x+2.31);
    Float g = .16+.74*x+.08*cos(3.14*x+2.16);
    Float b = .11+.79*x+.03*cos(3.14*x+1.59);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ValentineTones(Float x) {
    Float r = .63+.23*x+.11*cos(3.14*x-2.98);
    Float g = .28+.41*x+.19*cos(3.14*x+2.54);
    Float b = .38+.34*x+.18*cos(3.14*x+2.73);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GrayYellowTones(Float x) {
    Float r = .53+.05*x+.38*cos(3.14*x-2.77);
    Float g = .69-.41*x+.52*cos(3.14*x-2.68);
    Float b = 1.04-1.48*x+.85*cos(3.14*x-2.55);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkTerrain(Float x) {
    Float r = -.93+2.85*x+.9*cos(3.14*x+.04);
    Float g = -.95+3.2*x+1.14*cos(3.14*x+.27)+.1*cos(6.28*x-2.62);
    Float b = -.09+2.08*x+1.11*cos(3.14*x+.83)+.21*cos(6.28*x-2.66);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LightTerrain(Float x) {
    Float r = .84-.25*x+.3*cos(3.14*x-3.1);
    Float g = 1.18-.71*x+.42*cos(3.14*x+2.89);
    Float b = 1.26-.79*x+.52*cos(3.14*x+2.48);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GreenBrownTerrain(Float x) {
    Float r = -.46+2.2*x+.59*cos(3.14*x+.15)+.15*cos(6.28*x+2.83);
    Float g = -.72+2.76*x+.88*cos(3.14*x+.19)+.16*cos(6.28*x-3.03);
    Float b = -1.24+3.41*x+1.2*cos(3.14*x+.08);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SandyTerrain(Float x) {
    Float r = -.27+1.57*x+1.03*cos(3.14*x-.38)+.22*cos(6.28*x+1.79);
    Float g = .24+.1*x+.48*cos(3.14*x-1.53);
    Float b = .14+.09*x+.13*cos(3.14*x-1.11);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrightBands(Float x) {
    Float r = -10.69+23.11*x+11.44*cos(3.14*x+.02)+2.54*cos(6.28*x+1.49)+.2*cos(12.57*x+1.91);
    Float g = -13.44+28.36*x+13.75*cos(3.14*x+0.)+2.84*cos(6.28*x+1.66)+.19*cos(12.57*x+1.75);
    Float b = -14.93+31.5*x+15.46*cos(3.14*x+.02)+2.76*cos(6.28*x+1.68)+.36*cos(12.57*x+2.05);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkBands(Float x) {
    Float r = 19.05-36.56*x+18.26*cos(3.14*x+3.13)+3.78*cos(6.28*x-1.6)+.32*cos(12.57*x-1.2);
    Float g = 22.62-43.6*x+21.63*cos(3.14*x+3.13)+4.47*cos(6.28*x-1.56)+.41*cos(12.57*x-1.57);
    Float b = -12.71+26.79*x+13.6*cos(3.14*x+.01)+3.04*cos(6.28*x+1.56)+.31*cos(12.57*x+1.66);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Aquamarine(Float x) {
    Float r = .08+1.16*x+.59*cos(3.14*x+.09);
    Float g = .37+.74*x+.36*cos(3.14*x+.01);
    Float b = .54+.63*x+.32*cos(3.14*x+.23);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Pastel(Float x) {
    Float r = 1.13-1.06*x+.51*cos(3.14*x-2.31);
    Float g = .89-.57*x+.47*cos(3.14*x-2.49);
    Float b = -1.28+3.61*x+1.84*cos(3.14*x-.2)+.54*cos(6.28*x+.87)+.07*cos(12.57*x+.6);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BlueGreenYellow(Float x) {
    Float r = .01+1.*x+.35*cos(3.14*x+1.29);
    Float g = -.09+1.09*x+.22*cos(3.14*x-1.14);
    Float b = .22+.31*x+.23*cos(3.14*x-.72);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Rainbow(Float x) {
    Float r = 1.42-1.57*x+.99*cos(3.14*x+3.03);
    Float g = 1.64-3.2*x+1.76*cos(3.14*x-2.71)+.24*cos(6.28*x-1.44);
    Float b = -.15+.38*x+.77*cos(3.14*x-1.05)+.28*cos(6.28*x-.39);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkRainbow(Float x) {
    Float r = -2.48+6.03*x+2.74*cos(3.14*x-.01)+.73*cos(6.28*x+1.65);
    Float g = -1.09+3.61*x+1.88*cos(3.14*x+.18)+.6*cos(6.28*x+2.19)+.06*cos(12.57*x-2.77);
    Float b = -1.47+3.66*x+2.*cos(3.14*x+.02)+.49*cos(6.28*x+1.47);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 TemperatureMap(Float x) {
    Float r = 1.23-1.23*x+.96*cos(3.14*x-2.87)+.23*cos(6.28*x-2.13);
    Float g = .85-1.25*x+.95*cos(3.14*x-2.17);
    Float b = 2.05-3.22*x+1.27*cos(3.14*x-2.84)+.39*cos(6.28*x-1.56)+.09*cos(12.57*x-.03);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LightTemperatureMap(Float x) {
    Float r = -.33+1.85*x+.62*cos(3.14*x-.42)+.15*cos(6.28*x+2.12);
    Float g = .27+.15*x+.67*cos(3.14*x-1.52);
    Float b = 1.48-1.54*x+.39*cos(3.14*x-3.01)+.18*cos(6.28*x-2.43);
    return vec3(clp(r),clp(g),clp(b));
  }
};
