const ColorFunctions = {
  clp: function(x) {
    return Math.round(255.*(x<0.?0.:x>1.?1.:x));
  },
  tocol: function(r, g, b) {
    return 'rgb('+this.clp(r)+','+this.clp(g)+','+this.clp(b)+')';
  },
  AlpineColors: function(x) {
    var r = .39+.51*x+.18*Math.cos(3.14*x+2.18);
    var g = .2+.98*x+.2*Math.cos(3.14*x+.66);
    var b = .73-.01*x+.35*Math.cos(3.14*x+2.16);
    return this.tocol(r, g, b);
  },
  LakeColors: function(x) {
    var r = .25+.74*x+.06*Math.cos(3.14*x-.92);
    var g = .14+.68*x+.24*Math.cos(3.14*x-1.93);
    var b = .27+.8*x+.34*Math.cos(3.14*x-.83);
    return this.tocol(r, g, b);
  },
  ArmyColors: function(x) {
    var r = .37+.46*x+.07*Math.cos(3.14*x+.44);
    var g = .33+.71*x+.27*Math.cos(3.14*x+.44);
    var b = .41+.32*x+.17*Math.cos(3.14*x+1.14);
    return this.tocol(r, g, b);
  },
  MintColors: function(x) {
    var r = .47+.44*x+.14*Math.cos(3.14*x-1.64);
    var g = .99-.38*x+.09*Math.cos(3.14*x-1.67);
    var b = .64+.14*x+.13*Math.cos(3.14*x-1.61);
    return this.tocol(r, g, b);
  },
  AtlanticColors: function(x) {
    var r = 1.61-2.72*x+1.55*Math.cos(3.14*x-2.94)+.28*Math.cos(6.28*x-1.39);
    var g = .32+.01*x+.42*Math.cos(3.14*x-2.03);
    var b = .41-.05*x+.45*Math.cos(3.14*x-2.23);
    return this.tocol(r, g, b);
  },
  NeonColors: function(x) {
    var r = .83-.15*x+.14*Math.cos(3.14*x-2.51);
    var g = -.12+1.19*x+.95*Math.cos(3.14*x+.08)+.26*Math.cos(6.28*x+1.26);
    var b = .53-.19*x+.34*Math.cos(3.14*x+2.89)+.12*Math.cos(6.28*x-.8);
    return this.tocol(r, g, b);
  },
  AuroraColors: function(x) {
    var r = 1.18-.83*x+.83*Math.cos(3.14*x+2.62)+.26*Math.cos(6.28*x-2.45);
    var g = 1.25-2.02*x+1.01*Math.cos(3.14*x-2.87);
    var b = .02+.71*x+.26*Math.cos(3.14*x-1.55)+.24*Math.cos(6.28*x+.01);
    return this.tocol(r, g, b);
  },
  PearlColors: function(x) {
    var r = .5+.94*x+.55*Math.cos(3.14*x+.64);
    var g = .16+1.11*x+.56*Math.cos(3.14*x-.11)+.11*Math.cos(6.28*x+.22);
    var b = .07+1.27*x+.54*Math.cos(3.14*x-.29)+.18*Math.cos(6.28*x+.31);
    return this.tocol(r, g, b);
  },
  AvocadoColors: function(x) {
    var r = .43+.14*x+.46*Math.cos(3.14*x+2.68);
    var g = -.25+1.48*x+.3*Math.cos(3.14*x-.69);
    var b = -.11+.44*x+.1*Math.cos(3.14*x+.06);
    return this.tocol(r, g, b);
  },
  PlumColors: function(x) {
    var r = -.73+2.37*x+.71*Math.cos(3.14*x-.05);
    var g = -.19+1.27*x+.26*Math.cos(3.14*x+.83);
    var b = .45-.47*x+.48*Math.cos(3.14*x-2.75);
    return this.tocol(r, g, b);
  },
  BeachColors: function(x) {
    var r = .33+1.52*x+.78*Math.cos(3.14*x+.53)+.15*Math.cos(6.28*x+3.08);
    var g = .03+1.69*x+.62*Math.cos(3.14*x+.36)+.12*Math.cos(6.28*x+3.09);
    var b = .54-.06*x+.41*Math.cos(3.14*x-2.94)+.12*Math.cos(6.28*x-.02);
    return this.tocol(r, g, b);
  },
  RoseColors: function(x) {
    var r = .3+.49*x+.12*Math.cos(3.14*x-1.85)+.11*Math.cos(6.28*x+3.07);
    var g = .43-.48*x+.37*Math.cos(3.14*x-1.98);
    var b = .11-.07*x+.3*Math.cos(3.14*x-1.72);
    return this.tocol(r, g, b);
  },
  CandyColors: function(x) {
    var r = -.07+1.17*x+.53*Math.cos(3.14*x-.61);
    var g = .51+.04*x+.31*Math.cos(3.14*x+3.09);
    var b = .8-.38*x+.45*Math.cos(3.14*x-3.05);
    return this.tocol(r, g, b);
  },
  SolarColors: function(x) {
    var r = .26+.95*x+.31*Math.cos(3.14*x-.87);
    var g = .27+.29*x+.26*Math.cos(3.14*x+3.01);
    var b = .06+.02*x+.06*Math.cos(3.14*x+2.45);
    return this.tocol(r, g, b);
  },
  CMYKColors: function(x) {
    var r = 1.55-3.03*x+1.77*Math.cos(3.14*x-2.51)+.22*Math.cos(6.28*x-.72);
    var g = -1.73+3.26*x+2.22*Math.cos(3.14*x-.53)+.85*Math.cos(6.28*x+1.03)+.05*Math.cos(12.57*x+.11);
    var b = 1.39-2.95*x+1.53*Math.cos(3.14*x-2.35)+.55*Math.cos(6.28*x-.21)+.06*Math.cos(12.57*x+.46);
    return this.tocol(r, g, b);
  },
  SouthwestColors: function(x) {
    var r = -1.09+2.73*x+1.51*Math.cos(3.14*x-.43)+.34*Math.cos(6.28*x+1.29);
    var g = .99-.78*x+.53*Math.cos(3.14*x+3.12)+.14*Math.cos(6.28*x+3.14);
    var b = -.13+2.44*x+1.46*Math.cos(3.14*x+.93)+.51*Math.cos(6.28*x+3.02);
    return this.tocol(r, g, b);
  },
  DeepSeaColors: function(x) {
    var r = -.6+2.14*x+.76*Math.cos(3.14*x+.31);
    var g = .32+.28*x+.34*Math.cos(3.14*x+2.74);
    var b = .35+.6*x+.19*Math.cos(3.14*x-1.85);
    return this.tocol(r, g, b);
  },
  StarryNightColors: function(x) {
    var r = .54-.01*x+.43*Math.cos(3.14*x-3.04);
    var g = .47+.02*x+.38*Math.cos(3.14*x-2.53);
    var b = .37-.19*x+.39*Math.cos(3.14*x-2.02);
    return this.tocol(r, g, b);
  },
  FallColors: function(x) {
    var r = .56+.16*x+.27*Math.cos(3.14*x+3.);
    var g = .27+.39*x+.01*Math.cos(3.14*x+2.95)+.12*Math.cos(6.28*x+.01);
    var b = .48-.31*x+.14*Math.cos(3.14*x+2.25);
    return this.tocol(r, g, b);
  },
  SunsetColors: function(x) {
    var r = 1.-.81*x+.94*Math.cos(3.14*x-2.85)+.25*Math.cos(6.28*x-1.98);
    var g = .27+.5*x+.25*Math.cos(3.14*x+2.92);
    var b = .14-.07*x+.67*Math.cos(3.14*x-2.58)+.58*Math.cos(6.28*x-.86)+.06*Math.cos(12.57*x-1.68);
    return this.tocol(r, g, b);
  },
  FruitPunchColors: function(x) {
    var r = .81+.4*x+.3*Math.cos(3.14*x+.76);
    var g = .03+.75*x+.48*Math.cos(3.14*x-.44);
    var b = .74-1.31*x+.97*Math.cos(3.14*x-2.78)+.17*Math.cos(6.28*x-.08);
    return this.tocol(r, g, b);
  },
  ThermometerColors: function(x) {
    var r = .45-.28*x+.63*Math.cos(3.14*x-2.12);
    var g = -.14+.38*x+.86*Math.cos(3.14*x-1.34);
    var b = .43+.1*x+.5*Math.cos(3.14*x-.76);
    return this.tocol(r, g, b);
  },
  IslandColors: function(x) {
    var r = 2.03-2.23*x+1.09*Math.cos(3.14*x+2.86)+.2*Math.cos(6.28*x-2.98);
    var g = .25+.69*x+.36*Math.cos(3.14*x-1.17);
    var b = -1.17+2.65*x+1.51*Math.cos(3.14*x-.58)+.21*Math.cos(6.28*x+1.09);
    return this.tocol(r, g, b);
  },
  WatermelonColors: function(x) {
    var r = .25+.26*x+.47*Math.cos(3.14*x-2.17)+.11*Math.cos(6.28*x+.01);
    var g = .42-.66*x+.96*Math.cos(3.14*x-2.07)+.12*Math.cos(6.28*x-.03);
    var b = .81-1.54*x+1.13*Math.cos(3.14*x-2.47)+.17*Math.cos(6.28*x-.05);
    return this.tocol(r, g, b);
  },
  BrassTones: function(x) {
    var r = -1.03+2.84*x+1.45*Math.cos(3.14*x-.26)+.36*Math.cos(6.28*x+2.08)+.08*Math.cos(12.57*x+2.19);
    var g = -.89+2.55*x+1.3*Math.cos(3.14*x-.24)+.34*Math.cos(6.28*x+2.1)+.07*Math.cos(12.57*x+2.17);
    var b = .07-.07*x+.38*Math.cos(3.14*x-1.7);
    return this.tocol(r, g, b);
  },
  GreenPinkTones: function(x) {
    var r = -3.25+6.9*x+3.37*Math.cos(3.14*x-.17)+1.01*Math.cos(6.28*x+1.7)+.11*Math.cos(12.57*x+1.25);
    var g = 3.08-6.12*x+3.09*Math.cos(3.14*x-2.85)+.89*Math.cos(6.28*x-1.51)+.08*Math.cos(12.57*x-.96);
    var b = -3.34+6.89*x+3.41*Math.cos(3.14*x-.22)+.99*Math.cos(6.28*x+1.61)+.1*Math.cos(12.57*x+1.12);
    return this.tocol(r, g, b);
  },
  BrownCyanTones: function(x) {
    var r = .11+.45*x+.53*Math.cos(3.14*x-1.15);
    var g = .44+.01*x+.5*Math.cos(3.14*x-2.04);
    var b = .38+.11*x+.51*Math.cos(3.14*x-2.18);
    return this.tocol(r, g, b);
  },
  PigeonTones: function(x) {
    var r = .72-.52*x+.67*Math.cos(3.14*x-2.92)+.17*Math.cos(6.28*x-.69);
    var g = .15+.94*x+.1*Math.cos(3.14*x+1.13);
    var b = .66-.32*x+.56*Math.cos(3.14*x-2.92)+.16*Math.cos(6.28*x-.85);
    return this.tocol(r, g, b);
  },
  CherryTones: function(x) {
    var r = -.22+1.7*x+.51*Math.cos(3.14*x-.5);
    var g = .34+.51*x+.31*Math.cos(3.14*x+2.03);
    var b = .36+.47*x+.31*Math.cos(3.14*x+2.1);
    return this.tocol(r, g, b);
  },
  RedBlueTones: function(x) {
    var r = -.19+1.15*x+.81*Math.cos(3.14*x-.46)+.14*Math.cos(6.28*x+2.33);
    var g = 1.25-1.86*x+1.09*Math.cos(3.14*x-2.73)+.16*Math.cos(6.28*x-2.13);
    var b = .74-.7*x+.62*Math.cos(3.14*x-2.54);
    return this.tocol(r, g, b);
  },
  CoffeeTones: function(x) {
    var r = .31+.8*x+.13*Math.cos(3.14*x-.61);
    var g = .17+1.03*x+.18*Math.cos(3.14*x+.39);
    var b = -.16+1.66*x+.55*Math.cos(3.14*x+.61);
    return this.tocol(r, g, b);
  },
  RustTones: function(x) {
    var r = .12+.96*x+.06*Math.cos(3.14*x-1.97)+.09*Math.cos(6.28*x-3.09);
    var g = -.01+.47*x+.12*Math.cos(3.14*x-1.59);
    var b = .19-.15*x+.04*Math.cos(3.14*x+1.54);
    return this.tocol(r, g, b);
  },
  FuchsiaTones: function(x) {
    var r = .21+.66*x+.19*Math.cos(3.14*x-2.17);
    var g = .38+.33*x+.26*Math.cos(3.14*x+2.84);
    var b = .23+.63*x+.16*Math.cos(3.14*x-2.4);
    return this.tocol(r, g, b);
  },
  SiennaTones: function(x) {
    var r = .24+.89*x+.3*Math.cos(3.14*x-.82);
    var g = .36+.34*x+.19*Math.cos(3.14*x-2.8);
    var b = .38+.12*x+.32*Math.cos(3.14*x+2.77);
    return this.tocol(r, g, b);
  },
  GrayTones: function(x) {
    var r = .2+.65*x+.13*Math.cos(3.14*x+2.31);
    var g = .16+.74*x+.08*Math.cos(3.14*x+2.16);
    var b = .11+.79*x+.03*Math.cos(3.14*x+1.59);
    return this.tocol(r, g, b);
  },
  ValentineTones: function(x) {
    var r = .63+.23*x+.11*Math.cos(3.14*x-2.98);
    var g = .28+.41*x+.19*Math.cos(3.14*x+2.54);
    var b = .38+.34*x+.18*Math.cos(3.14*x+2.73);
    return this.tocol(r, g, b);
  },
  GrayYellowTones: function(x) {
    var r = .53+.05*x+.38*Math.cos(3.14*x-2.77);
    var g = .69-.41*x+.52*Math.cos(3.14*x-2.68);
    var b = 1.04-1.48*x+.85*Math.cos(3.14*x-2.55);
    return this.tocol(r, g, b);
  },
  DarkTerrain: function(x) {
    var r = -.93+2.85*x+.9*Math.cos(3.14*x+.04);
    var g = -.95+3.2*x+1.14*Math.cos(3.14*x+.27)+.1*Math.cos(6.28*x-2.62);
    var b = -.09+2.08*x+1.11*Math.cos(3.14*x+.83)+.21*Math.cos(6.28*x-2.66);
    return this.tocol(r, g, b);
  },
  LightTerrain: function(x) {
    var r = .84-.25*x+.3*Math.cos(3.14*x-3.1);
    var g = 1.18-.71*x+.42*Math.cos(3.14*x+2.89);
    var b = 1.26-.79*x+.52*Math.cos(3.14*x+2.48);
    return this.tocol(r, g, b);
  },
  GreenBrownTerrain: function(x) {
    var r = -.46+2.2*x+.59*Math.cos(3.14*x+.15)+.15*Math.cos(6.28*x+2.83);
    var g = -.72+2.76*x+.88*Math.cos(3.14*x+.19)+.16*Math.cos(6.28*x-3.03);
    var b = -1.24+3.41*x+1.2*Math.cos(3.14*x+.08);
    return this.tocol(r, g, b);
  },
  SandyTerrain: function(x) {
    var r = -.27+1.57*x+1.03*Math.cos(3.14*x-.38)+.22*Math.cos(6.28*x+1.79);
    var g = .24+.1*x+.48*Math.cos(3.14*x-1.53);
    var b = .14+.09*x+.13*Math.cos(3.14*x-1.11);
    return this.tocol(r, g, b);
  },
  BrightBands: function(x) {
    var r = -10.69+23.11*x+11.44*Math.cos(3.14*x+.02)+2.54*Math.cos(6.28*x+1.49)+.2*Math.cos(12.57*x+1.91);
    var g = -13.44+28.36*x+13.75*Math.cos(3.14*x+0.)+2.84*Math.cos(6.28*x+1.66)+.19*Math.cos(12.57*x+1.75);
    var b = -14.93+31.5*x+15.46*Math.cos(3.14*x+.02)+2.76*Math.cos(6.28*x+1.68)+.36*Math.cos(12.57*x+2.05);
    return this.tocol(r, g, b);
  },
  DarkBands: function(x) {
    var r = 19.05-36.56*x+18.26*Math.cos(3.14*x+3.13)+3.78*Math.cos(6.28*x-1.6)+.32*Math.cos(12.57*x-1.2);
    var g = 22.62-43.6*x+21.63*Math.cos(3.14*x+3.13)+4.47*Math.cos(6.28*x-1.56)+.41*Math.cos(12.57*x-1.57);
    var b = -12.71+26.79*x+13.6*Math.cos(3.14*x+.01)+3.04*Math.cos(6.28*x+1.56)+.31*Math.cos(12.57*x+1.66);
    return this.tocol(r, g, b);
  },
  Aquamarine: function(x) {
    var r = .08+1.16*x+.59*Math.cos(3.14*x+.09);
    var g = .37+.74*x+.36*Math.cos(3.14*x+.01);
    var b = .54+.63*x+.32*Math.cos(3.14*x+.23);
    return this.tocol(r, g, b);
  },
  Pastel: function(x) {
    var r = 1.13-1.06*x+.51*Math.cos(3.14*x-2.31);
    var g = .89-.57*x+.47*Math.cos(3.14*x-2.49);
    var b = -1.28+3.61*x+1.84*Math.cos(3.14*x-.2)+.54*Math.cos(6.28*x+.87)+.07*Math.cos(12.57*x+.6);
    return this.tocol(r, g, b);
  },
  BlueGreenYellow: function(x) {
    var r = .01+1.*x+.35*Math.cos(3.14*x+1.29);
    var g = -.09+1.09*x+.22*Math.cos(3.14*x-1.14);
    var b = .22+.31*x+.23*Math.cos(3.14*x-.72);
    return this.tocol(r, g, b);
  },
  Rainbow: function(x) {
    var r = 1.42-1.57*x+.99*Math.cos(3.14*x+3.03);
    var g = 1.64-3.2*x+1.76*Math.cos(3.14*x-2.71)+.24*Math.cos(6.28*x-1.44);
    var b = -.15+.38*x+.77*Math.cos(3.14*x-1.05)+.28*Math.cos(6.28*x-.39);
    return this.tocol(r, g, b);
  },
  DarkRainbow: function(x) {
    var r = -2.48+6.03*x+2.74*Math.cos(3.14*x-.01)+.73*Math.cos(6.28*x+1.65);
    var g = -1.09+3.61*x+1.88*Math.cos(3.14*x+.18)+.6*Math.cos(6.28*x+2.19)+.06*Math.cos(12.57*x-2.77);
    var b = -1.47+3.66*x+2.*Math.cos(3.14*x+.02)+.49*Math.cos(6.28*x+1.47);
    return this.tocol(r, g, b);
  },
  TemperatureMap: function(x) {
    var r = 1.23-1.23*x+.96*Math.cos(3.14*x-2.87)+.23*Math.cos(6.28*x-2.13);
    var g = .85-1.25*x+.95*Math.cos(3.14*x-2.17);
    var b = 2.05-3.22*x+1.27*Math.cos(3.14*x-2.84)+.39*Math.cos(6.28*x-1.56)+.09*Math.cos(12.57*x-.03);
    return this.tocol(r, g, b);
  },
  LightTemperatureMap: function(x) {
    var r = -.33+1.85*x+.62*Math.cos(3.14*x-.42)+.15*Math.cos(6.28*x+2.12);
    var g = .27+.15*x+.67*Math.cos(3.14*x-1.52);
    var b = 1.48-1.54*x+.39*Math.cos(3.14*x-3.01)+.18*Math.cos(6.28*x-2.43);
    return this.tocol(r, g, b);
  },
};