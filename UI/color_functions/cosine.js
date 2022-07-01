const ColorFunctions = {
  clp: function(x) {
    return Math.round(255.*(x<0.?0.:x>1.?1.:x));
  },
  tocol: function(r, g, b) {
    return 'rgb('+this.clp(r)+','+this.clp(g)+','+this.clp(b)+')';
  },
  AlpineColors: function(t) {
    var r = .34+.56*t+.15*Math.cos(3.47*t+1.97);
    var g = .26+.75*t+.09*Math.cos(4.56*t+.16);
    var b = .43+.3*t+.17*Math.cos(4.86*t+1.11);
    return this.tocol(r, g, b);
  },
  LakeColors: function(t) {
    var r = .25+.73*t+.06*Math.cos(3.28*t-1.);
    var g = .13+.67*t+.25*Math.cos(3.07*t-1.9);
    var b = .34+.73*t+.3*Math.cos(3.36*t-.98);
    return this.tocol(r, g, b);
  },
  ArmyColors: function(t) {
    var r = .43+.3*t+.02*Math.cos(12.93*t+.41);
    var g = .43+.43*t+.14*Math.cos(4.13*t+.06);
    var b = .89+2.69*t+3.54*Math.cos(.85*t+1.68);
    return this.tocol(r, g, b);
  },
  MintColors: function(t) {
    var r = -.97+.09*t+1.81*Math.cos(.82*t-.67);
    var g = .7-.49*t+.45*Math.cos(1.32*t-.9);
    var b = .07+.08*t+.74*Math.cos(1.24*t-.7);
    return this.tocol(r, g, b);
  },
  AtlanticColors: function(t) {
    var r = -.36-4.38*t+7.37*Math.cos(.73*t-1.51);
    var g = .31-.07*t+.48*Math.cos(2.92*t-1.96);
    var b = .46-.51*t+.75*Math.cos(2.48*t-2.03);
    return this.tocol(r, g, b);
  },
  NeonColors: function(t) {
    var r = .77+.07*t+.04*Math.cos(5.93*t+2.73);
    var g = .83-.79*t+.14*Math.cos(5.94*t+.76);
    var b = .14+.51*t+.13*Math.cos(5.69*t+.06);
    return this.tocol(r, g, b);
  },
  AuroraColors: function(t) {
    var r = .21+.59*t+.05*Math.cos(10.52*t+1.43);
    var g = .51-.23*t+.28*Math.cos(5.46*t+2.47);
    var b = .17+.7*t+.12*Math.cos(7.58*t-.64);
    return this.tocol(r, g, b);
  },
  PearlColors: function(t) {
    var r = .72+.13*t+.2*Math.cos(5.4*t-.18);
    var g = .78-.07*t+.11*Math.cos(6.57*t-.89);
    var b = .71+.15*t+.11*Math.cos(7.19*t-.77);
    return this.tocol(r, g, b);
  },
  AvocadoColors: function(t) {
    var r = .65-.2*t+.65*Math.cos(2.76*t+2.92);
    var g = -.68+2.05*t+.66*Math.cos(2.28*t-.1);
    var b = -.01+.25*t+.02*Math.cos(5.9*t-1.25);
    return this.tocol(r, g, b);
  },
  PlumColors: function(t) {
    var r = -1.02+2.94*t+1.01*Math.cos(2.77*t+.15);
    var g = -2.84+17.04*t+30.48*Math.cos(.544*t+1.478);
    var b = 2.04-4.71*t+3.86*Math.cos(1.48*t-2.12);
    return this.tocol(r, g, b);
  },
  BeachColors: function(t) {
    var r = -40.16+87.82*t+202.13*Math.cos(.4371*t+1.3668);
    var g = .56+.36*t+.08*Math.cos(7.61*t-2.56);
    var b = .15+.8*t+.1*Math.cos(6.82*t+.38);
    return this.tocol(r, g, b);
  },
  RoseColors: function(t) {
    var r = .32+.54*t+.17*Math.cos(5.63*t-2.92);
    var g = .48-.21*t+.17*Math.cos(5.38*t-2.92);
    var b = .22+0.*t+.16*Math.cos(4.87*t-2.52);
    return this.tocol(r, g, b);
  },
  CandyColors: function(t) {
    var r = .43+.47*t+.21*Math.cos(4.93*t-1.75);
    var g = 19.47-36.59*t+79.3*Math.cos(.474*t-1.816);
    var b = .48+.3*t+.17*Math.cos(4.66*t+2.53);
    return this.tocol(r, g, b);
  },
  SolarColors: function(t) {
    var r = -.33+1.62*t+.78*Math.cos(2.09*t-.13);
    var g = .03+.73*t+.08*Math.cos(5.13*t+1.92);
    var b = .17-.13*t+.16*Math.cos(2.13*t+3.14);
    return this.tocol(r, g, b);
  },
  CMYKColors: function(t) {
    var r = 21.34-109.66*t+274.8*Math.cos(.4044*t-1.6474);
    var g = .96-.7*t+.36*Math.cos(6.26*t+1.69);
    var b = .86-.54*t+.1*Math.cos(9.08*t-1.02);
    return this.tocol(r, g, b);
  },
  SouthwestColors: function(t) {
    var r = 2.15-26.75*t+51.14*Math.cos(.542*t-1.604);
    var g = .44+.3*t+.17*Math.cos(6.24*t+2.53);
    var b = .17+.38*t+.12*Math.cos(11.28*t+1.02);
    return this.tocol(r, g, b);
  },
  DeepSeaColors: function(t) {
    var r = -.45+1.77*t+.59*Math.cos(3.48*t+.17);
    var g = .39+.17*t+.4*Math.cos(2.95*t+2.86);
    var b = .22+.37*t+.47*Math.cos(1.93*t-1.41);
    return this.tocol(r, g, b);
  },
  StarryNightColors: function(t) {
    var r = .19+.74*t+.12*Math.cos(5.25*t+2.27);
    var g = .49-.07*t+.43*Math.cos(2.99*t-2.48);
    var b = .39-.02*t+.27*Math.cos(3.94*t-2.33);
    return this.tocol(r, g, b);
  },
  FallColors: function(t) {
    var r = .21+.81*t+.04*Math.cos(8.62*t-.26);
    var g = .26+.4*t+.12*Math.cos(6.28*t+.02);
    var b = 6.5-7.17*t+12.76*Math.cos(.571*t-2.069);
    return this.tocol(r, g, b);
  },
  SunsetColors: function(t) {
    var r = .14+1.15*t+.27*Math.cos(5.06*t-2.21);
    var g = -.05+1.07*t+.04*Math.cos(7.88*t+.1);
    var b = -.07+.73*t+.32*Math.cos(7.07*t-1.2);
    return this.tocol(r, g, b);
  },
  FruitPunchColors: function(t) {
    var r = .89-.02*t+.11*Math.cos(5.77*t-.21);
    var g = .57-.09*t+.14*Math.cos(5.77*t-2.09);
    var b = -.04+.68*t+.14*Math.cos(6.99*t+1.02);
    return this.tocol(r, g, b);
  },
  ThermometerColors: function(t) {
    var r = .45+.12*t+.39*Math.cos(4.18*t-2.51);
    var g = .28+.14*t+.55*Math.cos(4.18*t-1.92);
    var b = .46+.05*t+.48*Math.cos(3.22*t-.81);
    return this.tocol(r, g, b);
  },
  IslandColors: function(t) {
    var r = .67+.12*t+.14*Math.cos(7.54*t+1.45);
    var g = -8.59+8.*t+12.2*Math.cos(.712*t+.746);
    var b = -35.37+32.87*t+68.16*Math.cos(.508*t+1.022);
    return this.tocol(r, g, b);
  },
  WatermelonColors: function(t) {
    var r = 8.38-23.7*t+45.9*Math.cos(.542*t-1.752);
    var g = 6.22-56.24*t+127.22*Math.cos(.4551*t-1.6188);
    var b = 80.01-209.12*t+639.31*Math.cos(.3292*t-1.696);
    return this.tocol(r, g, b);
  },
  BrassTones: function(t) {
    var r = .16+.07*t+.73*Math.cos(3.31*t-1.67);
    var g = .24+0.*t+.6*Math.cos(3.49*t-1.8);
    var b = .12-.04*t+.31*Math.cos(3.55*t-1.89);
    return this.tocol(r, g, b);
  },
  GreenPinkTones: function(t) {
    var r = .53-.05*t+.55*Math.cos(5.5*t+2.78);
    var g = .21+.51*t+.62*Math.cos(4.82*t-1.55);
    var b = .6-.21*t+.57*Math.cos(5.27*t+2.86);
    return this.tocol(r, g, b);
  },
  BrownCyanTones: function(t) {
    var r = .2+.37*t+.47*Math.cos(3.37*t-1.29);
    var g = -1.14-14.82*t+26.53*Math.cos(.626*t-1.521);
    var b = .59-3.1*t+3.44*Math.cos(1.41*t-1.72);
    return this.tocol(r, g, b);
  },
  PigeonTones: function(t) {
    var r = .12+.86*t+.06*Math.cos(7.66*t-.48);
    var g = .13+.84*t+.04*Math.cos(7.9*t-.99);
    var b = .17+.79*t+.06*Math.cos(7.72*t-.96);
    return this.tocol(r, g, b);
  },
  CherryTones: function(t) {
    var r = -33.95+52.15*t+107.83*Math.cos(.4822*t+1.2484);
    var g = 11.9-9.82*t+19.47*Math.cos(.588*t-2.214);
    var b = 13.78-12.63*t+25.21*Math.cos(.562*t-2.139);
    return this.tocol(r, g, b);
  },
  RedBlueTones: function(t) {
    var r = .65-.25*t+.33*Math.cos(4.74*t-2.17);
    var g = .46+.01*t+.39*Math.cos(4.49*t-2.54);
    var b = .91-1.3*t+.96*Math.cos(2.62*t-2.37);
    return this.tocol(r, g, b);
  },
  CoffeeTones: function(t) {
    var r = -2.68+4.95*t+5.69*Math.cos(.8*t+1.);
    var g = .32+.64*t+.04*Math.cos(7.86*t-1.39);
    var b = .06+.86*t+.2*Math.cos(5.28*t-.16);
    return this.tocol(r, g, b);
  },
  RustTones: function(t) {
    var r = .12+1.02*t+.12*Math.cos(5.86*t-2.93);
    var g = .06+.47*t+.06*Math.cos(5.84*t-2.93);
    var b = .17-.16*t+.02*Math.cos(5.85*t+.2);
    return this.tocol(r, g, b);
  },
  FuchsiaTones: function(t) {
    var r = .39-3.14*t+5.33*Math.cos(.83*t-1.63);
    var g = .03+.91*t+.05*Math.cos(7.86*t-.06);
    var b = 1.44-4.92*t+8.11*Math.cos(.75*t-1.74);
    return this.tocol(r, g, b);
  },
  SiennaTones: function(t) {
    var r = -.41+1.68*t+.86*Math.cos(2.03*t-.04);
    var g = .33+.42*t+.15*Math.cos(3.42*t-2.91);
    var b = .4+.09*t+.34*Math.cos(3.09*t+2.8);
    return this.tocol(r, g, b);
  },
  GrayTones: function(t) {
    var r = .04+.84*t+.04*Math.cos(7.06*t-.08);
    var g = .06+.84*t+.03*Math.cos(7.43*t-.36);
    var b = .09+.79*t+.02*Math.cos(9.*t-1.35);
    return this.tocol(r, g, b);
  },
  ValentineTones: function(t) {
    var r = 2.39-3.68*t+5.19*Math.cos(.82*t-1.94);
    var g = .14+.6*t+.1*Math.cos(4.26*t+1.82);
    var b = .17+.66*t+.05*Math.cos(5.68*t+1.15);
    return this.tocol(r, g, b);
  },
  GrayYellowTones: function(t) {
    var r = .6-.13*t+.47*Math.cos(2.89*t-2.67);
    var g = 22.-61.93*t+141.55*Math.cos(.4457*t-1.7253);
    var b = 40.32-135.25*t+364.05*Math.cos(.3746*t-1.6809);
    return this.tocol(r, g, b);
  },
  DarkTerrain: function(t) {
    var r = -119.59+246.86*t+786.34*Math.cos(.3139*t+1.4182);
    var g = -5.32+12.06*t+7.19*Math.cos(1.75*t+.74);
    var b = -180.09+452.81*t+1665.28*Math.cos(.27247*t+1.4622);
    return this.tocol(r, g, b);
  },
  LightTerrain: function(t) {
    var r = .59+.26*t+.09*Math.cos(5.12*t+2.23);
    var g = 27.29-45.82*t+94.81*Math.cos(.49*t-1.854);
    var b = 39.02-51.94*t+114.12*Math.cos(.4623*t-1.9117);
    return this.tocol(r, g, b);
  },
  GreenBrownTerrain: function(t) {
    var r = .12+.95*t+.12*Math.cos(6.76*t-2.89);
    var g = .12+.88*t+.18*Math.cos(6.8*t-2.34);
    var b = -205.56+434.26*t+1593.1*Math.cos(.27279*t+1.44143);
    return this.tocol(r, g, b);
  },
  SandyTerrain: function(t) {
    var r = .9-.54*t+.32*Math.cos(4.28*t-2.37);
    var g = .48+.07*t+.27*Math.cos(4.7*t-2.32);
    var b = .26-.03*t+.06*Math.cos(5.68*t-2.62);
    return this.tocol(r, g, b);
  },
  BrightBands: function(t) {
    var r = .63+.13*t+.31*Math.cos(7.39*t-.03);
    var g = .52+.31*t+.29*Math.cos(5.46*t-3.02);
    var b = -577.52+1052.6*t+4334.34*Math.cos(.24336*t+1.43714);
    return this.tocol(r, g, b);
  },
  DarkBands: function(t) {
    var r = .52+.31*t+.15*Math.cos(11.59*t+.78);
    var g = .62+.21*t+.26*Math.cos(4.3*t+.76);
    var b = 1.08-.98*t+.35*Math.cos(5.11*t+1.76);
    return this.tocol(r, g, b);
  },
  Aquamarine: function(t) {
    var r = -.3+1.96*t+1.04*Math.cos(2.57*t+.36);
    var g = .26+.95*t+.47*Math.cos(2.85*t+.15);
    var b = -.2+2.36*t+1.51*Math.cos(1.8*t+.8);
    return this.tocol(r, g, b);
  },
  Pastel: function(t) {
    var r = 5.99-30.91*t+57.32*Math.cos(.549*t-1.662);
    var g = 1.69-3.63*t+2.89*Math.cos(1.56*t-1.99);
    var b = .81-.07*t+.22*Math.cos(5.58*t+.59);
    return this.tocol(r, g, b);
  },
  BlueGreenYellow: function(t) {
    var r = 2.08+4.68*t+6.45*Math.cos(.82*t+1.88);
    var g = -.47+1.41*t+.5*Math.cos(2.07*t-.42);
    var b = -1.06+1.97*t+1.61*Math.cos(1.48*t+.45);
    return this.tocol(r, g, b);
  },
  Rainbow: function(t) {
    var r = 132.23-245.97*t+755.63*Math.cos(.3275*t-1.7461);
    var g = .39-1.4*t+1.32*Math.cos(2.39*t-1.84);
    var b = -142.83+270.69*t+891.31*Math.cos(.3053*t+1.4092);
    return this.tocol(r, g, b);
  },
  DarkRainbow: function(t) {
    var r = .25+.64*t+.16*Math.cos(7.89*t+1.19);
    var g = .65-.34*t+.28*Math.cos(5.83*t+2.69);
    var b = .52-.4*t+.11*Math.cos(6.93*t+.6);
    return this.tocol(r, g, b);
  },
  TemperatureMap: function(t) {
    var r = .37+.71*t+.26*Math.cos(5.2*t-2.51);
    var g = .89-2.12*t+1.56*Math.cos(2.48*t-1.96);
    var b = 1.18-.94*t+.2*Math.cos(8.03*t+2.87);
    return this.tocol(r, g, b);
  },
  LightTemperatureMap: function(t) {
    var r = .38+.62*t+.24*Math.cos(4.9*t-2.61);
    var g = -5.49+.96*t+6.09*Math.cos(.97*t-.33);
    var b = 1.11-.73*t+.17*Math.cos(6.07*t-2.74);
    return this.tocol(r, g, b);
  },
};