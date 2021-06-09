const ColorFunctions = {
  clp: function(x) {
    return Math.round(255.*(x<0.?0.:x>1.?1.:x));
  },
  tocol: function(r, g, b) {
    return 'rgb('+this.clp(r)+','+this.clp(g)+','+this.clp(b)+')';
  },
  AlpineColors: function(t) {
    var r = .34+.562*t+.15*Math.cos(3.468*t+1.969);
    var g = .257+.75*t+.093*Math.cos(4.562*t+.16);
    var b = .431+.302*t+.173*Math.cos(4.864*t+1.108);
    return this.tocol(r, g, b);
  },
  LakeColors: function(t) {
    var r = .254+.731*t+.058*Math.cos(3.275*t-1.005);
    var g = .132+.669*t+.25*Math.cos(3.073*t-1.904);
    var b = .338+.725*t+.297*Math.cos(3.358*t-.976);
    return this.tocol(r, g, b);
  },
  ArmyColors: function(t) {
    var r = .43+.296*t+.017*Math.cos(12.935*t+.408);
    var g = .427+.427*t+.144*Math.cos(4.132*t+.063);
    var b = .886+2.688*t+3.539*Math.cos(.851*t+1.682);
    return this.tocol(r, g, b);
  },
  MintColors: function(t) {
    var r = -.968+.087*t+1.813*Math.cos(.817*t-.666);
    var g = .699-.492*t+.447*Math.cos(1.321*t-.899);
    var b = .071+.083*t+.736*Math.cos(1.242*t-.702);
    return this.tocol(r, g, b);
  },
  AtlanticColors: function(t) {
    var r = -.359-4.384*t+7.369*Math.cos(.729*t-1.507);
    var g = .309-.072*t+.477*Math.cos(2.921*t-1.956);
    var b = .464-.507*t+.749*Math.cos(2.484*t-2.026);
    return this.tocol(r, g, b);
  },
  NeonColors: function(t) {
    var r = .772+.066*t+.044*Math.cos(5.935*t+2.729);
    var g = .832-.792*t+.139*Math.cos(5.937*t+.756);
    var b = .142+.513*t+.133*Math.cos(5.692*t+.064);
    return this.tocol(r, g, b);
  },
  AuroraColors: function(t) {
    var r = .209+.586*t+.05*Math.cos(10.516*t+1.428);
    var g = .511-.234*t+.282*Math.cos(5.461*t+2.466);
    var b = .167+.704*t+.122*Math.cos(7.581*t-.642);
    return this.tocol(r, g, b);
  },
  PearlColors: function(t) {
    var r = .721+.133*t+.197*Math.cos(5.404*t-.179);
    var g = .784-.07*t+.112*Math.cos(6.567*t-.891);
    var b = .713+.151*t+.11*Math.cos(7.189*t-.772);
    return this.tocol(r, g, b);
  },
  AvocadoColors: function(t) {
    var r = .655-.2*t+.649*Math.cos(2.755*t+2.923);
    var g = -.678+2.051*t+.657*Math.cos(2.279*t-.098);
    var b = -.013+.246*t+.022*Math.cos(5.897*t-1.248);
    return this.tocol(r, g, b);
  },
  PlumColors: function(t) {
    var r = -1.019+2.938*t+1.008*Math.cos(2.766*t+.148);
    var g = -2.843+17.04*t+30.478*Math.cos(.544*t+1.478);
    var b = 2.035-4.714*t+3.865*Math.cos(1.482*t-2.124);
    return this.tocol(r, g, b);
  },
  BeachColors: function(t) {
    var r = -40.164+87.818*t+202.129*Math.cos(.4371*t+1.3668);
    var g = .559+.364*t+.081*Math.cos(7.608*t-2.555);
    var b = .149+.801*t+.105*Math.cos(6.822*t+.376);
    return this.tocol(r, g, b);
  },
  RoseColors: function(t) {
    var r = .324+.541*t+.17*Math.cos(5.629*t-2.919);
    var g = .482-.211*t+.17*Math.cos(5.382*t-2.917);
    var b = .221+.002*t+.159*Math.cos(4.866*t-2.524);
    return this.tocol(r, g, b);
  },
  CandyColors: function(t) {
    var r = .43+.468*t+.213*Math.cos(4.928*t-1.749);
    var g = 19.473-36.595*t+79.305*Math.cos(.474*t-1.816);
    var b = .483+.298*t+.165*Math.cos(4.659*t+2.531);
    return this.tocol(r, g, b);
  },
  SolarColors: function(t) {
    var r = -.325+1.625*t+.778*Math.cos(2.093*t-.134);
    var g = .027+.73*t+.077*Math.cos(5.134*t+1.925);
    var b = .172-.13*t+.155*Math.cos(2.134*t+3.136);
    return this.tocol(r, g, b);
  },
  CMYKColors: function(t) {
    var r = 21.344-109.661*t+274.797*Math.cos(.4044*t-1.6474);
    var g = .963-.695*t+.363*Math.cos(6.257*t+1.694);
    var b = .857-.536*t+.098*Math.cos(9.08*t-1.018);
    return this.tocol(r, g, b);
  },
  SouthwestColors: function(t) {
    var r = 2.153-26.755*t+51.145*Math.cos(.542*t-1.604);
    var g = .441+.304*t+.174*Math.cos(6.236*t+2.533);
    var b = .167+.385*t+.124*Math.cos(11.285*t+1.021);
    return this.tocol(r, g, b);
  },
  DeepSeaColors: function(t) {
    var r = -.452+1.772*t+.586*Math.cos(3.484*t+.168);
    var g = .395+.166*t+.399*Math.cos(2.951*t+2.859);
    var b = .216+.37*t+.472*Math.cos(1.93*t-1.406);
    return this.tocol(r, g, b);
  },
  StarryNightColors: function(t) {
    var r = .188+.741*t+.121*Math.cos(5.253*t+2.268);
    var g = .494-.068*t+.427*Math.cos(2.988*t-2.477);
    var b = .391-.019*t+.267*Math.cos(3.939*t-2.331);
    return this.tocol(r, g, b);
  },
  FallColors: function(t) {
    var r = .207+.806*t+.041*Math.cos(8.621*t-.258);
    var g = .262+.403*t+.124*Math.cos(6.28*t+.023);
    var b = 6.498-7.166*t+12.763*Math.cos(.571*t-2.069);
    return this.tocol(r, g, b);
  },
  SunsetColors: function(t) {
    var r = .138+1.15*t+.267*Math.cos(5.058*t-2.215);
    var g = -.054+1.068*t+.043*Math.cos(7.883*t+.103);
    var b = -.065+.734*t+.322*Math.cos(7.069*t-1.202);
    return this.tocol(r, g, b);
  },
  FruitPunchColors: function(t) {
    var r = .895-.02*t+.109*Math.cos(5.773*t-.213);
    var g = .569-.095*t+.137*Math.cos(5.769*t-2.086);
    var b = -.038+.685*t+.136*Math.cos(6.989*t+1.023);
    return this.tocol(r, g, b);
  },
  ThermometerColors: function(t) {
    var r = .453+.122*t+.385*Math.cos(4.177*t-2.507);
    var g = .284+.142*t+.554*Math.cos(4.181*t-1.918);
    var b = .464+.05*t+.475*Math.cos(3.217*t-.809);
    return this.tocol(r, g, b);
  },
  IslandColors: function(t) {
    var r = .67+.125*t+.142*Math.cos(7.543*t+1.454);
    var g = -8.587+7.996*t+12.196*Math.cos(.712*t+.746);
    var b = -35.367+32.875*t+68.157*Math.cos(.508*t+1.022);
    return this.tocol(r, g, b);
  },
  WatermelonColors: function(t) {
    var r = 8.384-23.701*t+45.905*Math.cos(.542*t-1.752);
    var g = 6.221-56.236*t+127.225*Math.cos(.4551*t-1.6188);
    var b = 80.008-209.122*t+639.314*Math.cos(.3292*t-1.696);
    return this.tocol(r, g, b);
  },
  BrassTones: function(t) {
    var r = .16+.07*t+.727*Math.cos(3.313*t-1.671);
    var g = .238+.004*t+.595*Math.cos(3.492*t-1.798);
    var b = .118-.036*t+.312*Math.cos(3.548*t-1.885);
    return this.tocol(r, g, b);
  },
  GreenPinkTones: function(t) {
    var r = .529-.054*t+.55*Math.cos(5.498*t+2.779);
    var g = .21+.512*t+.622*Math.cos(4.817*t-1.552);
    var b = .602-.212*t+.569*Math.cos(5.266*t+2.861);
    return this.tocol(r, g, b);
  },
  BrownCyanTones: function(t) {
    var r = .203+.373*t+.472*Math.cos(3.367*t-1.293);
    var g = -1.135-14.824*t+26.533*Math.cos(.626*t-1.521);
    var b = .587-3.096*t+3.441*Math.cos(1.409*t-1.721);
    return this.tocol(r, g, b);
  },
  PigeonTones: function(t) {
    var r = .115+.864*t+.065*Math.cos(7.663*t-.48);
    var g = .134+.845*t+.042*Math.cos(7.904*t-.989);
    var b = .166+.795*t+.057*Math.cos(7.722*t-.96);
    return this.tocol(r, g, b);
  },
  CherryTones: function(t) {
    var r = -33.955+52.149*t+107.832*Math.cos(.4822*t+1.2484);
    var g = 11.897-9.82*t+19.465*Math.cos(.588*t-2.214);
    var b = 13.784-12.632*t+25.214*Math.cos(.562*t-2.139);
    return this.tocol(r, g, b);
  },
  RedBlueTones: function(t) {
    var r = .649-.248*t+.33*Math.cos(4.743*t-2.171);
    var g = .46+.013*t+.388*Math.cos(4.495*t-2.538);
    var b = .912-1.304*t+.961*Math.cos(2.624*t-2.37);
    return this.tocol(r, g, b);
  },
  CoffeeTones: function(t) {
    var r = -2.682+4.95*t+5.693*Math.cos(.799*t+.997);
    var g = .32+.635*t+.04*Math.cos(7.862*t-1.389);
    var b = .064+.864*t+.199*Math.cos(5.276*t-.159);
    return this.tocol(r, g, b);
  },
  RustTones: function(t) {
    var r = .122+1.015*t+.12*Math.cos(5.857*t-2.927);
    var g = .062+.473*t+.056*Math.cos(5.844*t-2.93);
    var b = .17-.157*t+.019*Math.cos(5.848*t+.205);
    return this.tocol(r, g, b);
  },
  FuchsiaTones: function(t) {
    var r = .387-3.144*t+5.33*Math.cos(.828*t-1.625);
    var g = .034+.905*t+.05*Math.cos(7.864*t-.057);
    var b = 1.444-4.925*t+8.111*Math.cos(.747*t-1.737);
    return this.tocol(r, g, b);
  },
  SiennaTones: function(t) {
    var r = -.412+1.682*t+.857*Math.cos(2.03*t-.038);
    var g = .327+.416*t+.153*Math.cos(3.422*t-2.907);
    var b = .397+.093*t+.337*Math.cos(3.09*t+2.801);
    return this.tocol(r, g, b);
  },
  GrayTones: function(t) {
    var r = .043+.841*t+.044*Math.cos(7.064*t-.075);
    var g = .061+.84*t+.03*Math.cos(7.431*t-.36);
    var b = .088+.793*t+.017*Math.cos(9.001*t-1.351);
    return this.tocol(r, g, b);
  },
  ValentineTones: function(t) {
    var r = 2.391-3.682*t+5.193*Math.cos(.817*t-1.937);
    var g = .135+.605*t+.1*Math.cos(4.255*t+1.819);
    var b = .173+.657*t+.053*Math.cos(5.679*t+1.147);
    return this.tocol(r, g, b);
  },
  GrayYellowTones: function(t) {
    var r = .6-.129*t+.468*Math.cos(2.889*t-2.671);
    var g = 22.003-61.926*t+141.553*Math.cos(.4457*t-1.7253);
    var b = 40.32-135.252*t+364.051*Math.cos(.3746*t-1.6809);
    return this.tocol(r, g, b);
  },
  DarkTerrain: function(t) {
    var r = -119.592+246.859*t+786.342*Math.cos(.3139*t+1.4182);
    var g = -5.317+12.06*t+7.188*Math.cos(1.751*t+.736);
    var b = -180.095+452.81*t+1665.276*Math.cos(.27247*t+1.4622);
    return this.tocol(r, g, b);
  },
  LightTerrain: function(t) {
    var r = .585+.263*t+.087*Math.cos(5.119*t+2.234);
    var g = 27.294-45.823*t+94.808*Math.cos(.49*t-1.854);
    var b = 39.021-51.942*t+114.118*Math.cos(.4623*t-1.9117);
    return this.tocol(r, g, b);
  },
  GreenBrownTerrain: function(t) {
    var r = .117+.953*t+.119*Math.cos(6.758*t-2.887);
    var g = .12+.881*t+.181*Math.cos(6.804*t-2.345);
    var b = -205.56+434.262*t+1593.101*Math.cos(.27279*t+1.44143);
    return this.tocol(r, g, b);
  },
  SandyTerrain: function(t) {
    var r = .903-.539*t+.319*Math.cos(4.28*t-2.369);
    var g = .481+.071*t+.271*Math.cos(4.704*t-2.322);
    var b = .264-.027*t+.058*Math.cos(5.68*t-2.617);
    return this.tocol(r, g, b);
  },
  BrightBands: function(t) {
    var r = .631+.13*t+.314*Math.cos(7.388*t-.03);
    var g = .52+.312*t+.288*Math.cos(5.462*t-3.023);
    var b = -577.524+1052.6*t+4334.341*Math.cos(.24336*t+1.43714);
    return this.tocol(r, g, b);
  },
  DarkBands: function(t) {
    var r = .516+.307*t+.147*Math.cos(11.59*t+.777);
    var g = .621+.214*t+.262*Math.cos(4.299*t+.764);
    var b = 1.077-.984*t+.355*Math.cos(5.115*t+1.765);
    return this.tocol(r, g, b);
  },
  Aquamarine: function(t) {
    var r = -.304+1.962*t+1.037*Math.cos(2.567*t+.358);
    var g = .262+.949*t+.472*Math.cos(2.849*t+.153);
    var b = -.204+2.356*t+1.507*Math.cos(1.8*t+.802);
    return this.tocol(r, g, b);
  },
  Pastel: function(t) {
    var r = 5.992-30.909*t+57.315*Math.cos(.549*t-1.662);
    var g = 1.695-3.634*t+2.889*Math.cos(1.558*t-1.995);
    var b = .813-.068*t+.215*Math.cos(5.582*t+.594);
    return this.tocol(r, g, b);
  },
  BlueGreenYellow: function(t) {
    var r = 2.083+4.676*t+6.451*Math.cos(.818*t+1.879);
    var g = -.467+1.408*t+.504*Math.cos(2.071*t-.424);
    var b = -1.062+1.975*t+1.607*Math.cos(1.481*t+.447);
    return this.tocol(r, g, b);
  },
  Rainbow: function(t) {
    var r = 132.228-245.968*t+755.627*Math.cos(.3275*t-1.7461);
    var g = .385-1.397*t+1.319*Math.cos(2.391*t-1.839);
    var b = -142.825+270.693*t+891.307*Math.cos(.3053*t+1.4092);
    return this.tocol(r, g, b);
  },
  DarkRainbow: function(t) {
    var r = .25+.638*t+.163*Math.cos(7.885*t+1.194);
    var g = .655-.343*t+.28*Math.cos(5.831*t+2.688);
    var b = .523-.4*t+.113*Math.cos(6.931*t+.596);
    return this.tocol(r, g, b);
  },
  TemperatureMap: function(t) {
    var r = .372+.707*t+.265*Math.cos(5.201*t-2.515);
    var g = .888-2.123*t+1.556*Math.cos(2.483*t-1.959);
    var b = 1.182-.943*t+.195*Math.cos(8.032*t+2.875);
    return this.tocol(r, g, b);
  },
  LightTemperatureMap: function(t) {
    var r = .385+.619*t+.238*Math.cos(4.903*t-2.61);
    var g = -5.491+.959*t+6.089*Math.cos(.968*t-.329);
    var b = 1.107-.734*t+.172*Math.cos(6.07*t-2.741);
    return this.tocol(r, g, b);
  },
};