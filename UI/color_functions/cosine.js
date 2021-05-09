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
    var r = .345+.527*t+.109*Math.cos(2.623*t+.641);
    var g = .427+.427*t+.144*Math.cos(4.132*t+.063);
    var b = .386+.174*t+.066*Math.cos(6.358*t-.264);
    return this.tocol(r, g, b);
  },
  MintColors: function(t) {
    var r = -.619+.176*t+1.394*Math.cos(.93*t-.688);
    var g = .426-.597*t+.798*Math.cos(.992*t-.81);
    var b = .264+.088*t+.541*Math.cos(1.457*t-.817);
    return this.tocol(r, g, b);
  },
  AtlanticColors: function(t) {
    var r = .118-.518*t+.909*Math.cos(1.609*t-1.577);
    var g = .309-.072*t+.477*Math.cos(2.921*t-1.956);
    var b = .464-.507*t+.749*Math.cos(2.484*t-2.026);
    return this.tocol(r, g, b);
  },
  NeonColors: function(t) {
    var r = .772+.066*t+.044*Math.cos(5.935*t-3.554);
    var g = .832-.792*t+.139*Math.cos(5.937*t+.756);
    var b = .142+.513*t+.133*Math.cos(5.692*t+.064);
    return this.tocol(r, g, b);
  },
  AuroraColors: function(t) {
    var r = .209+.586*t+.05*Math.cos(10.516*t-4.856);
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
    var g = -.129+.886*t+.084*Math.cos(6.614*t-.542);
    var b = 2.035-4.714*t+3.865*Math.cos(1.482*t+4.159);
    return this.tocol(r, g, b);
  },
  BeachColors: function(t) {
    var r = -.179+2.163*t+1.031*Math.cos(2.623*t+.345);
    var g = .559+.364*t+.081*Math.cos(7.607*t-2.555);
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
    var g = .106+.828*t+.031*Math.cos(11.046*t-.4);
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
    var r = .61+.194*t+.081*Math.cos(10.715*t+.114);
    var g = .963-.695*t+.363*Math.cos(6.258*t+1.694);
    var b = .857-.536*t+.098*Math.cos(9.08*t-1.018);
    return this.tocol(r, g, b);
  },
  SouthwestColors: function(t) {
    var r = .848-2.173*t+1.659*Math.cos(1.874*t-1.808);
    var g = .441+.304*t+.174*Math.cos(6.236*t+2.533);
    var b = -.269+1.247*t+.315*Math.cos(3.882*t-.317);
    return this.tocol(r, g, b);
  },
  DeepSeaColors: function(t) {
    var r = -.452+1.772*t+.586*Math.cos(3.484*t+.168);
    var g = .395+.166*t+.399*Math.cos(2.951*t+2.859);
    var b = .209+.358*t+.487*Math.cos(1.901*t-1.398);
    return this.tocol(r, g, b);
  },
  StarryNightColors: function(t) {
    var r = .188+.741*t+.121*Math.cos(5.253*t+2.268);
    var g = .494-.068*t+.427*Math.cos(2.988*t+3.806);
    var b = .391-.019*t+.267*Math.cos(3.939*t-2.331);
    return this.tocol(r, g, b);
  },
  FallColors: function(t) {
    var r = .207+.806*t+.041*Math.cos(8.621*t-.258);
    var g = .262+.403*t+.124*Math.cos(6.28*t+.023);
    var b = .307-.118*t+.041*Math.cos(7.41*t-.251);
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
    var b = -.038+.685*t+.136*Math.cos(6.99*t+1.023);
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
    var g = -.363+1.194*t+.825*Math.cos(2.085*t-.458);
    var b = -.248+.869*t+.821*Math.cos(2.872*t-.925);
    return this.tocol(r, g, b);
  },
  WatermelonColors: function(t) {
    var r = .171+.9*t+.038*Math.cos(13.945*t-3.805);
    var g = .749-2.337*t+1.967*Math.cos(2.013*t-1.894);
    var b = .161+.66*t+.085*Math.cos(10.552*t-.376);
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
    var g = .307-1.235*t+1.606*Math.cos(1.806*t-1.637);
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
    var r = -.507+2.128*t+.743*Math.cos(2.704*t-.219);
    var g = -.011+.83*t+.103*Math.cos(7.685*t-.517);
    var b = -.01+.837*t+.101*Math.cos(7.825*t-.558);
    return this.tocol(r, g, b);
  },
  RedBlueTones: function(t) {
    var r = .649-.248*t+.33*Math.cos(4.743*t-2.171);
    var g = .46+.013*t+.388*Math.cos(4.495*t-2.538);
    var b = .912-1.304*t+.961*Math.cos(2.624*t+3.913);
    return this.tocol(r, g, b);
  },
  CoffeeTones: function(t) {
    var r = -.275+1.604*t+.75*Math.cos(1.609*t+.425);
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
    var r = .271-.695*t+1.499*Math.cos(1.307*t-1.688);
    var g = .034+.905*t+.05*Math.cos(7.864*t-.057);
    var b = .615-1.141*t+1.82*Math.cos(1.25*t-1.857);
    return this.tocol(r, g, b);
  },
  SiennaTones: function(t) {
    var r = -.412+1.682*t+.857*Math.cos(2.03*t-.038);
    var g = .327+.416*t+.153*Math.cos(3.422*t+3.376);
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
    var r = .515+.491*t+.012*Math.cos(9.911*t-.531);
    var g = .135+.605*t+.1*Math.cos(4.255*t+1.819);
    var b = .173+.657*t+.053*Math.cos(5.679*t+1.147);
    return this.tocol(r, g, b);
  },
  GrayYellowTones: function(t) {
    var r = .6-.129*t+.468*Math.cos(2.889*t+3.612);
    var g = .257+.751*t+.049*Math.cos(10.88*t-.313);
    var b = .458+.293*t+.082*Math.cos(10.715*t-.059);
    return this.tocol(r, g, b);
  },
  DarkTerrain: function(t) {
    var r = -.411+1.77*t+.425*Math.cos(4.184*t-.465);
    var g = -5.313+12.05*t+7.18*Math.cos(1.752*t+.736);
    var b = -.034+1.018*t+.43*Math.cos(5.044*t-.453);
    return this.tocol(r, g, b);
  },
  LightTerrain: function(t) {
    var r = .585+.263*t+.087*Math.cos(5.119*t+2.234);
    var g = .608+.282*t+.058*Math.cos(8.688*t-.184);
    var b = .55+.174*t+.129*Math.cos(7.825*t-.327);
    return this.tocol(r, g, b);
  },
  GreenBrownTerrain: function(t) {
    var r = .117+.953*t+.119*Math.cos(6.758*t-2.887);
    var g = .12+.881*t+.181*Math.cos(6.804*t-2.345);
    var b = -.34+1.52*t+.402*Math.cos(4.822*t-.711);
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
    var b = -.129+1.322*t+.767*Math.cos(4.932*t-1.138);
    return this.tocol(r, g, b);
  },
  DarkBands: function(t) {
    var r = .516+.307*t+.147*Math.cos(11.59*t-5.506);
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
    var r = .94-.191*t+.054*Math.cos(10.715*t+.139);
    var g = 1.695-3.634*t+2.889*Math.cos(1.558*t+4.289);
    var b = .813-.068*t+.215*Math.cos(5.582*t+.594);
    return this.tocol(r, g, b);
  },
  BlueGreenYellow: function(t) {
    var r = 2.081+4.672*t+6.442*Math.cos(.818*t+1.879);
    var g = -.467+1.408*t+.504*Math.cos(2.071*t-.424);
    var b = -1.062+1.975*t+1.607*Math.cos(1.481*t+.447);
    return this.tocol(r, g, b);
  },
  Rainbow: function(t) {
    var r = .132+.851*t+.109*Math.cos(9.597*t-.378);
    var g = .385-1.397*t+1.319*Math.cos(2.391*t-1.839);
    var b = .116+.645*t+.54*Math.cos(3.882*t-.475);
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
    var g = .888-2.123*t+1.556*Math.cos(2.483*t+4.324);
    var b = 1.182-.943*t+.195*Math.cos(8.032*t-3.409);
    return this.tocol(r, g, b);
  },
  LightTemperatureMap: function(t) {
    var r = .385+.619*t+.238*Math.cos(4.903*t-2.61);
    var g = -.108+.2*t+1.021*Math.cos(2.463*t-1.172);
    var b = 1.107-.734*t+.172*Math.cos(6.07*t-2.741);
    return this.tocol(r, g, b);
  },
};