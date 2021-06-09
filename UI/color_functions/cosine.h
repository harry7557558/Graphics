template<typename vec3, typename Float>
class ColorFunctions {
  static Float clp(Float x) {
    return (Float)(x<0.?0.:x>1.?1.:x);
  }
public:
  static vec3 AlpineColors(Float t) {
    Float r = .34+.562*t+.15*cos(3.468*t+1.969);
    Float g = .257+.75*t+.093*cos(4.562*t+.16);
    Float b = .431+.302*t+.173*cos(4.864*t+1.108);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LakeColors(Float t) {
    Float r = .254+.731*t+.058*cos(3.275*t-1.005);
    Float g = .132+.669*t+.25*cos(3.073*t-1.904);
    Float b = .338+.725*t+.297*cos(3.358*t-.976);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ArmyColors(Float t) {
    Float r = .43+.296*t+.017*cos(12.935*t+.408);
    Float g = .427+.427*t+.144*cos(4.132*t+.063);
    Float b = .886+2.688*t+3.539*cos(.851*t+1.682);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 MintColors(Float t) {
    Float r = -.968+.087*t+1.813*cos(.817*t-.666);
    Float g = .699-.492*t+.447*cos(1.321*t-.899);
    Float b = .071+.083*t+.736*cos(1.242*t-.702);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AtlanticColors(Float t) {
    Float r = -.359-4.384*t+7.369*cos(.729*t-1.507);
    Float g = .309-.072*t+.477*cos(2.921*t-1.956);
    Float b = .464-.507*t+.749*cos(2.484*t-2.026);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 NeonColors(Float t) {
    Float r = .772+.066*t+.044*cos(5.935*t+2.729);
    Float g = .832-.792*t+.139*cos(5.937*t+.756);
    Float b = .142+.513*t+.133*cos(5.692*t+.064);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AuroraColors(Float t) {
    Float r = .209+.586*t+.05*cos(10.516*t+1.428);
    Float g = .511-.234*t+.282*cos(5.461*t+2.466);
    Float b = .167+.704*t+.122*cos(7.581*t-.642);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PearlColors(Float t) {
    Float r = .721+.133*t+.197*cos(5.404*t-.179);
    Float g = .784-.07*t+.112*cos(6.567*t-.891);
    Float b = .713+.151*t+.11*cos(7.189*t-.772);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AvocadoColors(Float t) {
    Float r = .655-.2*t+.649*cos(2.755*t+2.923);
    Float g = -.678+2.051*t+.657*cos(2.279*t-.098);
    Float b = -.013+.246*t+.022*cos(5.897*t-1.248);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PlumColors(Float t) {
    Float r = -1.019+2.938*t+1.008*cos(2.766*t+.148);
    Float g = -2.843+17.04*t+30.478*cos(.544*t+1.478);
    Float b = 2.035-4.714*t+3.865*cos(1.482*t-2.124);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BeachColors(Float t) {
    Float r = -40.164+87.818*t+202.129*cos(.4371*t+1.3668);
    Float g = .559+.364*t+.081*cos(7.608*t-2.555);
    Float b = .149+.801*t+.105*cos(6.822*t+.376);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RoseColors(Float t) {
    Float r = .324+.541*t+.17*cos(5.629*t-2.919);
    Float g = .482-.211*t+.17*cos(5.382*t-2.917);
    Float b = .221+.002*t+.159*cos(4.866*t-2.524);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CandyColors(Float t) {
    Float r = .43+.468*t+.213*cos(4.928*t-1.749);
    Float g = 19.473-36.595*t+79.305*cos(.474*t-1.816);
    Float b = .483+.298*t+.165*cos(4.659*t+2.531);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SolarColors(Float t) {
    Float r = -.325+1.625*t+.778*cos(2.093*t-.134);
    Float g = .027+.73*t+.077*cos(5.134*t+1.925);
    Float b = .172-.13*t+.155*cos(2.134*t+3.136);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CMYKColors(Float t) {
    Float r = 21.344-109.661*t+274.797*cos(.4044*t-1.6474);
    Float g = .963-.695*t+.363*cos(6.257*t+1.694);
    Float b = .857-.536*t+.098*cos(9.08*t-1.018);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SouthwestColors(Float t) {
    Float r = 2.153-26.755*t+51.145*cos(.542*t-1.604);
    Float g = .441+.304*t+.174*cos(6.236*t+2.533);
    Float b = .167+.385*t+.124*cos(11.285*t+1.021);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DeepSeaColors(Float t) {
    Float r = -.452+1.772*t+.586*cos(3.484*t+.168);
    Float g = .395+.166*t+.399*cos(2.951*t+2.859);
    Float b = .216+.37*t+.472*cos(1.93*t-1.406);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 StarryNightColors(Float t) {
    Float r = .188+.741*t+.121*cos(5.253*t+2.268);
    Float g = .494-.068*t+.427*cos(2.988*t-2.477);
    Float b = .391-.019*t+.267*cos(3.939*t-2.331);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FallColors(Float t) {
    Float r = .207+.806*t+.041*cos(8.621*t-.258);
    Float g = .262+.403*t+.124*cos(6.28*t+.023);
    Float b = 6.498-7.166*t+12.763*cos(.571*t-2.069);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SunsetColors(Float t) {
    Float r = .138+1.15*t+.267*cos(5.058*t-2.215);
    Float g = -.054+1.068*t+.043*cos(7.883*t+.103);
    Float b = -.065+.734*t+.322*cos(7.069*t-1.202);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FruitPunchColors(Float t) {
    Float r = .895-.02*t+.109*cos(5.773*t-.213);
    Float g = .569-.095*t+.137*cos(5.769*t-2.086);
    Float b = -.038+.685*t+.136*cos(6.989*t+1.023);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ThermometerColors(Float t) {
    Float r = .453+.122*t+.385*cos(4.177*t-2.507);
    Float g = .284+.142*t+.554*cos(4.181*t-1.918);
    Float b = .464+.05*t+.475*cos(3.217*t-.809);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 IslandColors(Float t) {
    Float r = .67+.125*t+.142*cos(7.543*t+1.454);
    Float g = -8.587+7.996*t+12.196*cos(.712*t+.746);
    Float b = -35.367+32.875*t+68.157*cos(.508*t+1.022);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 WatermelonColors(Float t) {
    Float r = 8.384-23.701*t+45.905*cos(.542*t-1.752);
    Float g = 6.221-56.236*t+127.225*cos(.4551*t-1.6188);
    Float b = 80.008-209.122*t+639.314*cos(.3292*t-1.696);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrassTones(Float t) {
    Float r = .16+.07*t+.727*cos(3.313*t-1.671);
    Float g = .238+.004*t+.595*cos(3.492*t-1.798);
    Float b = .118-.036*t+.312*cos(3.548*t-1.885);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GreenPinkTones(Float t) {
    Float r = .529-.054*t+.55*cos(5.498*t+2.779);
    Float g = .21+.512*t+.622*cos(4.817*t-1.552);
    Float b = .602-.212*t+.569*cos(5.266*t+2.861);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrownCyanTones(Float t) {
    Float r = .203+.373*t+.472*cos(3.367*t-1.293);
    Float g = -1.135-14.824*t+26.533*cos(.626*t-1.521);
    Float b = .587-3.096*t+3.441*cos(1.409*t-1.721);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PigeonTones(Float t) {
    Float r = .115+.864*t+.065*cos(7.663*t-.48);
    Float g = .134+.845*t+.042*cos(7.904*t-.989);
    Float b = .166+.795*t+.057*cos(7.722*t-.96);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CherryTones(Float t) {
    Float r = -33.955+52.149*t+107.832*cos(.4822*t+1.2484);
    Float g = 11.897-9.82*t+19.465*cos(.588*t-2.214);
    Float b = 13.784-12.632*t+25.214*cos(.562*t-2.139);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RedBlueTones(Float t) {
    Float r = .649-.248*t+.33*cos(4.743*t-2.171);
    Float g = .46+.013*t+.388*cos(4.495*t-2.538);
    Float b = .912-1.304*t+.961*cos(2.624*t-2.37);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CoffeeTones(Float t) {
    Float r = -2.682+4.95*t+5.693*cos(.799*t+.997);
    Float g = .32+.635*t+.04*cos(7.862*t-1.389);
    Float b = .064+.864*t+.199*cos(5.276*t-.159);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RustTones(Float t) {
    Float r = .122+1.015*t+.12*cos(5.857*t-2.927);
    Float g = .062+.473*t+.056*cos(5.844*t-2.93);
    Float b = .17-.157*t+.019*cos(5.848*t+.205);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FuchsiaTones(Float t) {
    Float r = .387-3.144*t+5.33*cos(.828*t-1.625);
    Float g = .034+.905*t+.05*cos(7.864*t-.057);
    Float b = 1.444-4.925*t+8.111*cos(.747*t-1.737);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SiennaTones(Float t) {
    Float r = -.412+1.682*t+.857*cos(2.03*t-.038);
    Float g = .327+.416*t+.153*cos(3.422*t-2.907);
    Float b = .397+.093*t+.337*cos(3.09*t+2.801);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GrayTones(Float t) {
    Float r = .043+.841*t+.044*cos(7.064*t-.075);
    Float g = .061+.84*t+.03*cos(7.431*t-.36);
    Float b = .088+.793*t+.017*cos(9.001*t-1.351);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ValentineTones(Float t) {
    Float r = 2.391-3.682*t+5.193*cos(.817*t-1.937);
    Float g = .135+.605*t+.1*cos(4.255*t+1.819);
    Float b = .173+.657*t+.053*cos(5.679*t+1.147);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GrayYellowTones(Float t) {
    Float r = .6-.129*t+.468*cos(2.889*t-2.671);
    Float g = 22.003-61.926*t+141.553*cos(.4457*t-1.7253);
    Float b = 40.32-135.252*t+364.051*cos(.3746*t-1.6809);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkTerrain(Float t) {
    Float r = -119.592+246.859*t+786.342*cos(.3139*t+1.4182);
    Float g = -5.317+12.06*t+7.188*cos(1.751*t+.736);
    Float b = -180.095+452.81*t+1665.276*cos(.27247*t+1.4622);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LightTerrain(Float t) {
    Float r = .585+.263*t+.087*cos(5.119*t+2.234);
    Float g = 27.294-45.823*t+94.808*cos(.49*t-1.854);
    Float b = 39.021-51.942*t+114.118*cos(.4623*t-1.9117);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GreenBrownTerrain(Float t) {
    Float r = .117+.953*t+.119*cos(6.758*t-2.887);
    Float g = .12+.881*t+.181*cos(6.804*t-2.345);
    Float b = -205.56+434.262*t+1593.101*cos(.27279*t+1.44143);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SandyTerrain(Float t) {
    Float r = .903-.539*t+.319*cos(4.28*t-2.369);
    Float g = .481+.071*t+.271*cos(4.704*t-2.322);
    Float b = .264-.027*t+.058*cos(5.68*t-2.617);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrightBands(Float t) {
    Float r = .631+.13*t+.314*cos(7.388*t-.03);
    Float g = .52+.312*t+.288*cos(5.462*t-3.023);
    Float b = -577.524+1052.6*t+4334.341*cos(.24336*t+1.43714);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkBands(Float t) {
    Float r = .516+.307*t+.147*cos(11.59*t+.777);
    Float g = .621+.214*t+.262*cos(4.299*t+.764);
    Float b = 1.077-.984*t+.355*cos(5.115*t+1.765);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Aquamarine(Float t) {
    Float r = -.304+1.962*t+1.037*cos(2.567*t+.358);
    Float g = .262+.949*t+.472*cos(2.849*t+.153);
    Float b = -.204+2.356*t+1.507*cos(1.8*t+.802);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Pastel(Float t) {
    Float r = 5.992-30.909*t+57.315*cos(.549*t-1.662);
    Float g = 1.695-3.634*t+2.889*cos(1.558*t-1.995);
    Float b = .813-.068*t+.215*cos(5.582*t+.594);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BlueGreenYellow(Float t) {
    Float r = 2.083+4.676*t+6.451*cos(.818*t+1.879);
    Float g = -.467+1.408*t+.504*cos(2.071*t-.424);
    Float b = -1.062+1.975*t+1.607*cos(1.481*t+.447);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Rainbow(Float t) {
    Float r = 132.228-245.968*t+755.627*cos(.3275*t-1.7461);
    Float g = .385-1.397*t+1.319*cos(2.391*t-1.839);
    Float b = -142.825+270.693*t+891.307*cos(.3053*t+1.4092);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkRainbow(Float t) {
    Float r = .25+.638*t+.163*cos(7.885*t+1.194);
    Float g = .655-.343*t+.28*cos(5.831*t+2.688);
    Float b = .523-.4*t+.113*cos(6.931*t+.596);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 TemperatureMap(Float t) {
    Float r = .372+.707*t+.265*cos(5.201*t-2.515);
    Float g = .888-2.123*t+1.556*cos(2.483*t-1.959);
    Float b = 1.182-.943*t+.195*cos(8.032*t+2.875);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LightTemperatureMap(Float t) {
    Float r = .385+.619*t+.238*cos(4.903*t-2.61);
    Float g = -5.491+.959*t+6.089*cos(.968*t-.329);
    Float b = 1.107-.734*t+.172*cos(6.07*t-2.741);
    return vec3(clp(r),clp(g),clp(b));
  }
};
