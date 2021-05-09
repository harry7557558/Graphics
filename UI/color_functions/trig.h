template<typename vec3, typename Float>
class ColorFunctions {
  static Float clp(Float x) {
    return (Float)(x<0.?0.:x>1.?1.:x);
  }
public:
  static vec3 AlpineColors(Float x) {
    Float r = .388+.512*x+.181*cos(3.142*x+2.181);
    Float g = .199+.976*x+.199*cos(3.142*x+.662);
    Float b = -.512+2.345*x+.963*cos(3.142*x+.199)+.226*cos(6.283*x+1.363);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LakeColors(Float x) {
    Float r = .247+.739*x+.063*cos(3.142*x-.917);
    Float g = -.182+1.301*x+.32*cos(3.142*x-.825)+.058*cos(6.283*x+1.498);
    Float b = .468+.592*x+.179*cos(3.142*x-.667)+.06*cos(6.283*x-2.996)+.021*cos(12.566*x+2.472);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ArmyColors(Float x) {
    Float r = .947-.552*x+.451*cos(3.142*x+2.811)+.108*cos(6.283*x-2.077);
    Float g = 1.304-1.217*x+.7*cos(3.142*x+2.945)+.181*cos(6.283*x-1.622);
    Float b = 1.051-.861*x+.551*cos(3.142*x+2.71)+.115*cos(6.283*x-1.862);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 MintColors(Float x) {
    Float r = .474+.436*x+.14*cos(3.142*x-1.635);
    Float g = .99-.38*x+.089*cos(3.142*x-1.67);
    Float b = .644+.141*x+.133*cos(3.142*x-1.611);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AtlanticColors(Float x) {
    Float r = 1.607-2.72*x+1.547*cos(3.142*x-2.941)+.28*cos(6.283*x-1.388);
    Float g = 1.575-2.507*x+1.453*cos(3.142*x-2.883)+.236*cos(6.283*x-1.572);
    Float b = 1.633-2.506*x+1.51*cos(3.142*x-2.895)+.23*cos(6.283*x-1.544);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 NeonColors(Float x) {
    Float r = .83-.148*x+.135*cos(3.142*x-2.509);
    Float g = 1.636-2.115*x+.729*cos(3.142*x+2.815)+.088*cos(6.283*x-1.428)+.036*cos(12.566*x-2.08);
    Float b = .532-.185*x+.339*cos(3.142*x+2.888)+.123*cos(6.283*x-.801);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AuroraColors(Float x) {
    Float r = 1.176-.831*x+.83*cos(3.142*x+2.616)+.257*cos(6.283*x-2.45);
    Float g = .036+.564*x+.317*cos(3.142*x-.493)+.248*cos(6.283*x+1.791);
    Float b = .017+.711*x+.26*cos(3.142*x-1.555)+.235*cos(6.283*x+.008);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PearlColors(Float x) {
    Float r = .111+1.453*x+.701*cos(3.142*x+.168)+.105*cos(6.283*x+.473);
    Float g = .159+1.111*x+.562*cos(3.142*x-.105)+.111*cos(6.283*x+.216);
    Float b = .069+1.266*x+.543*cos(3.142*x-.288)+.182*cos(6.283*x+.31);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 AvocadoColors(Float x) {
    Float r = -1.482+3.952*x+1.476*cos(3.142*x+.133)+.38*cos(6.283*x+1.559)+.028*cos(12.566*x+1.368);
    Float g = -.251+1.483*x+.299*cos(3.142*x-.687);
    Float b = -.106+.439*x+.098*cos(3.142*x+.065);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PlumColors(Float x) {
    Float r = -.729+2.374*x+.71*cos(3.142*x-.046);
    Float g = -.192+1.267*x+.263*cos(3.142*x+.825);
    Float b = .453-.47*x+.48*cos(3.142*x-2.748);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BeachColors(Float x) {
    Float r = .325+1.524*x+.779*cos(3.142*x+.527)+.147*cos(6.283*x+3.076);
    Float g = .03+1.694*x+.623*cos(3.142*x+.36)+.116*cos(6.283*x+3.087);
    Float b = .535-.062*x+.406*cos(3.142*x-2.94)+.121*cos(6.283*x-.024);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RoseColors(Float x) {
    Float r = .299+.486*x+.117*cos(3.142*x-1.852)+.109*cos(6.283*x+3.069);
    Float g = .503-.377*x+.171*cos(3.142*x-2.184)+.089*cos(6.283*x+3.038);
    Float b = .165-.021*x+.174*cos(3.142*x-1.691)+.054*cos(6.283*x+3.058);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CandyColors(Float x) {
    Float r = .774-.379*x+.371*cos(3.142*x-2.605)+.154*cos(6.283*x-1.902);
    Float g = .605-.004*x+.36*cos(3.142*x+2.765)+.052*cos(6.283*x-3.058);
    Float b = .8-.375*x+.452*cos(3.142*x-3.047);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SolarColors(Float x) {
    Float r = .257+.951*x+.306*cos(3.142*x-.869);
    Float g = .266+.288*x+.26*cos(3.142*x+3.012);
    Float b = .062+.016*x+.06*cos(3.142*x+2.445);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CMYKColors(Float x) {
    Float r = 1.548-3.031*x+1.765*cos(3.142*x-2.507)+.221*cos(6.283*x-.724);
    Float g = -1.734+3.258*x+2.223*cos(3.142*x-.534)+.851*cos(6.283*x+1.026)+.05*cos(12.566*x+.109);
    Float b = 1.389-2.952*x+1.531*cos(3.142*x-2.345)+.546*cos(6.283*x-.206)+.056*cos(12.566*x+.462);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SouthwestColors(Float x) {
    Float r = -1.089+2.726*x+1.514*cos(3.142*x-.434)+.337*cos(6.283*x+1.286);
    Float g = .992-.784*x+.529*cos(3.142*x+3.123)+.144*cos(6.283*x+3.137);
    Float b = -.355+2.292*x+1.067*cos(3.142*x+.711)+.312*cos(6.283*x+2.973)+.046*cos(12.566*x+.005)+.016*cos(18.85*x+.014)+.007*cos(25.133*x+2.161);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DeepSeaColors(Float x) {
    Float r = -.865+2.54*x+.968*cos(3.142*x+.133)+.081*cos(6.283*x+1.016)+.025*cos(12.566*x+1.009)+.021*cos(18.85*x+1.301);
    Float g = .323+.28*x+.339*cos(3.142*x+2.74);
    Float b = .35+.599*x+.187*cos(3.142*x-1.847);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 StarryNightColors(Float x) {
    Float r = -.092+1.192*x+.18*cos(3.142*x-.53)+.115*cos(6.283*x+1.394);
    Float g = .469+.021*x+.379*cos(3.142*x-2.529);
    Float b = .368-.19*x+.393*cos(3.142*x-2.022);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FallColors(Float x) {
    Float r = .404+.222*x+.286*cos(3.142*x-2.557)+.088*cos(6.283*x+.067);
    Float g = .272+.386*x+.009*cos(3.142*x+2.949)+.123*cos(6.283*x+.009);
    Float b = .477-.312*x+.136*cos(3.142*x+2.254);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SunsetColors(Float x) {
    Float r = 1.002-.806*x+.937*cos(3.142*x-2.848)+.246*cos(6.283*x-1.979);
    Float g = -.166+1.201*x+.124*cos(3.142*x-.699)+.089*cos(6.283*x+.827);
    Float b = 7.061-13.028*x+7.02*cos(3.142*x+3.092)+1.788*cos(6.283*x-1.526)+.194*cos(12.566*x-1.916)+.049*cos(18.85*x-2.292)+.015*cos(25.133*x-2.897);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FruitPunchColors(Float x) {
    Float r = .666+.482*x+.261*cos(3.142*x+.185)+.072*cos(6.283*x+.1);
    Float g = .087+.841*x+.481*cos(3.142*x-.074)+.076*cos(6.283*x+3.028);
    Float b = .743-1.307*x+.97*cos(3.142*x-2.782)+.17*cos(6.283*x-.077);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ThermometerColors(Float x) {
    Float r = 1.011-1.233*x+.9*cos(3.142*x-2.659)+.105*cos(6.283*x-2.117);
    Float g = .121+.077*x+.664*cos(3.142*x-1.49)+.084*cos(6.283*x-2.796);
    Float b = .428+.096*x+.501*cos(3.142*x-.759);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 IslandColors(Float x) {
    Float r = 2.031-2.229*x+1.09*cos(3.142*x+2.856)+.201*cos(6.283*x-2.984);
    Float g = -.421+1.871*x+.847*cos(3.142*x-.565)+.123*cos(6.283*x+1.123);
    Float b = -1.173+2.65*x+1.509*cos(3.142*x-.584)+.207*cos(6.283*x+1.087);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 WatermelonColors(Float x) {
    Float r = .245+.256*x+.467*cos(3.142*x-2.172)+.108*cos(6.283*x+.008);
    Float g = .418-.663*x+.956*cos(3.142*x-2.066)+.123*cos(6.283*x-.028);
    Float b = .814-1.542*x+1.128*cos(3.142*x-2.473)+.167*cos(6.283*x-.048);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrassTones(Float x) {
    Float r = .978-1.276*x+.796*cos(3.142*x-2.527)+.183*cos(6.283*x-2.455)+.043*cos(12.566*x+2.614)+.018*cos(18.85*x-.696)+.017*cos(25.133*x-3.093);
    Float g = .83-1.005*x+.637*cos(3.142*x-2.484)+.156*cos(6.283*x-2.603)+.04*cos(12.566*x+2.521)+.017*cos(18.85*x-.656)+.015*cos(25.133*x-3.096);
    Float b = .28-.554*x+.504*cos(3.142*x-2.152)+.043*cos(6.283*x-1.134)+.012*cos(12.566*x+1.422)+.015*cos(18.85*x-.354);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GreenPinkTones(Float x) {
    Float r = -2.248+5.087*x+2.457*cos(3.142*x-.168)+.833*cos(6.283*x+1.801)+.085*cos(12.566*x+1.321)+.01*cos(18.85*x-2.587);
    Float g = 1.323-2.151*x+1.112*cos(3.142*x-2.654)+.483*cos(6.283*x-1.783)+.032*cos(12.566*x-1.064)+.024*cos(18.85*x+2.64);
    Float b = -1.316+3.182*x+1.554*cos(3.142*x-.302)+.619*cos(6.283*x+1.822)+.056*cos(12.566*x+1.205)+.019*cos(18.85*x-2.531);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrownCyanTones(Float x) {
    Float r = .112+.45*x+.534*cos(3.142*x-1.153);
    Float g = .491-.278*x+.691*cos(3.142*x-2.124)+.07*cos(6.283*x-.394);
    Float b = .601-.427*x+.746*cos(3.142*x-2.41)+.061*cos(6.283*x-.967);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 PigeonTones(Float x) {
    Float r = .72-.517*x+.673*cos(3.142*x-2.916)+.166*cos(6.283*x-.687);
    Float g = .568-.146*x+.497*cos(3.142*x-2.923)+.134*cos(6.283*x-.864);
    Float b = .655-.321*x+.563*cos(3.142*x-2.921)+.163*cos(6.283*x-.847);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CherryTones(Float x) {
    Float r = -.408+1.868*x+.664*cos(3.142*x-.658)+.074*cos(6.283*x+.212);
    Float g = .341+.514*x+.309*cos(3.142*x+2.032);
    Float b = .363+.469*x+.314*cos(3.142*x+2.1);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RedBlueTones(Float x) {
    Float r = -.186+1.151*x+.811*cos(3.142*x-.463)+.142*cos(6.283*x+2.33);
    Float g = 1.249-1.858*x+1.086*cos(3.142*x-2.732)+.163*cos(6.283*x-2.135);
    Float b = .737-.701*x+.616*cos(3.142*x-2.543);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 CoffeeTones(Float x) {
    Float r = .277+.715*x+.196*cos(3.142*x-1.224)+.05*cos(6.283*x-.153);
    Float g = .118+.933*x+.154*cos(3.142*x-.647)+.073*cos(6.283*x-.119);
    Float b = -.123+1.359*x+.333*cos(3.142*x+.43)+.083*cos(6.283*x-.343);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 RustTones(Float x) {
    Float r = .018+.905*x+.269*cos(3.142*x-1.772)+.011*cos(6.283*x-1.846)+.022*cos(12.566*x-.025);
    Float g = .058+.452*x+.03*cos(3.142*x-1.911)+.041*cos(6.283*x-3.104);
    Float b = .193-.155*x+.04*cos(3.142*x+1.542);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 FuchsiaTones(Float x) {
    Float r = .206+.665*x+.186*cos(3.142*x-2.168);
    Float g = .152+.539*x+.184*cos(3.142*x-2.488)+.086*cos(6.283*x+.228);
    Float b = .228+.626*x+.165*cos(3.142*x-2.404);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SiennaTones(Float x) {
    Float r = .245+.889*x+.305*cos(3.142*x-.818);
    Float g = .357+.339*x+.191*cos(3.142*x-2.797);
    Float b = .38+.12*x+.323*cos(3.142*x+2.769);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GrayTones(Float x) {
    Float r = .074+.728*x+.067*cos(3.142*x-2.376)+.064*cos(6.283*x+.119);
    Float g = .061+.784*x+.055*cos(3.142*x-2.002)+.051*cos(6.283*x+.078);
    Float b = .025+.821*x+.082*cos(3.142*x-1.403)+.049*cos(6.283*x+.057);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 ValentineTones(Float x) {
    Float r = .634+.233*x+.105*cos(3.142*x-2.985);
    Float g = .277+.406*x+.194*cos(3.142*x+2.54);
    Float b = .377+.335*x+.184*cos(3.142*x+2.726);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GrayYellowTones(Float x) {
    Float r = .534+.045*x+.376*cos(3.142*x-2.768);
    Float g = .689-.413*x+.517*cos(3.142*x-2.678);
    Float b = .997-1.658*x+1.04*cos(3.142*x-2.436)+.091*cos(6.283*x-.185);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkTerrain(Float x) {
    Float r = -.096+1.159*x+.07*cos(3.142*x+.46)+.173*cos(6.283*x-1.572)+.024*cos(12.566*x-.787);
    Float g = -.016+1.058*x+.103*cos(3.142*x+1.144)+.269*cos(6.283*x-1.574)+.031*cos(12.566*x-.719);
    Float b = 1.314-.929*x+.996*cos(3.142*x+2.413)+.42*cos(6.283*x-1.854)+.033*cos(12.566*x-1.018);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LightTerrain(Float x) {
    Float r = .836-.25*x+.299*cos(3.142*x-3.102);
    Float g = 1.177-.708*x+.424*cos(3.142*x+2.892);
    Float b = 1.373-.879*x+.631*cos(3.142*x+2.385)+.05*cos(6.283*x-2.97);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 GreenBrownTerrain(Float x) {
    Float r = -.456+2.198*x+.593*cos(3.142*x+.15)+.153*cos(6.283*x+2.825);
    Float g = -.72+2.757*x+.875*cos(3.142*x+.189)+.162*cos(6.283*x-3.03);
    Float b = -1.619+4.383*x+1.686*cos(3.142*x+.155)+.116*cos(6.283*x+2.243);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 SandyTerrain(Float x) {
    Float r = .856-.492*x+.237*cos(3.142*x-1.829)+.115*cos(6.283*x+3.084)+.026*cos(12.566*x-2.294);
    Float g = .956-.856*x+.46*cos(3.142*x-2.901)+.187*cos(6.283*x-2.624)+.022*cos(12.566*x-3.008);
    Float b = -.238+.949*x+.478*cos(3.142*x-.063)+.09*cos(6.283*x+2.03);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BrightBands(Float x) {
    Float r = -15.094+28.349*x+14.303*cos(3.142*x-.183)+3.387*cos(6.283*x+1.15)+.301*cos(12.566*x+.968)+.118*cos(18.85*x+.097)+.084*cos(25.133*x+.329);
    Float g = 31.195-61.605*x+31.094*cos(3.142*x-3.125)+6.593*cos(6.283*x-1.576)+.723*cos(12.566*x-1.552)+.246*cos(18.85*x-1.496)+.111*cos(25.133*x-1.355);
    Float b = -5.6+14.722*x+7.402*cos(3.142*x+.254)+1.395*cos(6.283*x+2.297)+.335*cos(12.566*x+2.606)+.09*cos(18.85*x-2.44)+.075*cos(25.133*x+1.97);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkBands(Float x) {
    Float r = -25.372+53.463*x+26.644*cos(3.142*x+.041)+5.688*cos(6.283*x+1.658)+.614*cos(12.566*x+1.508)+.243*cos(18.85*x+1.638)+.143*cos(25.133*x+2.06);
    Float g = -15.718+32.09*x+16.082*cos(3.142*x-.029)+3.459*cos(6.283*x+1.462)+.349*cos(12.566*x+1.378)+.214*cos(18.85*x+1.376)+.062*cos(25.133*x+1.487);
    Float b = 17.533-34.788*x+17.058*cos(3.142*x-3.103)+3.391*cos(6.283*x-1.454)+.303*cos(12.566*x-1.417)+.182*cos(18.85*x-1.31)+.016*cos(25.133*x-1.29);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Aquamarine(Float x) {
    Float r = .078+1.156*x+.593*cos(3.142*x+.088);
    Float g = .368+.735*x+.361*cos(3.142*x+.008);
    Float b = .535+.628*x+.317*cos(3.142*x+.232);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Pastel(Float x) {
    Float r = .52+.009*x+.523*cos(3.142*x-1.23)+.113*cos(6.283*x+1.097);
    Float g = -.471+2.033*x+.965*cos(3.142*x-.398)+.247*cos(6.283*x+1.404);
    Float b = -1.284+3.613*x+1.841*cos(3.142*x-.199)+.54*cos(6.283*x+.874)+.067*cos(12.566*x+.6);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 BlueGreenYellow(Float x) {
    Float r = .011+1.004*x+.353*cos(3.142*x+1.29);
    Float g = -.091+1.088*x+.218*cos(3.142*x-1.145);
    Float b = .221+.31*x+.232*cos(3.142*x-.722);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 Rainbow(Float x) {
    Float r = 2.01-2.371*x+1.427*cos(3.142*x+2.85)+.147*cos(6.283*x-2.611)+.03*cos(12.566*x+3.114);
    Float g = 2.007-3.531*x+1.814*cos(3.142*x-2.908)+.29*cos(6.283*x-1.923)+.034*cos(12.566*x-3.051);
    Float b = -.15+.385*x+.767*cos(3.142*x-1.048)+.281*cos(6.283*x-.393);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 DarkRainbow(Float x) {
    Float r = -.032+1.237*x+.371*cos(3.142*x+.131)+.256*cos(6.283*x+1.942)+.046*cos(12.566*x-1.755);
    Float g = .258+1.012*x+.689*cos(3.142*x+.645)+.441*cos(6.283*x+2.625)+.082*cos(12.566*x-2.507)+.009*cos(18.85*x-2.116);
    Float b = -1.472+3.662*x+1.998*cos(3.142*x+.024)+.488*cos(6.283*x+1.466);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 TemperatureMap(Float x) {
    Float r = 1.23-1.227*x+.958*cos(3.142*x-2.875)+.227*cos(6.283*x-2.127);
    Float g = 2.512-4.882*x+2.553*cos(3.142*x-2.731)+.391*cos(6.283*x-1.308)+.035*cos(12.566*x-1.27)+.021*cos(18.85*x-.69);
    Float b = -1.384+4.044*x+2.401*cos(3.142*x-.028)+.387*cos(6.283*x+1.914)+.09*cos(12.566*x+.862)+.028*cos(18.85*x+2.287);
    return vec3(clp(r),clp(g),clp(b));
  }
  static vec3 LightTemperatureMap(Float x) {
    Float r = 1.034-1.062*x+.955*cos(3.142*x-2.715)+.165*cos(6.283*x-1.705)+.031*cos(12.566*x-1.065);
    Float g = .056+.33*x+.875*cos(3.142*x-1.433)+.089*cos(6.283*x+.194);
    Float b = -.49+2.209*x+1.485*cos(3.142*x-.14)+.271*cos(6.283*x+1.846)+.039*cos(12.566*x+1.116);
    return vec3(clp(r),clp(g),clp(b));
  }
};
