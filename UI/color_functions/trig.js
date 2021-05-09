const ColorFunctions = {
  clp: function(x) {
    return Math.round(255.*(x<0.?0.:x>1.?1.:x));
  },
  tocol: function(r, g, b) {
    return 'rgb('+this.clp(r)+','+this.clp(g)+','+this.clp(b)+')';
  },
  AlpineColors: function(x) {
    var r = .388+.512*x+.181*Math.cos(3.142*x+2.181);
    var g = .199+.976*x+.199*Math.cos(3.142*x+.662);
    var b = -.512+2.345*x+.963*Math.cos(3.142*x+.199)+.226*Math.cos(6.283*x+1.363);
    return this.tocol(r, g, b);
  },
  LakeColors: function(x) {
    var r = .247+.739*x+.063*Math.cos(3.142*x-.917);
    var g = -.182+1.301*x+.32*Math.cos(3.142*x-.825)+.058*Math.cos(6.283*x+1.498);
    var b = .468+.592*x+.179*Math.cos(3.142*x-.667)+.06*Math.cos(6.283*x-2.996)+.021*Math.cos(12.566*x+2.472);
    return this.tocol(r, g, b);
  },
  ArmyColors: function(x) {
    var r = .947-.552*x+.451*Math.cos(3.142*x+2.811)+.108*Math.cos(6.283*x-2.077);
    var g = 1.304-1.217*x+.7*Math.cos(3.142*x+2.945)+.181*Math.cos(6.283*x-1.622);
    var b = 1.051-.861*x+.551*Math.cos(3.142*x+2.71)+.115*Math.cos(6.283*x-1.862);
    return this.tocol(r, g, b);
  },
  MintColors: function(x) {
    var r = .474+.436*x+.14*Math.cos(3.142*x-1.635);
    var g = .99-.38*x+.089*Math.cos(3.142*x-1.67);
    var b = .644+.141*x+.133*Math.cos(3.142*x-1.611);
    return this.tocol(r, g, b);
  },
  AtlanticColors: function(x) {
    var r = 1.607-2.72*x+1.547*Math.cos(3.142*x-2.941)+.28*Math.cos(6.283*x-1.388);
    var g = 1.575-2.507*x+1.453*Math.cos(3.142*x-2.883)+.236*Math.cos(6.283*x-1.572);
    var b = 1.633-2.506*x+1.51*Math.cos(3.142*x-2.895)+.23*Math.cos(6.283*x-1.544);
    return this.tocol(r, g, b);
  },
  NeonColors: function(x) {
    var r = .83-.148*x+.135*Math.cos(3.142*x-2.509);
    var g = 1.636-2.115*x+.729*Math.cos(3.142*x+2.815)+.088*Math.cos(6.283*x-1.428)+.036*Math.cos(12.566*x-2.08);
    var b = .532-.185*x+.339*Math.cos(3.142*x+2.888)+.123*Math.cos(6.283*x-.801);
    return this.tocol(r, g, b);
  },
  AuroraColors: function(x) {
    var r = 1.176-.831*x+.83*Math.cos(3.142*x+2.616)+.257*Math.cos(6.283*x-2.45);
    var g = .036+.564*x+.317*Math.cos(3.142*x-.493)+.248*Math.cos(6.283*x+1.791);
    var b = .017+.711*x+.26*Math.cos(3.142*x-1.555)+.235*Math.cos(6.283*x+.008);
    return this.tocol(r, g, b);
  },
  PearlColors: function(x) {
    var r = .111+1.453*x+.701*Math.cos(3.142*x+.168)+.105*Math.cos(6.283*x+.473);
    var g = .159+1.111*x+.562*Math.cos(3.142*x-.105)+.111*Math.cos(6.283*x+.216);
    var b = .069+1.266*x+.543*Math.cos(3.142*x-.288)+.182*Math.cos(6.283*x+.31);
    return this.tocol(r, g, b);
  },
  AvocadoColors: function(x) {
    var r = -1.482+3.952*x+1.476*Math.cos(3.142*x+.133)+.38*Math.cos(6.283*x+1.559)+.028*Math.cos(12.566*x+1.368);
    var g = -.251+1.483*x+.299*Math.cos(3.142*x-.687);
    var b = -.106+.439*x+.098*Math.cos(3.142*x+.065);
    return this.tocol(r, g, b);
  },
  PlumColors: function(x) {
    var r = -.729+2.374*x+.71*Math.cos(3.142*x-.046);
    var g = -.192+1.267*x+.263*Math.cos(3.142*x+.825);
    var b = .453-.47*x+.48*Math.cos(3.142*x-2.748);
    return this.tocol(r, g, b);
  },
  BeachColors: function(x) {
    var r = .325+1.524*x+.779*Math.cos(3.142*x+.527)+.147*Math.cos(6.283*x+3.076);
    var g = .03+1.694*x+.623*Math.cos(3.142*x+.36)+.116*Math.cos(6.283*x+3.087);
    var b = .535-.062*x+.406*Math.cos(3.142*x-2.94)+.121*Math.cos(6.283*x-.024);
    return this.tocol(r, g, b);
  },
  RoseColors: function(x) {
    var r = .299+.486*x+.117*Math.cos(3.142*x-1.852)+.109*Math.cos(6.283*x+3.069);
    var g = .503-.377*x+.171*Math.cos(3.142*x-2.184)+.089*Math.cos(6.283*x+3.038);
    var b = .165-.021*x+.174*Math.cos(3.142*x-1.691)+.054*Math.cos(6.283*x+3.058);
    return this.tocol(r, g, b);
  },
  CandyColors: function(x) {
    var r = .774-.379*x+.371*Math.cos(3.142*x-2.605)+.154*Math.cos(6.283*x-1.902);
    var g = .605-.004*x+.36*Math.cos(3.142*x+2.765)+.052*Math.cos(6.283*x-3.058);
    var b = .8-.375*x+.452*Math.cos(3.142*x-3.047);
    return this.tocol(r, g, b);
  },
  SolarColors: function(x) {
    var r = .257+.951*x+.306*Math.cos(3.142*x-.869);
    var g = .266+.288*x+.26*Math.cos(3.142*x+3.012);
    var b = .062+.016*x+.06*Math.cos(3.142*x+2.445);
    return this.tocol(r, g, b);
  },
  CMYKColors: function(x) {
    var r = 1.548-3.031*x+1.765*Math.cos(3.142*x-2.507)+.221*Math.cos(6.283*x-.724);
    var g = -1.734+3.258*x+2.223*Math.cos(3.142*x-.534)+.851*Math.cos(6.283*x+1.026)+.05*Math.cos(12.566*x+.109);
    var b = 1.389-2.952*x+1.531*Math.cos(3.142*x-2.345)+.546*Math.cos(6.283*x-.206)+.056*Math.cos(12.566*x+.462);
    return this.tocol(r, g, b);
  },
  SouthwestColors: function(x) {
    var r = -1.089+2.726*x+1.514*Math.cos(3.142*x-.434)+.337*Math.cos(6.283*x+1.286);
    var g = .992-.784*x+.529*Math.cos(3.142*x+3.123)+.144*Math.cos(6.283*x+3.137);
    var b = -.355+2.292*x+1.067*Math.cos(3.142*x+.711)+.312*Math.cos(6.283*x+2.973)+.046*Math.cos(12.566*x+.005)+.016*Math.cos(18.85*x+.014)+.007*Math.cos(25.133*x+2.161);
    return this.tocol(r, g, b);
  },
  DeepSeaColors: function(x) {
    var r = -.865+2.54*x+.968*Math.cos(3.142*x+.133)+.081*Math.cos(6.283*x+1.016)+.025*Math.cos(12.566*x+1.009)+.021*Math.cos(18.85*x+1.301);
    var g = .323+.28*x+.339*Math.cos(3.142*x+2.74);
    var b = .35+.599*x+.187*Math.cos(3.142*x-1.847);
    return this.tocol(r, g, b);
  },
  StarryNightColors: function(x) {
    var r = -.092+1.192*x+.18*Math.cos(3.142*x-.53)+.115*Math.cos(6.283*x+1.394);
    var g = .469+.021*x+.379*Math.cos(3.142*x-2.529);
    var b = .368-.19*x+.393*Math.cos(3.142*x-2.022);
    return this.tocol(r, g, b);
  },
  FallColors: function(x) {
    var r = .404+.222*x+.286*Math.cos(3.142*x-2.557)+.088*Math.cos(6.283*x+.067);
    var g = .272+.386*x+.009*Math.cos(3.142*x+2.949)+.123*Math.cos(6.283*x+.009);
    var b = .477-.312*x+.136*Math.cos(3.142*x+2.254);
    return this.tocol(r, g, b);
  },
  SunsetColors: function(x) {
    var r = 1.002-.806*x+.937*Math.cos(3.142*x-2.848)+.246*Math.cos(6.283*x-1.979);
    var g = -.166+1.201*x+.124*Math.cos(3.142*x-.699)+.089*Math.cos(6.283*x+.827);
    var b = 7.061-13.028*x+7.02*Math.cos(3.142*x+3.092)+1.788*Math.cos(6.283*x-1.526)+.194*Math.cos(12.566*x-1.916)+.049*Math.cos(18.85*x-2.292)+.015*Math.cos(25.133*x-2.897);
    return this.tocol(r, g, b);
  },
  FruitPunchColors: function(x) {
    var r = .666+.482*x+.261*Math.cos(3.142*x+.185)+.072*Math.cos(6.283*x+.1);
    var g = .087+.841*x+.481*Math.cos(3.142*x-.074)+.076*Math.cos(6.283*x+3.028);
    var b = .743-1.307*x+.97*Math.cos(3.142*x-2.782)+.17*Math.cos(6.283*x-.077);
    return this.tocol(r, g, b);
  },
  ThermometerColors: function(x) {
    var r = 1.011-1.233*x+.9*Math.cos(3.142*x-2.659)+.105*Math.cos(6.283*x-2.117);
    var g = .121+.077*x+.664*Math.cos(3.142*x-1.49)+.084*Math.cos(6.283*x-2.796);
    var b = .428+.096*x+.501*Math.cos(3.142*x-.759);
    return this.tocol(r, g, b);
  },
  IslandColors: function(x) {
    var r = 2.031-2.229*x+1.09*Math.cos(3.142*x+2.856)+.201*Math.cos(6.283*x-2.984);
    var g = -.421+1.871*x+.847*Math.cos(3.142*x-.565)+.123*Math.cos(6.283*x+1.123);
    var b = -1.173+2.65*x+1.509*Math.cos(3.142*x-.584)+.207*Math.cos(6.283*x+1.087);
    return this.tocol(r, g, b);
  },
  WatermelonColors: function(x) {
    var r = .245+.256*x+.467*Math.cos(3.142*x-2.172)+.108*Math.cos(6.283*x+.008);
    var g = .418-.663*x+.956*Math.cos(3.142*x-2.066)+.123*Math.cos(6.283*x-.028);
    var b = .814-1.542*x+1.128*Math.cos(3.142*x-2.473)+.167*Math.cos(6.283*x-.048);
    return this.tocol(r, g, b);
  },
  BrassTones: function(x) {
    var r = .978-1.276*x+.796*Math.cos(3.142*x-2.527)+.183*Math.cos(6.283*x-2.455)+.043*Math.cos(12.566*x+2.614)+.018*Math.cos(18.85*x-.696)+.017*Math.cos(25.133*x-3.093);
    var g = .83-1.005*x+.637*Math.cos(3.142*x-2.484)+.156*Math.cos(6.283*x-2.603)+.04*Math.cos(12.566*x+2.521)+.017*Math.cos(18.85*x-.656)+.015*Math.cos(25.133*x-3.096);
    var b = .28-.554*x+.504*Math.cos(3.142*x-2.152)+.043*Math.cos(6.283*x-1.134)+.012*Math.cos(12.566*x+1.422)+.015*Math.cos(18.85*x-.354);
    return this.tocol(r, g, b);
  },
  GreenPinkTones: function(x) {
    var r = -2.248+5.087*x+2.457*Math.cos(3.142*x-.168)+.833*Math.cos(6.283*x+1.801)+.085*Math.cos(12.566*x+1.321)+.01*Math.cos(18.85*x-2.587);
    var g = 1.323-2.151*x+1.112*Math.cos(3.142*x-2.654)+.483*Math.cos(6.283*x-1.783)+.032*Math.cos(12.566*x-1.064)+.024*Math.cos(18.85*x+2.64);
    var b = -1.316+3.182*x+1.554*Math.cos(3.142*x-.302)+.619*Math.cos(6.283*x+1.822)+.056*Math.cos(12.566*x+1.205)+.019*Math.cos(18.85*x-2.531);
    return this.tocol(r, g, b);
  },
  BrownCyanTones: function(x) {
    var r = .112+.45*x+.534*Math.cos(3.142*x-1.153);
    var g = .491-.278*x+.691*Math.cos(3.142*x-2.124)+.07*Math.cos(6.283*x-.394);
    var b = .601-.427*x+.746*Math.cos(3.142*x-2.41)+.061*Math.cos(6.283*x-.967);
    return this.tocol(r, g, b);
  },
  PigeonTones: function(x) {
    var r = .72-.517*x+.673*Math.cos(3.142*x-2.916)+.166*Math.cos(6.283*x-.687);
    var g = .568-.146*x+.497*Math.cos(3.142*x-2.923)+.134*Math.cos(6.283*x-.864);
    var b = .655-.321*x+.563*Math.cos(3.142*x-2.921)+.163*Math.cos(6.283*x-.847);
    return this.tocol(r, g, b);
  },
  CherryTones: function(x) {
    var r = -.408+1.868*x+.664*Math.cos(3.142*x-.658)+.074*Math.cos(6.283*x+.212);
    var g = .341+.514*x+.309*Math.cos(3.142*x+2.032);
    var b = .363+.469*x+.314*Math.cos(3.142*x+2.1);
    return this.tocol(r, g, b);
  },
  RedBlueTones: function(x) {
    var r = -.186+1.151*x+.811*Math.cos(3.142*x-.463)+.142*Math.cos(6.283*x+2.33);
    var g = 1.249-1.858*x+1.086*Math.cos(3.142*x-2.732)+.163*Math.cos(6.283*x-2.135);
    var b = .737-.701*x+.616*Math.cos(3.142*x-2.543);
    return this.tocol(r, g, b);
  },
  CoffeeTones: function(x) {
    var r = .277+.715*x+.196*Math.cos(3.142*x-1.224)+.05*Math.cos(6.283*x-.153);
    var g = .118+.933*x+.154*Math.cos(3.142*x-.647)+.073*Math.cos(6.283*x-.119);
    var b = -.123+1.359*x+.333*Math.cos(3.142*x+.43)+.083*Math.cos(6.283*x-.343);
    return this.tocol(r, g, b);
  },
  RustTones: function(x) {
    var r = .018+.905*x+.269*Math.cos(3.142*x-1.772)+.011*Math.cos(6.283*x-1.846)+.022*Math.cos(12.566*x-.025);
    var g = .058+.452*x+.03*Math.cos(3.142*x-1.911)+.041*Math.cos(6.283*x-3.104);
    var b = .193-.155*x+.04*Math.cos(3.142*x+1.542);
    return this.tocol(r, g, b);
  },
  FuchsiaTones: function(x) {
    var r = .206+.665*x+.186*Math.cos(3.142*x-2.168);
    var g = .152+.539*x+.184*Math.cos(3.142*x-2.488)+.086*Math.cos(6.283*x+.228);
    var b = .228+.626*x+.165*Math.cos(3.142*x-2.404);
    return this.tocol(r, g, b);
  },
  SiennaTones: function(x) {
    var r = .245+.889*x+.305*Math.cos(3.142*x-.818);
    var g = .357+.339*x+.191*Math.cos(3.142*x-2.797);
    var b = .38+.12*x+.323*Math.cos(3.142*x+2.769);
    return this.tocol(r, g, b);
  },
  GrayTones: function(x) {
    var r = .074+.728*x+.067*Math.cos(3.142*x-2.376)+.064*Math.cos(6.283*x+.119);
    var g = .061+.784*x+.055*Math.cos(3.142*x-2.002)+.051*Math.cos(6.283*x+.078);
    var b = .025+.821*x+.082*Math.cos(3.142*x-1.403)+.049*Math.cos(6.283*x+.057);
    return this.tocol(r, g, b);
  },
  ValentineTones: function(x) {
    var r = .634+.233*x+.105*Math.cos(3.142*x-2.985);
    var g = .277+.406*x+.194*Math.cos(3.142*x+2.54);
    var b = .377+.335*x+.184*Math.cos(3.142*x+2.726);
    return this.tocol(r, g, b);
  },
  GrayYellowTones: function(x) {
    var r = .534+.045*x+.376*Math.cos(3.142*x-2.768);
    var g = .689-.413*x+.517*Math.cos(3.142*x-2.678);
    var b = .997-1.658*x+1.04*Math.cos(3.142*x-2.436)+.091*Math.cos(6.283*x-.185);
    return this.tocol(r, g, b);
  },
  DarkTerrain: function(x) {
    var r = -.096+1.159*x+.07*Math.cos(3.142*x+.46)+.173*Math.cos(6.283*x-1.572)+.024*Math.cos(12.566*x-.787);
    var g = -.016+1.058*x+.103*Math.cos(3.142*x+1.144)+.269*Math.cos(6.283*x-1.574)+.031*Math.cos(12.566*x-.719);
    var b = 1.314-.929*x+.996*Math.cos(3.142*x+2.413)+.42*Math.cos(6.283*x-1.854)+.033*Math.cos(12.566*x-1.018);
    return this.tocol(r, g, b);
  },
  LightTerrain: function(x) {
    var r = .836-.25*x+.299*Math.cos(3.142*x-3.102);
    var g = 1.177-.708*x+.424*Math.cos(3.142*x+2.892);
    var b = 1.373-.879*x+.631*Math.cos(3.142*x+2.385)+.05*Math.cos(6.283*x-2.97);
    return this.tocol(r, g, b);
  },
  GreenBrownTerrain: function(x) {
    var r = -.456+2.198*x+.593*Math.cos(3.142*x+.15)+.153*Math.cos(6.283*x+2.825);
    var g = -.72+2.757*x+.875*Math.cos(3.142*x+.189)+.162*Math.cos(6.283*x-3.03);
    var b = -1.619+4.383*x+1.686*Math.cos(3.142*x+.155)+.116*Math.cos(6.283*x+2.243);
    return this.tocol(r, g, b);
  },
  SandyTerrain: function(x) {
    var r = .856-.492*x+.237*Math.cos(3.142*x-1.829)+.115*Math.cos(6.283*x+3.084)+.026*Math.cos(12.566*x-2.294);
    var g = .956-.856*x+.46*Math.cos(3.142*x-2.901)+.187*Math.cos(6.283*x-2.624)+.022*Math.cos(12.566*x-3.008);
    var b = -.238+.949*x+.478*Math.cos(3.142*x-.063)+.09*Math.cos(6.283*x+2.03);
    return this.tocol(r, g, b);
  },
  BrightBands: function(x) {
    var r = -15.094+28.349*x+14.303*Math.cos(3.142*x-.183)+3.387*Math.cos(6.283*x+1.15)+.301*Math.cos(12.566*x+.968)+.118*Math.cos(18.85*x+.097)+.084*Math.cos(25.133*x+.329);
    var g = 31.195-61.605*x+31.094*Math.cos(3.142*x-3.125)+6.593*Math.cos(6.283*x-1.576)+.723*Math.cos(12.566*x-1.552)+.246*Math.cos(18.85*x-1.496)+.111*Math.cos(25.133*x-1.355);
    var b = -5.6+14.722*x+7.402*Math.cos(3.142*x+.254)+1.395*Math.cos(6.283*x+2.297)+.335*Math.cos(12.566*x+2.606)+.09*Math.cos(18.85*x-2.44)+.075*Math.cos(25.133*x+1.97);
    return this.tocol(r, g, b);
  },
  DarkBands: function(x) {
    var r = -25.372+53.463*x+26.644*Math.cos(3.142*x+.041)+5.688*Math.cos(6.283*x+1.658)+.614*Math.cos(12.566*x+1.508)+.243*Math.cos(18.85*x+1.638)+.143*Math.cos(25.133*x+2.06);
    var g = -15.718+32.09*x+16.082*Math.cos(3.142*x-.029)+3.459*Math.cos(6.283*x+1.462)+.349*Math.cos(12.566*x+1.378)+.214*Math.cos(18.85*x+1.376)+.062*Math.cos(25.133*x+1.487);
    var b = 17.533-34.788*x+17.058*Math.cos(3.142*x-3.103)+3.391*Math.cos(6.283*x-1.454)+.303*Math.cos(12.566*x-1.417)+.182*Math.cos(18.85*x-1.31)+.016*Math.cos(25.133*x-1.29);
    return this.tocol(r, g, b);
  },
  Aquamarine: function(x) {
    var r = .078+1.156*x+.593*Math.cos(3.142*x+.088);
    var g = .368+.735*x+.361*Math.cos(3.142*x+.008);
    var b = .535+.628*x+.317*Math.cos(3.142*x+.232);
    return this.tocol(r, g, b);
  },
  Pastel: function(x) {
    var r = .52+.009*x+.523*Math.cos(3.142*x-1.23)+.113*Math.cos(6.283*x+1.097);
    var g = -.471+2.033*x+.965*Math.cos(3.142*x-.398)+.247*Math.cos(6.283*x+1.404);
    var b = -1.284+3.613*x+1.841*Math.cos(3.142*x-.199)+.54*Math.cos(6.283*x+.874)+.067*Math.cos(12.566*x+.6);
    return this.tocol(r, g, b);
  },
  BlueGreenYellow: function(x) {
    var r = .011+1.004*x+.353*Math.cos(3.142*x+1.29);
    var g = -.091+1.088*x+.218*Math.cos(3.142*x-1.145);
    var b = .221+.31*x+.232*Math.cos(3.142*x-.722);
    return this.tocol(r, g, b);
  },
  Rainbow: function(x) {
    var r = 2.01-2.371*x+1.427*Math.cos(3.142*x+2.85)+.147*Math.cos(6.283*x-2.611)+.03*Math.cos(12.566*x+3.114);
    var g = 2.007-3.531*x+1.814*Math.cos(3.142*x-2.908)+.29*Math.cos(6.283*x-1.923)+.034*Math.cos(12.566*x-3.051);
    var b = -.15+.385*x+.767*Math.cos(3.142*x-1.048)+.281*Math.cos(6.283*x-.393);
    return this.tocol(r, g, b);
  },
  DarkRainbow: function(x) {
    var r = -.032+1.237*x+.371*Math.cos(3.142*x+.131)+.256*Math.cos(6.283*x+1.942)+.046*Math.cos(12.566*x-1.755);
    var g = .258+1.012*x+.689*Math.cos(3.142*x+.645)+.441*Math.cos(6.283*x+2.625)+.082*Math.cos(12.566*x-2.507)+.009*Math.cos(18.85*x-2.116);
    var b = -1.472+3.662*x+1.998*Math.cos(3.142*x+.024)+.488*Math.cos(6.283*x+1.466);
    return this.tocol(r, g, b);
  },
  TemperatureMap: function(x) {
    var r = 1.23-1.227*x+.958*Math.cos(3.142*x-2.875)+.227*Math.cos(6.283*x-2.127);
    var g = 2.512-4.882*x+2.553*Math.cos(3.142*x-2.731)+.391*Math.cos(6.283*x-1.308)+.035*Math.cos(12.566*x-1.27)+.021*Math.cos(18.85*x-.69);
    var b = -1.384+4.044*x+2.401*Math.cos(3.142*x-.028)+.387*Math.cos(6.283*x+1.914)+.09*Math.cos(12.566*x+.862)+.028*Math.cos(18.85*x+2.287);
    return this.tocol(r, g, b);
  },
  LightTemperatureMap: function(x) {
    var r = 1.034-1.062*x+.955*Math.cos(3.142*x-2.715)+.165*Math.cos(6.283*x-1.705)+.031*Math.cos(12.566*x-1.065);
    var g = .056+.33*x+.875*Math.cos(3.142*x-1.433)+.089*Math.cos(6.283*x+.194);
    var b = -.49+2.209*x+1.485*Math.cos(3.142*x-.14)+.271*Math.cos(6.283*x+1.846)+.039*Math.cos(12.566*x+1.116);
    return this.tocol(r, g, b);
  },
};