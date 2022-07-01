vec3 AlpineColorsT(float x) {
    return clamp(vec3(.39,.2,.73)+vec3(.51,.98,-.01)*x+vec3(.18,.2,.35)*cos(3.14*x+vec3(2.18,.66,2.16)),0.,1.);
}
vec3 LakeColorsT(float x) {
    return clamp(vec3(.25,.14,.27)+vec3(.74,.68,.8)*x+vec3(.06,.24,.34)*cos(3.14*x+vec3(-.92,-1.93,-.83)),0.,1.);
}
vec3 ArmyColorsT(float x) {
    return clamp(vec3(.37,.33,.41)+vec3(.46,.71,.32)*x+vec3(.07,.27,.17)*cos(3.14*x+vec3(.44,.44,1.14)),0.,1.);
}
vec3 MintColorsT(float x) {
    return clamp(vec3(.47,.99,.64)+vec3(.44,-.38,.14)*x+vec3(.14,.09,.13)*cos(3.14*x+vec3(-1.64,-1.67,-1.61)),0.,1.);
}
vec3 AtlanticColorsT(float x) {
    return clamp(vec3(1.61,.32,.41)+vec3(-2.72,.01,-.05)*x+vec3(1.55,.42,.45)*cos(3.14*x+vec3(-2.94,-2.03,-2.23))+vec3(.28,0,0)*cos(6.28*x+vec3(-1.39,0,0)),0.,1.);
}
vec3 NeonColorsT(float x) {
    return clamp(vec3(.83,-.12,.53)+vec3(-.15,1.19,-.19)*x+vec3(.14,.95,.34)*cos(3.14*x+vec3(-2.51,.08,2.89))+vec3(0,.26,.12)*cos(6.28*x+vec3(0,1.26,-.8)),0.,1.);
}
vec3 AuroraColorsT(float x) {
    return clamp(vec3(1.18,1.25,.02)+vec3(-.83,-2.02,.71)*x+vec3(.83,1.01,.26)*cos(3.14*x+vec3(2.62,-2.87,-1.55))+vec3(.26,0,.24)*cos(6.28*x+vec3(-2.45,0,.01)),0.,1.);
}
vec3 PearlColorsT(float x) {
    return clamp(vec3(.5,.16,.07)+vec3(.94,1.11,1.27)*x+vec3(.55,.56,.54)*cos(3.14*x+vec3(.64,-.11,-.29))+vec3(0,.11,.18)*cos(6.28*x+vec3(0,.22,.31)),0.,1.);
}
vec3 AvocadoColorsT(float x) {
    return clamp(vec3(.43,-.25,-.11)+vec3(.14,1.48,.44)*x+vec3(.46,.3,.1)*cos(3.14*x+vec3(2.68,-.69,.06)),0.,1.);
}
vec3 PlumColorsT(float x) {
    return clamp(vec3(-.73,-.19,.45)+vec3(2.37,1.27,-.47)*x+vec3(.71,.26,.48)*cos(3.14*x+vec3(-.05,.83,-2.75)),0.,1.);
}
vec3 BeachColorsT(float x) {
    return clamp(vec3(.33,.03,.54)+vec3(1.52,1.69,-.06)*x+vec3(.78,.62,.41)*cos(3.14*x+vec3(.53,.36,-2.94))+vec3(.15,.12,.12)*cos(6.28*x+vec3(3.08,3.09,-.02)),0.,1.);
}
vec3 RoseColorsT(float x) {
    return clamp(vec3(.3,.43,.11)+vec3(.49,-.48,-.07)*x+vec3(.12,.37,.3)*cos(3.14*x+vec3(-1.85,-1.98,-1.72))+vec3(.11,0,0)*cos(6.28*x+vec3(3.07,0,0)),0.,1.);
}
vec3 CandyColorsT(float x) {
    return clamp(vec3(-.07,.51,.8)+vec3(1.17,.04,-.38)*x+vec3(.53,.31,.45)*cos(3.14*x+vec3(-.61,3.09,-3.05)),0.,1.);
}
vec3 SolarColorsT(float x) {
    return clamp(vec3(.26,.27,.06)+vec3(.95,.29,.02)*x+vec3(.31,.26,.06)*cos(3.14*x+vec3(-.87,3.01,2.45)),0.,1.);
}
vec3 CMYKColorsT(float x) {
    return clamp(vec3(1.55,-1.73,1.39)+vec3(-3.03,3.26,-2.95)*x+vec3(1.77,2.22,1.53)*cos(3.14*x+vec3(-2.51,-.53,-2.35))+vec3(.22,.85,.55)*cos(6.28*x+vec3(-.72,1.03,-.21))+vec3(0,.05,.06)*cos(12.57*x+vec3(0,.11,.46)),0.,1.);
}
vec3 SouthwestColorsT(float x) {
    return clamp(vec3(-1.09,.99,-.13)+vec3(2.73,-.78,2.44)*x+vec3(1.51,.53,1.46)*cos(3.14*x+vec3(-.43,3.12,.93))+vec3(.34,.14,.51)*cos(6.28*x+vec3(1.29,3.14,3.02)),0.,1.);
}
vec3 DeepSeaColorsT(float x) {
    return clamp(vec3(-.6,.32,.35)+vec3(2.14,.28,.6)*x+vec3(.76,.34,.19)*cos(3.14*x+vec3(.31,2.74,-1.85)),0.,1.);
}
vec3 StarryNightColorsT(float x) {
    return clamp(vec3(.54,.47,.37)+vec3(-.01,.02,-.19)*x+vec3(.43,.38,.39)*cos(3.14*x+vec3(-3.04,-2.53,-2.02)),0.,1.);
}
vec3 FallColorsT(float x) {
    return clamp(vec3(.56,.27,.48)+vec3(.16,.39,-.31)*x+vec3(.27,.01,.14)*cos(3.14*x+vec3(3,2.95,2.25))+vec3(0,.12,0)*cos(6.28*x+vec3(0,.01,0)),0.,1.);
}
vec3 SunsetColorsT(float x) {
    return clamp(vec3(1,.27,.14)+vec3(-.81,.5,-.07)*x+vec3(.94,.25,.67)*cos(3.14*x+vec3(-2.85,2.92,-2.58))+vec3(.25,0,.58)*cos(6.28*x+vec3(-1.98,0,-.86))+vec3(0,0,.06)*cos(12.57*x+vec3(0,0,-1.68)),0.,1.);
}
vec3 FruitPunchColorsT(float x) {
    return clamp(vec3(.81,.03,.74)+vec3(.4,.75,-1.31)*x+vec3(.3,.48,.97)*cos(3.14*x+vec3(.76,-.44,-2.78))+vec3(0,0,.17)*cos(6.28*x+vec3(0,0,-.08)),0.,1.);
}
vec3 ThermometerColorsT(float x) {
    return clamp(vec3(.45,-.14,.43)+vec3(-.28,.38,.1)*x+vec3(.63,.86,.5)*cos(3.14*x+vec3(-2.12,-1.34,-.76)),0.,1.);
}
vec3 IslandColorsT(float x) {
    return clamp(vec3(2.03,.25,-1.17)+vec3(-2.23,.69,2.65)*x+vec3(1.09,.36,1.51)*cos(3.14*x+vec3(2.86,-1.17,-.58))+vec3(.2,0,.21)*cos(6.28*x+vec3(-2.98,0,1.09)),0.,1.);
}
vec3 WatermelonColorsT(float x) {
    return clamp(vec3(.25,.42,.81)+vec3(.26,-.66,-1.54)*x+vec3(.47,.96,1.13)*cos(3.14*x+vec3(-2.17,-2.07,-2.47))+vec3(.11,.12,.17)*cos(6.28*x+vec3(.01,-.03,-.05)),0.,1.);
}
vec3 BrassTonesT(float x) {
    return clamp(vec3(-1.03,-.89,.07)+vec3(2.84,2.55,-.07)*x+vec3(1.45,1.3,.38)*cos(3.14*x+vec3(-.26,-.24,-1.7))+vec3(.36,.34,0)*cos(6.28*x+vec3(2.08,2.1,0))+vec3(.08,.07,0)*cos(12.57*x+vec3(2.19,2.17,0)),0.,1.);
}
vec3 GreenPinkTonesT(float x) {
    return clamp(vec3(-3.25,3.08,-3.34)+vec3(6.9,-6.12,6.89)*x+vec3(3.37,3.09,3.41)*cos(3.14*x+vec3(-.17,-2.85,-.22))+vec3(1.01,.89,.99)*cos(6.28*x+vec3(1.7,-1.51,1.61))+vec3(.11,.08,.1)*cos(12.57*x+vec3(1.25,-.96,1.12)),0.,1.);
}
vec3 BrownCyanTonesT(float x) {
    return clamp(vec3(.11,.44,.38)+vec3(.45,.01,.11)*x+vec3(.53,.5,.51)*cos(3.14*x+vec3(-1.15,-2.04,-2.18)),0.,1.);
}
vec3 PigeonTonesT(float x) {
    return clamp(vec3(.72,.15,.66)+vec3(-.52,.94,-.32)*x+vec3(.67,.1,.56)*cos(3.14*x+vec3(-2.92,1.13,-2.92))+vec3(.17,0,.16)*cos(6.28*x+vec3(-.69,0,-.85)),0.,1.);
}
vec3 CherryTonesT(float x) {
    return clamp(vec3(-.22,.34,.36)+vec3(1.7,.51,.47)*x+vec3(.51,.31,.31)*cos(3.14*x+vec3(-.5,2.03,2.1)),0.,1.);
}
vec3 RedBlueTonesT(float x) {
    return clamp(vec3(-.19,1.25,.74)+vec3(1.15,-1.86,-.7)*x+vec3(.81,1.09,.62)*cos(3.14*x+vec3(-.46,-2.73,-2.54))+vec3(.14,.16,0)*cos(6.28*x+vec3(2.33,-2.13,0)),0.,1.);
}
vec3 CoffeeTonesT(float x) {
    return clamp(vec3(.31,.17,-.16)+vec3(.8,1.03,1.66)*x+vec3(.13,.18,.55)*cos(3.14*x+vec3(-.61,.39,.61)),0.,1.);
}
vec3 RustTonesT(float x) {
    return clamp(vec3(.12,-.01,.19)+vec3(.96,.47,-.15)*x+vec3(.06,.12,.04)*cos(3.14*x+vec3(-1.97,-1.59,1.54))+vec3(.09,0,0)*cos(6.28*x+vec3(-3.09,0,0)),0.,1.);
}
vec3 FuchsiaTonesT(float x) {
    return clamp(vec3(.21,.38,.23)+vec3(.66,.33,.63)*x+vec3(.19,.26,.16)*cos(3.14*x+vec3(-2.17,2.84,-2.4)),0.,1.);
}
vec3 SiennaTonesT(float x) {
    return clamp(vec3(.24,.36,.38)+vec3(.89,.34,.12)*x+vec3(.3,.19,.32)*cos(3.14*x+vec3(-.82,-2.8,2.77)),0.,1.);
}
vec3 GrayTonesT(float x) {
    return clamp(vec3(.2,.16,.11)+vec3(.65,.74,.79)*x+vec3(.13,.08,.03)*cos(3.14*x+vec3(2.31,2.16,1.59)),0.,1.);
}
vec3 ValentineTonesT(float x) {
    return clamp(vec3(.63,.28,.38)+vec3(.23,.41,.34)*x+vec3(.11,.19,.18)*cos(3.14*x+vec3(-2.98,2.54,2.73)),0.,1.);
}
vec3 GrayYellowTonesT(float x) {
    return clamp(vec3(.53,.69,1.04)+vec3(.05,-.41,-1.48)*x+vec3(.38,.52,.85)*cos(3.14*x+vec3(-2.77,-2.68,-2.55)),0.,1.);
}
vec3 DarkTerrainT(float x) {
    return clamp(vec3(-.93,-.95,-.09)+vec3(2.85,3.2,2.08)*x+vec3(.9,1.14,1.11)*cos(3.14*x+vec3(.04,.27,.83))+vec3(0,.1,.21)*cos(6.28*x+vec3(0,-2.62,-2.66)),0.,1.);
}
vec3 LightTerrainT(float x) {
    return clamp(vec3(.84,1.18,1.26)+vec3(-.25,-.71,-.79)*x+vec3(.3,.42,.52)*cos(3.14*x+vec3(-3.1,2.89,2.48)),0.,1.);
}
vec3 GreenBrownTerrainT(float x) {
    return clamp(vec3(-.46,-.72,-1.24)+vec3(2.2,2.76,3.41)*x+vec3(.59,.88,1.2)*cos(3.14*x+vec3(.15,.19,.08))+vec3(.15,.16,0)*cos(6.28*x+vec3(2.83,-3.03,0)),0.,1.);
}
vec3 SandyTerrainT(float x) {
    return clamp(vec3(-.27,.24,.14)+vec3(1.57,.1,.09)*x+vec3(1.03,.48,.13)*cos(3.14*x+vec3(-.38,-1.53,-1.11))+vec3(.22,0,0)*cos(6.28*x+vec3(1.79,0,0)),0.,1.);
}
vec3 BrightBandsT(float x) {
    return clamp(vec3(-10.69,-13.44,-14.93)+vec3(23.11,28.36,31.5)*x+vec3(11.44,13.75,15.46)*cos(3.14*x+vec3(.02,0,.02))+vec3(2.54,2.84,2.76)*cos(6.28*x+vec3(1.49,1.66,1.68))+vec3(.2,.19,.36)*cos(12.57*x+vec3(1.91,1.75,2.05)),0.,1.);
}
vec3 DarkBandsT(float x) {
    return clamp(vec3(19.05,22.62,-12.71)+vec3(-36.56,-43.6,26.79)*x+vec3(18.26,21.63,13.6)*cos(3.14*x+vec3(3.13,3.13,.01))+vec3(3.78,4.47,3.04)*cos(6.28*x+vec3(-1.6,-1.56,1.56))+vec3(.32,.41,.31)*cos(12.57*x+vec3(-1.2,-1.57,1.66)),0.,1.);
}
vec3 AquamarineT(float x) {
    return clamp(vec3(.08,.37,.54)+vec3(1.16,.74,.63)*x+vec3(.59,.36,.32)*cos(3.14*x+vec3(.09,.01,.23)),0.,1.);
}
vec3 PastelT(float x) {
    return clamp(vec3(1.13,.89,-1.28)+vec3(-1.06,-.57,3.61)*x+vec3(.51,.47,1.84)*cos(3.14*x+vec3(-2.31,-2.49,-.2))+vec3(0,0,.54)*cos(6.28*x+vec3(0,0,.87))+vec3(0,0,.07)*cos(12.57*x+vec3(0,0,.6)),0.,1.);
}
vec3 BlueGreenYellowT(float x) {
    return clamp(vec3(.01,-.09,.22)+vec3(1,1.09,.31)*x+vec3(.35,.22,.23)*cos(3.14*x+vec3(1.29,-1.14,-.72)),0.,1.);
}
vec3 RainbowT(float x) {
    return clamp(vec3(1.42,1.64,-.15)+vec3(-1.57,-3.2,.38)*x+vec3(.99,1.76,.77)*cos(3.14*x+vec3(3.03,-2.71,-1.05))+vec3(0,.24,.28)*cos(6.28*x+vec3(0,-1.44,-.39)),0.,1.);
}
vec3 DarkRainbowT(float x) {
    return clamp(vec3(-2.48,-1.09,-1.47)+vec3(6.03,3.61,3.66)*x+vec3(2.74,1.88,2)*cos(3.14*x+vec3(-.01,.18,.02))+vec3(.73,.6,.49)*cos(6.28*x+vec3(1.65,2.19,1.47))+vec3(0,.06,0)*cos(12.57*x+vec3(0,-2.77,0)),0.,1.);
}
vec3 TemperatureMapT(float x) {
    return clamp(vec3(1.23,.85,2.05)+vec3(-1.23,-1.25,-3.22)*x+vec3(.96,.95,1.27)*cos(3.14*x+vec3(-2.87,-2.17,-2.84))+vec3(.23,0,.39)*cos(6.28*x+vec3(-2.13,0,-1.56))+vec3(0,0,.09)*cos(12.57*x+vec3(0,0,-.03)),0.,1.);
}
vec3 LightTemperatureMapT(float x) {
    return clamp(vec3(-.33,.27,1.48)+vec3(1.85,.15,-1.54)*x+vec3(.62,.67,.39)*cos(3.14*x+vec3(-.42,-1.52,-3.01))+vec3(.15,0,.18)*cos(6.28*x+vec3(2.12,0,-2.43)),0.,1.);
}
