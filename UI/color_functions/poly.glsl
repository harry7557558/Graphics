vec3 AlpineColorsP(float t) {
    return clamp((((vec3(0,0,-2.66)*t+vec3(0,0,4.45))*t+vec3(.61,.5,-.91))*t+vec3(.15,.09,-.5))*t+vec3(.27,.4,.51),0.,1.);
}
vec3 LakeColorsP(float t) {
    return clamp(((vec3(0,0,1.02)*t+vec3(-.21,-.93,-2.57))*t+vec3(.85,1.81,1.9))*t+vec3(.29,.02,.49),0.,1.);
}
vec3 ArmyColorsP(float t) {
    return clamp(((vec3(0,1.1,0)*t+vec3(.13,-1.17,.65))*t+vec3(.18,.29,-.5))*t+vec3(.45,.58,.51),0.,1.);
}
vec3 MintColorsP(float t) {
    return clamp((vec3(-.58,-.37,-.55)*t+vec3(1.04,.01,.7))*t+vec3(.46,.97,.63),0.,1.);
}
vec3 AtlanticColorsP(float t) {
    return clamp((((vec3(-.97,0,0)*t+vec3(1.5,-.85,-1.25))*t+vec3(-1.4,-.25,.41))*t+vec3(1.26,1.48,1.33))*t+vec3(.1,.12,.13),0.,1.);
}
vec3 NeonColorsP(float t) {
    return clamp((((vec3(0,-3.39,-3.65)*t+vec3(0,5.26,7.87))*t+vec3(-.33,-1.04,-4.42))*t+vec3(.44,-1.62,.7))*t+vec3(.69,.95,.27),0.,1.);
}
vec3 AuroraColorsP(float t) {
    return clamp((((vec3(6.62,2.78,-8.72)*t+vec3(-12.68,-9.91,17.42))*t+vec3(7.58,9,-10.1))*t+vec3(-.87,-1.95,2.11))*t+vec3(.25,.31,.24),0.,1.);
}
vec3 PearlColorsP(float t) {
    return clamp((((vec3(-4.27,-4.24,-6.53)*t+vec3(10.54,10.44,14.12))*t+vec3(-7.15,-7.65,-9.04))*t+vec3(.94,1.44,1.66))*t+vec3(.89,.82,.76),0.,1.);
}
vec3 AvocadoColorsP(float t) {
    return clamp(((vec3(-1.86,1.04,0)*t+vec3(3.64,-2.34,.03))*t+vec3(-.83,2.33,.17))*t+vec3(.03,-.03,.01),0.,1.);
}
vec3 PlumColorsP(float t) {
    return clamp(((vec3(3.19,0,-2)*t+vec3(-4.92,.8,2.23))*t+vec3(2.7,.03,.17))*t+vec3(-.03,.04,0),0.,1.);
}
vec3 BeachColorsP(float t) {
    return clamp((((vec3(4.55,3.94,-4.62)*t+vec3(-6.3,-5.4,7.53))*t+vec3(1.93,1.2,-2.6))*t+vec3(.01,.8,.42))*t+vec3(.86,.5,.25),0.,1.);
}
vec3 RoseColorsP(float t) {
    return clamp((((vec3(4.77,4.06,0)*t+vec3(-9.87,-8.77,0))*t+vec3(5.17,4.81,-1.2))*t+vec3(.48,-.29,1.24))*t+vec3(.16,.32,.04),0.,1.);
}
vec3 CandyColorsP(float t) {
    return clamp((((vec3(2.68,0,0)*t+vec3(-3.42,-1.41,-2.02))*t+vec3(-.7,2.18,2.86))*t+vec3(1.75,-.12,-.32))*t+vec3(.38,.2,.35),0.,1.);
}
vec3 SolarColorsP(float t) {
    return clamp(((vec3(.88,-1.16,0)*t+vec3(-2.29,1.87,.16))*t+vec3(1.97,.08,-.03))*t+vec3(.44,.01,.01),0.,1.);
}
vec3 CMYKColorsP(float t) {
    return clamp((((vec3(-2.95,-8.11,-11.92)*t+vec3(2.91,8.04,21.96))*t+vec3(-2.1,1.3,-12.03))*t+vec3(2,-2.05,1.26))*t+vec3(.27,.82,.88),0.,1.);
}
vec3 SouthwestColorsP(float t) {
    return clamp((((vec3(-1.65,5.87,16.25)*t+vec3(1.99,-14.14,-30.02))*t+vec3(-1.92,9.87,17.36))*t+vec3(1.45,-1.34,-2.91))*t+vec3(.43,.33,.27),0.,1.);
}
vec3 DeepSeaColorsP(float t) {
    return clamp(((vec3(3.26,-1.4,0)*t+vec3(-3.94,2.65,-.74))*t+vec3(1.38,-.35,1.47))*t+vec3(.13,.02,.28),0.,1.);
}
vec3 StarryNightColorsP(float t) {
    return clamp(((vec3(-1.92,-1.39,-.77)*t+vec3(2.68,1.18,-.3))*t+vec3(.07,.84,1.22))*t+vec3(.11,.15,.18),0.,1.);
}
vec3 FallColorsP(float t) {
    return clamp((((vec3(0,-5,0)*t+vec3(-1.21,9.93,0))*t+vec3(1.97,-5.2,.44))*t+vec3(-.06,.67,-.54))*t+vec3(.29,.38,.38),0.,1.);
}
vec3 SunsetColorsP(float t) {
    return clamp((((vec3(4.93,0,-14.8)*t+vec3(-8.68,-1.09,36.43))*t+vec3(2.56,1.85,-27.3))*t+vec3(2.25,.22,6.77))*t+vec3(-.02,.03,-.1),0.,1.);
}
vec3 FruitPunchColorsP(float t) {
    return clamp((((vec3(0,3.2,-5.66)*t+vec3(.99,-4.45,7.54))*t+vec3(-.63,.35,-1.47))*t+vec3(-.39,.78,.09))*t+vec3(1.03,.49,-0),0.,1.);
}
vec3 ThermometerColorsP(float t) {
    return clamp((((vec3(3.75,5.74,0)*t+vec3(-9.01,-10.57,1.63))*t+vec3(4.86,2.57,-3.87))*t+vec3(.78,2.25,1.61))*t+vec3(.15,.1,.77),0.,1.);
}
vec3 IslandColorsP(float t) {
    return clamp((((vec3(7.05,0,0)*t+vec3(-18.07,0,1.42))*t+vec3(14.48,-1.38,-4.68))*t+vec3(-3.6,1.73,3.34))*t+vec3(.8,.4,.22),0.,1.);
}
vec3 WatermelonColorsP(float t) {
    return clamp((((vec3(0,0,-4.28)*t+vec3(-1.21,-1.96,4.77))*t+vec3(1.21,.6,-1.17))*t+vec3(.78,1.6,.9))*t+vec3(.13,.11,.09),0.,1.);
}
vec3 BrassTonesP(float t) {
    return clamp((((vec3(3.34,3.34,1.86)*t+vec3(-6.77,-6.91,-3.96))*t+vec3(1.16,1.73,1.2))*t+vec3(2.35,1.89,.93))*t+vec3(.09,.11,.03),0.,1.);
}
vec3 GreenPinkTonesP(float t) {
    return clamp((((vec3(11.14,6.54,9.33)*t+vec3(-29.11,-6.8,-25.28))*t+vec3(21.17,-4.06,18.64))*t+vec3(-3.07,4.27,-2.6))*t+vec3(.08,.19,.11),0.,1.);
}
vec3 BrownCyanTonesP(float t) {
    return clamp(((vec3(.97,-1.01,-1.33)*t+vec3(-3.47,-.32,.26))*t+vec3(2.52,1.78,1.77))*t+vec3(.3,.19,.07),0.,1.);
}
vec3 PigeonTonesP(float t) {
    return clamp((((vec3(-4.68,-3.16,-3.96)*t+vec3(8.84,6.49,8.27))*t+vec3(-4.7,-3.97,-5.14))*t+vec3(1.36,1.5,1.65))*t+vec3(.17,.15,.19),0.,1.);
}
vec3 CherryTonesP(float t) {
    return clamp(((vec3(2,0,0)*t+vec3(-4.02,1.15,1.12))*t+vec3(2.83,-.3,-.27))*t+vec3(.21,.19,.19),0.,1.);
}
vec3 RedBlueTonesP(float t) {
    return clamp((((vec3(5.28,5.07,0)*t+vec3(-9.7,-11.42,-2.29))*t+vec3(3.1,5.87,1.99))*t+vec3(.99,.66,.6))*t+vec3(.46,.15,.21),0.,1.);
}
vec3 CoffeeTonesP(float t) {
    return clamp((((vec3(0,0,-3.64)*t+vec3(0,.75,9.28))*t+vec3(-.31,-.86,-6.4))*t+vec3(.85,.79,1.53))*t+vec3(.43,.34,.24),0.,1.);
}
vec3 RustTonesP(float t) {
    return clamp((((vec3(3.72,0,0)*t+vec3(-7.43,0,0))*t+vec3(3.71,-.49,.16))*t+vec3(1.02,.97,-.32))*t+vec3(.01,-.02,.2),0.,1.);
}
vec3 FuchsiaTonesP(float t) {
    return clamp(((vec3(0,-1.11,0)*t+vec3(-.64,1.97,-.46))*t+vec3(1.55,-.04,1.38))*t+vec3(.07,.13,.07),0.,1.);
}
vec3 SiennaTonesP(float t) {
    return clamp(((vec3(.93,-.8,-1.35)*t+vec3(-2.32,.94,2.5))*t+vec3(1.86,.56,-.44))*t+vec3(.44,.18,.09),0.,1.);
}
vec3 GrayTonesP(float t) {
    return clamp((vec3(.39,.26,.11)*t+vec3(.47,.59,.68))*t+vec3(.1,.1,.11),0.,1.);
}
vec3 ValentineTonesP(float t) {
    return clamp((vec3(-.07,.45,.3)*t+vec3(.56,.34,.44))*t+vec3(.51,.09,.18),0.,1.);
}
vec3 GrayYellowTonesP(float t) {
    return clamp(((vec3(-1.57,-2.08,-3.17)*t+vec3(1.79,2.16,2.8))*t+vec3(.52,.42,.3))*t+vec3(.18,.22,.31),0.,1.);
}
vec3 DarkTerrainP(float t) {
    return clamp((((vec3(0,2.38,4.4)*t+vec3(4.05,1.4,-3.2))*t+vec3(-5.9,-5.75,-1.04))*t+vec3(2.92,3,.48))*t+vec3(-.04,.04,.45),0.,1.);
}
vec3 LightTerrainP(float t) {
    return clamp(((vec3(-1.35,-1.85,-1.86)*t+vec3(1.98,3.21,4.13))*t+vec3(-.29,-1.25,-2.23))*t+vec3(.54,.78,.86),0.,1.);
}
vec3 GreenBrownTerrainP(float t) {
    return clamp((((vec3(5.66,6.03,0)*t+vec3(-9.79,-7.78,5.38))*t+vec3(4.02,.54,-7.66))*t+vec3(1.13,2.27,3.32))*t+vec3(-0,-.02,-.05),0.,1.);
}
vec3 SandyTerrainP(float t) {
    return clamp((((vec3(3.48,4.39,0)*t+vec3(-7.65,-8.69,0))*t+vec3(3.48,3.53,-.49))*t+vec3(.27,.83,.43))*t+vec3(.68,.3,.2),0.,1.);
}
vec3 BrightBandsP(float t) {
    return clamp((((vec3(-16.63,6.8,-8.13)*t+vec3(28.73,-14.81,26.66))*t+vec3(-12.13,8.23,-27.59))*t+vec3(-.11,.17,9.55))*t+vec3(.98,.23,-.04),0.,1.);
}
vec3 DarkBandsP(float t) {
    return clamp((((vec3(17.15,-2.97,-5.17)*t+vec3(-33.07,6.52,5.24))*t+vec3(20.27,-3.03,2.06))*t+vec3(-3.87,-.41,-2.81))*t+vec3(.68,.81,1),0.,1.);
}
vec3 AquamarineP(float t) {
    return clamp(((vec3(2.66,1.62,1.39)*t+vec3(-3.77,-2.42,-1.78))*t+vec3(1.1,.82,.41))*t+vec3(.67,.73,.84),0.,1.);
}
vec3 PastelP(float t) {
    return clamp((((vec3(0,0,-6.39)*t+vec3(-1.54,-1.64,11.82))*t+vec3(.75,1.3,-4.98))*t+vec3(.41,.52,-.49))*t+vec3(.77,.5,.98),0.,1.);
}
vec3 BlueGreenYellowP(float t) {
    return clamp((vec3(1.4,-.82,-.63)*t+vec3(-.63,1.69,.52))*t+vec3(.15,.01,.42),0.,1.);
}
vec3 RainbowP(float t) {
    return clamp((((vec3(0,1.63,-8.06)*t+vec3(-4.42,-4.94,20.35))*t+vec3(7.09,1.89,-17.07))*t+vec3(-2.3,1.49,4.45))*t+vec3(.46,.04,.45),0.,1.);
}
vec3 DarkRainbowP(float t) {
    return clamp((((vec3(2.62,6.12,-2.34)*t+vec3(-9.73,-15.58,2.43))*t+vec3(9.43,10.93,1.04))*t+vec3(-2.02,-1.77,-1.62))*t+vec3(.33,.41,.65),0.,1.);
}
vec3 TemperatureMapP(float t) {
    return clamp((((vec3(5.84,0,11.29)*t+vec3(-11.36,-2.4,-19.02))*t+vec3(4.9,.36,7.83))*t+vec3(1.31,1.84,-.76))*t+vec3(.16,.28,.95),0.,1.);
}
vec3 LightTemperatureMapP(float t) {
    return clamp((((vec3(4.11,0,5.78)*t+vec3(-8.72,0,-10.5))*t+vec3(4.23,-2.77,4.36))*t+vec3(1.04,2.85,-.37))*t+vec3(.18,.27,.95),0.,1.);
}
