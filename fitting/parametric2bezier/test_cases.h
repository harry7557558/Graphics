// Test cases for Bezier curve fitting experiment
// A collection of parametric equations

// Parametric curve class
template<typename Fun>
class ParametricCurve {
public:
	double t0, t1;  // parameter interval
	const Fun p;  // equation
	ParametricCurve(Fun p) :t0(NAN), t1(NAN), p(p) { }
	ParametricCurve(Fun p, double t0, double t1) :t0(t0), t1(t1), p(p) { }
	//ParametricCurve() :t0(NAN), t1(NAN) {}  // not recommended
};
typedef ParametricCurve<vec2(*)(double)> ParametricCurveP;
typedef ParametricCurve<std::function<vec2(double)>> ParametricCurveL;

// functions for one-linerization
template<typename Fun> double Sum(Fun f, int n0, int n1, int step = 1) {
	double r = 0;
	for (int n = n0; n <= n1; n += step) r += f(n);
	return r;
};
double fract(double x) { return x - floor(x); }
double hashf(double x, double y) { return fmod(sin(12.9898*x + 78.233*y + 1.) * 43758.5453, 1.); };

// count the number of function calls
static uint32_t Parametric_callCount = 0;
#define _return Parametric_callCount++; return


// Test equations - some from Wikipedia

const int CSN = 128;  // number of test functions
const int CS0 = 0, CS1 = 128;  // only test functions in this range when debugging

// 0-43: General tests; 44-91: Ill-conditioned functions; 92-127: Performance tests;
// Test case ID: __LINE__ - 40

const ParametricCurveL Cs[CSN] = {
ParametricCurveL([](double t) { _return vec2(sin(t), cos(t) + .5*sin(t)); }, -PI, PI),
ParametricCurveL([](double t) { _return vec2(sin(t), 0.5*sin(2.*t)); }, -0.1, 2.*PI - 0.1),
ParametricCurveL([](double t) { _return vec2(sin(t), cos(t))*cos(2 * t); }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(sin(t),cos(t))*cos(3.*t); }, 0, PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*sin(5.*t); }, 0, PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*sin(6.*t); }, 0, 2.*PI),
ParametricCurveL([](double x) { _return vec2(x, exp(-x * x)); }, -1., 2.),
ParametricCurveL([](double t) { _return vec2(sinh(t), cosh(t) - 1.); }, -1., 1.4),
ParametricCurveL([](double x) { _return vec2(x, sin(5.*x)); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, x == 0. ? 1. : sin(2.*PI*x) / (2.*PI*x)); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, x*x*x - x); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(.5*x, 0.04*(x*x*x*x + 2.*x*x*x - 6.*x*x - x + 1.)); }, -4., 4.),
ParametricCurveL([](double t) { _return vec2(cos(t) + .5*cos(2.*t), sin(t) + .5*sin(2.*t)); }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(2.*t), sin(2.*t))*sin(t); }, 0, PI),
ParametricCurveL([](double t) { _return vec2(cos(2.*t), sin(2.*t))*sin(t); }, -1, 2 * PI - 1),
ParametricCurveL([](double t) { _return vec2(cos(t) + cos(2.*t), sin(t) + sin(2.*t))*.5; }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t) + .5*cos(2.*t), sin(t) - .5*sin(2.*t)); }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t) + .5*cos(2.*t), sin(t) - .5*sin(2.*t)); }, -.5*PI, 1.5*PI),
ParametricCurveL([](double t) { _return vec2(cos(3.*t), sin(2.*t)); }, -PI + 1., PI + 1.),
ParametricCurveL([](double t) { _return vec2(cos(5.*t + PI / 4.), sin(4.*t)); }, 1., 2.*PI + 1.),
ParametricCurveL([](double a) { _return vec2(cos(a),sin(a)) * .8*(pow(cos(6.*a),2.) + .5); }, 0., 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(2.*PI*a),sin(2.*PI*a)) * pow(abs(1.2*a),3.8) + vec2(-.7,0.); }, -1., 1.),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.08*a; }, 0, 6.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.08*exp(0.25*a); }, -PI, 4.*PI),
ParametricCurveL([](double t) { _return vec2(.1*t + .3*cos(t), sin(t)); }, -13., 14.),
ParametricCurveL([](double t) { _return 0.4*vec2(cos(1.5*t), sin(1.5*t)) + vec2(cos(t), -sin(t)); }, 0, 4.*PI),
ParametricCurveL([](double a) { _return(sin(a) - cos(2.*a) + sin(3.*a))*vec2(cos(a), sin(a)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return(sin(a) - cos(2.*a) + sin(3.*a))*vec2(cos(a), sin(a)); }, 0, 3.*PI),
ParametricCurveL([](double a) { _return 0.5*(cos(a) + sin(a)*sin(a) + 1.)*vec2(cos(a), sin(a)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return 0.5*(cos(a) + sin(a)*sin(a) + 1.)*vec2(cos(a), sin(a)); }, -1, 2.*PI - 1.),
ParametricCurveL([](double x) { _return vec2(x, exp(sin(x)) - 1.5); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, exp(sin(PI*x)) - 1.5); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, sin(sin(5.*x))); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, sin(10.*x*x)); }, -2, 2),
ParametricCurveL([](double t) { _return vec2(cos(t) + .1*cos(10.*t), sin(t) + .1*sin(10.*t)); }, 0, 2.*PI),
ParametricCurveL([](double x) { _return vec2(x, x*x - cos(10.*x) - 1.)*.5; }, -1.8, 2.),
ParametricCurveL([](double t) { _return vec2(cos(4.*t) + sin(t), cos(3.*t) + .7*sin(5.*t))*.8; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(-.7*cos(5.*t) + sin(t), cos(3.*t) + .7*sin(5.*t))*.8; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(1.5*cos(t) + cos(1.5*t), 1.5*sin(t) - sin(1.5*t))*.5; }, 0., 4.*PI),
ParametricCurveL([](double t) { _return vec2(2.1*cos(t) + cos(2.1*t), 2.1*sin(t) - sin(2.1*t))*.5; }, 0., 10.*PI),
ParametricCurveL([](double t) { _return vec2(-1.2*cos(t) + cos(1.2*t), -1.2*sin(t) + sin(1.2*t))*.5; }, 0., 10.*PI),
ParametricCurveL([](double t) { _return vec2(-1.9*cos(t) + cos(1.9*t), -1.9*sin(t) + sin(1.9*t))*.5; }, 0., 20.*PI),
ParametricCurveL([](double t) { _return vec2(-sin(t) - .3*cos(t), .1*sin(t) - .5*cos(t))*sin(5.*t) + vec2(0., 1. - .5*pow(sin(5.*t) - 1., 2.)); }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(sin(t) + .2*cos(30.*t)*sin(t), -.4*cos(t) - .1*cos(30.*t)*cos(t) + .2*sin(30.*t)); }, 0, 2.*PI),
ParametricCurveL([](double x) { _return vec2(x, log(x + 1)); }, -0.99, 2.),
ParametricCurveL([](double x) { _return vec2(0.5*x - 1., 0.1*tgamma(x) - 1.); }, 0.05, 5),
ParametricCurveL([](double x) { _return vec2(x, sqrt(1. - x * x)); }, -1., 1.),
ParametricCurveL([](double x) { _return vec2(x, asin(x)*(2. / PI)); }, -1., 1.),
ParametricCurveL([](double x) { _return vec2(x, abs(x - 0.123) - 1.); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, 0.1*tan(x)); }, -.499*PI, .499*PI),
ParametricCurveL([](double x) { _return vec2(x, sin(10.*sqrt(x + 2.))); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, .5*acos(cos(5.*x)) - .25*PI); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, x - floor(x)); }, -2, 2),
ParametricCurveL([](double t) { _return vec2(.1*floor(10.*t + 1.), sin(2.*PI*t)*(10.*t - floor(10.*t))); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, sin(x - 1) / log(x + PI)) * 0.5; }, -PI, PI),
ParametricCurveL([](double x) { _return vec2(x, sqrt(x + 1.) - 1.); }, -1., 1.),
ParametricCurveL([](double x) { _return vec2(x, cbrt(x)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, cbrt(abs(x)) - 1.); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, cbrt(abs(x) - 1.)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, sqrt(abs(x) - 1.)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, cbrt(abs(abs(x) - 1.) - .5)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, pow(abs(cbrt(abs(abs(x) - 1.) - .5)), .2) * (abs(abs(x) - 1.) > .5 ? 1. : -1.)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, log(exp(abs(x))*sin(x)) - 10.)*.1; }, -20., 20.),
ParametricCurveL([](double x) { _return vec2(x, log(abs(exp(abs(x))*sin(x))) - 10.)*.1; }, -20., 20.),
ParametricCurveL([](double x) { _return vec2(x, log(tgamma(x)))*.5; }, -4., 4.),
ParametricCurveL([](double x) { _return vec2(x, log(abs(tgamma(x))))*.5; }, -4., 4.),
ParametricCurveL([](double x) { _return vec2(x, (x > 0. ? asin(sqrt(x)) / sqrt(x) : asinh(sqrt(-x)) / sqrt(-x)) - 1.); }, -2., 1.),
ParametricCurveL([](double x) { _return vec2(x, (x > 0. ? asin(sqrt(x)) / sqrt(x) : asinh(sqrt(-x)) / sqrt(-x)) - 1.); }, -2., 2.),
ParametricCurveL([](double t) { _return vec2(tan(t), 1. / tan(t)); }, -.25*PI, .75*PI),
ParametricCurveL([](double t) { _return vec2(tan(t), 1. / tan(t)); }, 0., PI),
ParametricCurveL([](double t) { _return vec2(cos(t) / sin(t), 1.)*pow(tan(t / 2.), 1.05) * 2. - vec2(0,1); }, 0., .5*PI),
ParametricCurveL([](double t) { _return vec2(cos(t) / sin(t), sin(t))*pow(tan(t / 2.), 1.02) * 2. - vec2(0,1); }, 0., .5*PI),
ParametricCurveL([](double x) { _return vec2(x, max(exp(.2*x), 2.*sin(10.*x)) - 1.); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, max(sin(10.5*x), cos(2.*x))); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, clamp(tan(10.*x), -1., 1.)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, clamp(.2 / sin(10.*x), -1., 1.)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, fract(10. / (x*x + 1.))); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, fract(pow(x,4.) - .1)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, fract(sin(5.*x) / x)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, fract(10.01*tanh(2.*x))); }, -2., 2.),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*.08*floor(a); }, 0., 6.*PI),
ParametricCurveL([](double a) { _return vec2(cos(2.*PI*a), sin(2.*PI*a))*(.5*fract(20.*a) + .8); }, 0., 1.),
ParametricCurveL([](double t) { _return vec2(cos(PI / 6.*floor(t)),sin(PI / 6.*floor(t))) + .2*vec2(cos(2.*PI*t),sin(2.*PI*t)); }, 0., 12.),
ParametricCurveL([](double t) { _return vec2(cos(PI / 6.*floor(t)),sin(PI / 6.*floor(t))) + .2*vec2(cos(2.*PI*t),sin(2.*PI*t)) + .08*vec2(sin(11.*t),cos(10.*t)); }, 0., 12.),
ParametricCurveL([](double t) { _return vec2(cos(2.*PI*t),sin(2.*PI*t))*.1*floor(t) + vec2(.5 - .05*floor(t)); }, 0., 12.),
ParametricCurveL([](double t) { _return vec2(([&]() { vec2 p = vec2(cos(2.*PI*t),sin(2.*PI*t))*.1*floor(t) - vec2(.5 - .05*floor(t)); return p + .05*vec2(sin(10.*p.x), sin(10.*p.y)); })()); }, 0., 12.),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { return sin(PI*n*x) / n; }, 1, 21, 2)); }, -2.5, 2.5),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { return sin(2.*PI*n*x) / (PI*n); }, 1, 20)); }, -2.5, 2.5),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { return sin(2.*PI*n*x) / (n*n); }, 1, 5)); }, -2.5, 2.5),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { return (sin(PI*n*x) - n * cos(PI*n*x)) / (n*(n*n + 1)); }, 1, 1001, 2)); }, -2.5, 2.5),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { return exp(-n)*sin(exp(n)*(x + n)); }, 1, 5)); }, -2.5, 2.5),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { double u = exp2(n)*x, v = u - floor(u); return exp2(-n)*mix(hashf(floor(u),-n), hashf(ceil(u),-n), v*v*(3. - 2.*v)); }, 1, 6)); }, -2.5, 2.5),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.08*exp(0.25*a)*(1. - .2*exp(sin(10.*a))); }, -PI, 4.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.04*exp(0.25*a)*(sin(10.*a) + 1.2); }, -PI, 4.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.04*exp(0.25*a)*(pow(sin(10.*a), 10.) + 1.); }, -PI, 4.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.04*exp(0.25*a)*(-exp(10.*(sin(20.*a) - 1)) + 1.8); }, -PI, 4.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.06*exp(0.25*a)*(pow(0.6*asin(sin(10.*a)) - .05, 8.) + 0.8); }, -PI, 4.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.08*exp(0.25*a)*(0.1*(asin(sin(10.*a)) + asin(sin(30.*a))) + 1.); }, -PI, 4.*PI),
ParametricCurveL([](double t) { _return vec2(.04041 + .6156*cos(t) - .3412*sin(t) + .1344*cos(2.*t) - .1224*sin(2.*t) + .08335*cos(3.*t) + .2634*sin(3.*t) - .07623*cos(4.*t) - .09188*sin(4.*t) + .01339*cos(5.*t) - .01866*sin(5.*t) + .1631*cos(6.*t) + .006984*sin(6.*t) + .02867*cos(7.*t) - .01512*sin(7.*t) + .00989*cos(8.*t) + .02405*sin(8.*t) + .002186*cos(9.*t), +.04205 + .2141*cos(t) + .4436*sin(t) + .1148*cos(2.*t) - .146*sin(2.*t) - .09506*cos(3.*t) - .06217*sin(3.*t) - .0758*cos(4.*t) - .02987*sin(4.*t) + .2293*cos(5.*t) + .1629*sin(5.*t) + .005689*cos(6.*t) + .07154*sin(6.*t) - .02175*cos(7.*t) + .1169*sin(7.*t) - .01123*cos(8.*t) + .02682*sin(8.*t) - .01068*cos(9.*t)); }, 0., 2.*PI),
ParametricCurveL([](double a) { _return 0.3*(exp(sin(a)) - 2.*cos(4.*a) + sin((2.*a - PI) / 24.))*vec2(cos(a), sin(a)); }, -8.*PI, 8.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t) - pow(cos(40.*t), 3.), sin(40.*t) - pow(sin(t), 4.) + .5)*.8; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(60.*t) - 1.6*pow(cos(t), 3.), sin(60.*t) - pow(sin(t), 3.))*.6; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t) - cos(t)*sin(60.*t), 2.*sin(t) - sin(60.*t))*.5; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(80.*t) - 1.4*cos(t)*sin(2.*t), 2.*sin(t) - sin(80.*t))*.5; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(sin(3.*t) + sin(t),cos(3.*t))*.8 + vec2(sin(160.*t),cos(160.*t))*.4; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(3.*t),2.*cos(t))*.6 + vec2(cos(100.*t),sin(100.*t))*sin(t)*.6; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(1.8*cos(t), 0.6*cos(5.*t)) + vec2(cos(100.*t),sin(100.*t))*pow(abs(sin(t)),0.6)*0.6; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t),0.618*cos(2.*t)) + 0.618*vec2(cos(60.*t),sin(60.*t))*cos(2.*t); }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(1.12*t),sin(t)) + .5*vec2(cos(60.*t),-pow(sin(60.*t),2.)); }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t),sin(t)) + vec2(cos(60.*t),sin(60.*t))*.5; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t) + .2*sin(20.*t), sin(t) + .2*cos(20.*t)) * .02*t; }, 0., 20.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*.35*(.1*(t + 1.) + pow(sin(10.*t),2.) + .1*pow(sin(100.*t),2.)); }, 0., 10.*PI),
ParametricCurveL([](double x) { _return vec2(x, sin(x) + sin(10.*x) + sin(100.*x))*.4; }, -5., 5.),
ParametricCurveL([](double x) { _return vec2(x, .5*(sin(40.*x) + sin(45.*x))); }, -2.5, 2.5),
ParametricCurveL([](double a) { _return vec2(cos(a),sin(a))*(2.0*Sum([&](int n) { return exp2(-n)*sin(exp2(n)*a); }, 1, 10)); }, 0., 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a) + .3)*(2.0*Sum([&](int n) { return exp2(-n)*cos(exp2(n)*a); }, 1, 16)); }, 0., 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a),-sin(a))*1.2*(sin(a) + Sum([&](int n) { return exp2(-n)*cos(exp2(n)*a); }, 1, 10)) + vec2(0.,.6); }, 0., 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(0.1*exp(.25*a)) + vec2(.08*exp(.2*a)*Sum([&](int n) { return exp2(-n)*pow(cos(exp2(n)*a),2.); }, 1, 10)); }, 0., 12.),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.8 + .4*Sum([&](int n) { return sin(5.*n*a) / n; }, 1, 11, 2)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.8 + .3*Sum([&](int n) { return sin(5.*n*a) / n; }, 1, 10, 1)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.8 + .5*Sum([&](int n) { return sin(5.*n*n*a) / (n*n); }, 1, 3, 1)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.8 + .4*Sum([&](int n) { return (n & 2 ? -1. : 1.)*sin(5.*n*a) / (n*n); }, 1, 21, 2)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.8 + 3.*Sum([&](int n) { return (n & 2 ? -1. : 1.)*cos(6.*n*a) / (n*n); }, 3, 21, 2)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.9 + .15*Sum([&](int n) { return (cos(8.*n*(a + .1)) + cos(8.*n*(a - .1))) / n; }, 1, 11, 2)); }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*.005*exp(.25*t)*(1. + Sum([&](int n) { return pow(cos(5.*n*t), 2.) / n; }, 1, 5, 1)); }, 0, 6.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*.006*exp(.25*t)*(1. + Sum([&](int n) { return pow(sin(5.*n*t), 2.) / n; }, 1, 5, 1)); }, 0, 6.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*.007*exp(.25*t)*(1. + Sum([&](int n) { return pow(sin(5.*n*n*t), 2.) / (n*n); }, 1, 5, 1)); }, 0, 6.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*.009*exp(.25*t)*(1. + Sum([&](int n) { return pow(sin(5.*exp2(n)*t), 2.) / exp2(n); }, 1, 5, 1)); }, 0, 6.*PI),
};


