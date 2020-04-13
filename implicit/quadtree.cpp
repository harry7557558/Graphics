// Graphing Implicit Surface


/* ==================== User Instructions ====================

 *  Move View:                  drag background
 *  Zoom:                       mouse scroll, hold shift to lock center
 */



 // ========================================= Win32 Standard =========================================

 // some compressed copy-paste code
#pragma region Windows

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>

#define WIN_NAME "Spline Tester"
#define WinW_Default 600
#define WinH_Default 400
#define MIN_WinW 400
#define MIN_WinH 300

// implement the following functions:
void render();
void WindowCreate(int _W, int _H);
void WindowResize(int _oldW, int _oldH, int _W, int _H);
void WindowClose();
void MouseMove(int _X, int _Y);
void MouseWheel(int _DELTA);
void MouseDownL(int _X, int _Y);
void MouseUpL(int _X, int _Y);
void MouseDownR(int _X, int _Y);
void MouseUpR(int _X, int _Y);
void KeyDown(WPARAM _KEY);
void KeyUp(WPARAM _KEY);

HWND _HWND; int _WIN_W, _WIN_H;
HBITMAP _HIMG; COLORREF *_WINIMG;
#define Canvas(x,y) _WINIMG[(y)*_WIN_W+(x)]

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { HDC hdc = GetDC(_HWND), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); } break;
	switch (message) {
	case WM_CREATE: { RECT Client; GetClientRect(hWnd, &Client); _WIN_W = Client.right, _WIN_H = Client.bottom; WindowCreate(_WIN_W, _WIN_H); break; }
	case WM_CLOSE: { DestroyWindow(hWnd); WindowClose(); return 0; } case WM_DESTROY: { PostQuitMessage(0); return 0; }
	case WM_MOVE:; case WM_SIZE: {
		RECT Client; GetClientRect(hWnd, &Client); WindowResize(_WIN_W, _WIN_H, Client.right, Client.bottom); _WIN_W = Client.right, _WIN_H = Client.bottom;
		BITMAPINFO bmi; bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32; bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0; bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0;
		if (_HIMG != NULL) DeleteObject(_HIMG); HDC hdc = GetDC(hWnd); _HIMG = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&_WINIMG, NULL, 0); DeleteDC(hdc); _RDBK }
	case WM_GETMINMAXINFO: { LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam; lpMMI->ptMinTrackSize.x = MIN_WinW, lpMMI->ptMinTrackSize.y = MIN_WinH; break; }
	case WM_PAINT: { PAINTSTRUCT ps; HDC hdc = BeginPaint(hWnd, &ps), HMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HMem, _HIMG); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HMem, 0, 0, SRCCOPY); SelectObject(HMem, hbmOld); EndPaint(hWnd, &ps); DeleteDC(HMem), DeleteDC(hdc); break; }
#define _USER_FUNC_PARAMS GET_X_LPARAM(lParam), _WIN_H - 1 - GET_Y_LPARAM(lParam)
	case WM_MOUSEMOVE: { MouseMove(_USER_FUNC_PARAMS); _RDBK }
	case WM_MOUSEWHEEL: { MouseWheel(GET_WHEEL_DELTA_WPARAM(wParam)); _RDBK }
	case WM_LBUTTONDOWN: { SetCapture(hWnd); MouseDownL(_USER_FUNC_PARAMS); _RDBK }
	case WM_LBUTTONUP: { ReleaseCapture(); MouseUpL(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONDOWN: { MouseDownR(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONUP: { MouseUpR(_USER_FUNC_PARAMS); _RDBK }
	case WM_SYSKEYDOWN:; case WM_KEYDOWN: { KeyDown(wParam); _RDBK }
	case WM_SYSKEYUP:; case WM_KEYUP: { KeyUp(wParam); _RDBK }
	} return DefWindowProc(hWnd, message, wParam, lParam);
}
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME); if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

// debug
#define dbgprint(format, ...) { wchar_t buf[0x4FFF]; swprintf(buf, 0x4FFF, _T(format), ##__VA_ARGS__); OutputDebugStringW(buf); }

#pragma endregion


// ================================== GLSL Style Classes/Functions ==================================

#pragma region GLSL Classes/Functions

#include <cmath>
#pragma warning(disable: 4244 4305)

#define PI 3.1415926535897932384626
#define mix(x,y,a) ((x)*(1.0-(a))+(y)*(a))
#define clamp(x,a,b) ((x)<(a)?(a):(x)>(b)?(b):(x))

class vec2 {
public:
	double x, y;
	vec2() {}
	vec2(double a) :x(a), y(a) {}
	vec2(double x, double y) :x(x), y(y) {}
	vec2 operator - () const { return vec2(-x, -y); }
	vec2 operator + (const vec2 &v) const { return vec2(x + v.x, y + v.y); }
	vec2 operator - (const vec2 &v) const { return vec2(x - v.x, y - v.y); }
	vec2 operator * (const vec2 &v) const { return vec2(x * v.x, y * v.y); }	// not standard but useful
	vec2 operator * (const double &a) const { return vec2(x*a, y*a); }
	double sqr() const { return x * x + y * y; } 	// not standard
	friend double length(const vec2 &v) { return sqrt(v.x*v.x + v.y*v.y); }
	friend vec2 normalize(const vec2 &v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y)); }
	friend double dot(const vec2 &u, const vec2 &v) { return u.x*v.x + u.y*v.y; }
	friend double det(const vec2 &u, const vec2 &v) { return u.x*v.y - u.y*v.x; } 	// not standard
#if 0
	vec2 operator == (const vec2 &v) const { return x == v.x && y == v.y; }
	vec2 operator != (const vec2 &v) const { return x != v.x || y != v.y; }
	void operator += (const vec2 &v) { x += v.x, y += v.y; }
	void operator -= (const vec2 &v) { x -= v.x, y -= v.y; }
	void operator *= (const vec2 &v) { x *= v.x, y *= v.y; }
	vec2 operator / (const vec2 &v) const { return vec2(x / v.x, y / v.y); }
	void operator /= (const vec2 &v) { x /= v.x, y /= v.y; }
	friend vec2 operator * (const double &a, const vec2 &v) { return vec2(a*v.x, a*v.y); }
	void operator *= (const double &a) { x *= a, y *= a; }
	vec2 operator / (const double &a) const { return vec2(x / a, y / a); }
	void operator /= (const double &a) { x /= a, y /= a; }
#endif
	vec2 rot() const { return vec2(-y, x); }   // not standard
};


#pragma endregion


// ======================================== Data / Parameters ========================================

#include <stdio.h>
#pragma warning(disable: 4996)

#pragma region Global Variables

// window parameters
char text[64];	// window title
vec2 Center = vec2(0, 0);	// origin in screen coordinate
double Unit = 100.0;		// screen unit to object unit
#define fromInt(p) (((p) - Center) * (1.0 / Unit))
#define fromFloat(p) ((p) * Unit + Center)

// user parameters
vec2 Cursor = vec2(0, 0);
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
NTime::time_point _Global_Timer = NTime::now();
#define iTime std::chrono::duration<double>(NTime::now()-_Global_Timer).count()

#pragma endregion



namespace slow_function_1 {
	double sdLine(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a, ba = b - a; return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0., 1.)); }
	double u(vec2 p, double t) { return min(min(p.x - 0.16, sdLine(p, 0, vec2(1.22, 0))), min(sdLine(vec2(p.x, abs(p.y)), vec2(0.45, 0), vec2(0.45 + 0.42*cos(t), 0.42*sin(t))), sdLine(vec2(p.x, abs(p.y)), vec2(0.8, 0), vec2(0.8 + 0.35*cos(t), 0.35*sin(t))))); }
	double fun(vec2 p, double t) { double a = abs(fmod(atan2(p.y, p.x) + .5*t, PI / 3.)) - PI / 6.; return u(vec2(cos(a), sin(a))*length(p)*(1.0 + 0.3*cos(t)), 0.9 - 0.3*cos(t)) - 0.05; }
}

namespace slow_function_2 {
	double smin(double a, double b, double k) { return abs(a - b) > k ? min(a, b) : .5*(-0.5 / k * (a - b)*(a - b) + a + b - .5*k); }
	double u(vec2 p) { return length(vec2(1.06*(p.x + 0.058 * cos(14.4*p.y) - (0.74 + 0.25)), p.y)) - 0.25; }
	double Fmod(double x, double y) { return (x - y * floor(x / y)); }
	double Vr(vec2 p, double td) { double a = Fmod(atan2(p.y, p.x) - td, 2.*PI / 8.) - PI / 8.; return smin(u(vec2(cos(a), sin(a))*length(p)), length(p) - 0.74, 0.156); }
	double Vs(vec2 p, double t) { return min(Vr((p - vec2(-0.82, -0.35)) *(1. / 0.85) + vec2(0, 0.095*sin(3.*t)), -0.5*t), Vr((p - vec2(0.81, 0.54)) *(1. / 0.58) + vec2(0, 0.044*sin(4.*(t + 2.56))), 0.625*(t + 2.56))); }
	double ss(double x) { return x * x*x*((6.*x - 15.)*x + 10.); }
	double whfs(double x) { return x < 0.5 ? 0. : x < 1.5 ? ss(x - 0.5) : x < 2. ? 1. : 1. - ss(x - 2.); }
	double W(double x, double y, double t) { return y - (0.14*exp(sin(6.77*x + 5. * t)) + 4.*whfs(Fmod(0.15*t, 3.)) - 2.5); }
	double fun(vec2 p, double t) { return max(max(abs(p.x) - 3., abs(p.y) - 2.), smin(Vs(p, t), W(p.x, p.y, t), 0.1)); }
}

namespace slow_function_3 {
	double map(double x, double y, double z) { return pow(x*x + 2.25*y*y + z * z - 1, 3) - (x*x + 0.1125*y*y)*z*z*z; }
	const double z0 = -0.9, z1 = 1.2, Dz = 8., x0 = -1.1, x1 = 1.1, Dx = 10., y0 = -0.75, y1 = 0.75, Dy = 12.;
	double imp(vec2 p, double t) {
		double rx = .5*sin(t), rz = t;
		double I[2][2] = { 1. / cos(rz), 0, tan(rx)*tan(rz), 1. / cos(rx) };
		double J[2][2] = { -1. / sin(rz), 0, -tan(rx) / tan(rz), 1. / cos(rx) };
		double K[2][2] = { -sin(rz), -cos(rz) / sin(rx), cos(rz), -sin(rz) / sin(rx) };
		vec2 A1 = vec2(-sin(rz), -cos(rz)*sin(rx)), A2 = vec2(cos(rz), -sin(rz)*sin(rx)), A3 = vec2(0, cos(rx));
		double r = 1e8;
		for (double n = 0.; n < Dx; n++) {
			double cx = x0 + n / Dx * (x1 - x0);
			vec2 yz = p - A1*cx;
			yz = vec2(I[0][0] * yz.x + I[0][1] * yz.y, I[1][0] * yz.x + I[1][1] * yz.y);
			r *= tanh(map(cx, yz.x, yz.y));
		}
		for (double n = 0.; n < Dy; n++) {
			double cy = y0 + n / Dy * (y1 - y0);
			vec2 xz = p - A2*cy;
			xz = vec2(J[0][0] * xz.x + J[0][1] * xz.y, J[1][0] * xz.x + J[1][1] * xz.y);
			r *= tanh(map(xz.x, cy, xz.y));
		}
		for (double n = 0.; n < Dz; n++) {
			double cz = z0 + n / Dz * (z1 - z0);
			vec2 xy = p - A3*cz;
			xy = vec2(K[0][0] * xy.x + K[0][1] * xy.y, K[1][0] * xy.x + K[1][1] * xy.y);
			r *= tanh(map(xy.x, xy.y, cz));
		}
		return r;
	}
}


int evals;
double fun(double x, double y) {
	evals++;
	//return x * x + y * y - 1;
	//return hypot(x, y) - 1;
	//return x * x*x*(x - 2) + y * y*y*(y - 2) + x;
	//return x * x*x*x + y * y*y*y + x * y;
	//return y - sin(5*x);
	//return abs(x) - abs(y);
	//return abs(x*y) - 1;
	//return sin(x) + sin(y);
	//return hypot(x, y)*cos(hypot(x, y)) - x;
	//return 1 / x - x / y;
	//return sin(x + y * y) + y * y*exp(x + y) + 5 * cos(x*x + y);
	return slow_function_1::fun(vec2(x, y), 1.7);
	//return slow_function_2::fun(vec2(x, y), 6.4);
	//return slow_function_3::imp(vec2(x, y), iTime);
	//return abs(slow_function_1::fun(vec2(x, y), iTime)) - 0.01;
}


// ============================================ Rendering ============================================

auto t0 = NTime::now();

double V[1920][1080];

#define WHITE 0xFFFFFF
#define RED 0xFF0000
#define YELLOW 0xFFFF00
#define BLUE 0x0000FF

int lines;
auto drawLine = [](vec2 p, vec2 q, COLORREF col) {
	lines++;
	vec2 d = q - p;
	double slope = d.y / d.x;
	if (abs(slope) <= 1.0) {
		if (p.x > q.x) std::swap(p, q);
		int x0 = max(0, int(p.x)), x1 = min(_WIN_W - 1, int(q.x)), y;
		double yf = slope * x0 + (p.y - slope * p.x);
		for (int x = x0; x <= x1; x++) {
			y = (int)yf;
			if (y >= 0 && y < _WIN_H) Canvas(x, y) = col;
			yf += slope;
		}
	}
	else {
		slope = d.x / d.y;
		if (p.y > q.y) std::swap(p, q);
		int y0 = max(0, int(p.y)), y1 = min(_WIN_H - 1, int(q.y)), x;
		double xf = slope * y0 + (p.x - slope * p.y);
		for (int y = y0; y <= y1; y++) {
			x = (int)xf;
			if (x >= 0 && x < _WIN_W) Canvas(x, y) = col;
			xf += slope;
		}
	}
};
#define drawLineF(p,q,col) drawLine(fromFloat(p),fromFloat(q),col)
auto drawBox = [](double x0, double x1, double y0, double y1, COLORREF col) {
	drawLineF(vec2(x0, y0), vec2(x1, y0), col);
	drawLineF(vec2(x0, y1), vec2(x1, y1), col);
	drawLineF(vec2(x0, y0), vec2(x0, y1), col);
	drawLineF(vec2(x1, y0), vec2(x1, y1), col);
};

int SEARCH_DPS;
int PLOT_DPS;

auto Intp1 = [](double a, double b)->double {
	return a / (a - b);
};
auto Intp2 = [](double a, double b, double c)->bool {
	double c2 = 0.5*(a + c) - b, c1 = -1.5*a + 2.0*b - 0.5*c, c0 = a;
	double t = -c1 / (2.0 * c2);
	if (t < 0 || t > 1) return false;
	return a * ((c2*t + c1)*t + c0) < 0;
};
#define NE(a,b) (abs((b)-(a))>1e-12)
bool quadTree(double x0, double y0, double dx, double dy, double v00, double v01, double v10, double v11, int dps, bool testSign = true) {
	double x1 = x0 + dx, y1 = y0 + dy;
	/*if (NE(fun(x0, y0), v00) || NE(fun(x0, y1), v01) || NE(fun(x1, y0), v10) || NE(fun(x1, y1), v11)) {
		dbgprint("E");
	}*/
	bool s00 = signbit(v00), s01 = signbit(v01), s10 = signbit(v10), s11 = signbit(v11);
	int s = s00 + s01 + s10 + s11;
	if (testSign && s % 4 == 0) return false;

	//drawBox(x0, x0 + dx, y0, y0 + dy, BLUE);
	if (dps >= PLOT_DPS) {
		vec2 p00(x0, y0), p01(x0, y1), p10(x1, y0), p11(x1, y1), xd(dx, 0), yd(0, dy);
		if (s == 3) s00 ^= 1, s01 ^= 1, s10 ^= 1, s11 ^= 1, s = 1;
		if (s == 1) {
			if (s00) drawLineF(p00 + xd * Intp1(v00, v10), p00 + yd * Intp1(v00, v01), YELLOW);
			if (s01) drawLineF(p01 + xd * Intp1(v01, v11), p01 - yd * Intp1(v01, v00), YELLOW);
			if (s10) drawLineF(p10 - xd * Intp1(v10, v00), p10 + yd * Intp1(v10, v11), YELLOW);
			if (s11) drawLineF(p11 - xd * Intp1(v11, v01), p11 - yd * Intp1(v11, v10), YELLOW);
			return true;
		}
		if (s == 2) {
			if ((s00&&s01) || (s10&&s11)) drawLineF(p00 + xd * Intp1(v00, v10), p01 + xd * Intp1(v01, v11), YELLOW);
			else if ((s00&&s10) || (s01&&s11)) drawLineF(p00 + yd * Intp1(v00, v01), p10 + yd * Intp1(v10, v11), YELLOW);
			else if (dps < PLOT_DPS * 2) {
				//drawBox(p00.x, p10.x, p00.y, p01.y, RED);
				dx *= 0.5, dy *= 0.5;
				double xc = x0 + dx, yc = y0 + dy;
				double vc0 = fun(xc, y0), vc1 = fun(xc, y1), v0c = fun(x0, yc), v1c = fun(x1, yc), vcc = fun(xc, yc);
				quadTree(x0, y0, dx, dy, v00, v0c, vc0, vcc, dps + 1);
				quadTree(xc, y0, dx, dy, vc0, vcc, v10, v1c, dps + 1);
				quadTree(x0, yc, dx, dy, v0c, v01, vcc, vc1, dps + 1);
				quadTree(xc, yc, dx, dy, vcc, vc1, v1c, v11, dps + 1);
				// line may "break" in this situation
				return true;
			}
		}
		return false;
	}
	else {
		dx *= 0.5, dy *= 0.5;
		double xc = x0 + dx, yc = y0 + dy;
		double vc0 = fun(xc, y0), vc1 = fun(xc, y1), v0c = fun(x0, yc), v1c = fun(x1, yc), vcc = fun(xc, yc);
		bool r = quadTree(x0, y0, dx, dy, v00, v0c, vc0, vcc, dps + 1, !(Intp2(v00, vc0, v10) || Intp2(v00, v0c, v01)));
		r |= quadTree(xc, y0, dx, dy, vc0, vcc, v10, v1c, dps + 1, !(Intp2(v10, vc0, v00) || Intp2(v10, v1c, v11)));
		r |= quadTree(x0, yc, dx, dy, v0c, v01, vcc, vc1, dps + 1, !(Intp2(v01, vc1, v11) || Intp2(v01, v0c, v00)));
		r |= quadTree(xc, yc, dx, dy, vcc, vc1, v1c, v11, dps + 1, !(Intp2(v11, vc1, v01) || Intp2(v11, v1c, v10)));
		return r;
	}
}
void marchSquare(double x0, double y0, double dx, double dy) {
	int N = 1 << SEARCH_DPS, M = N + 1;
	dx /= N, dy /= N;
	double *val = new double[M*M];
	for (int i = 0; i <= N; i++) for (int j = 0; j <= N; j++) {
		val[i*M + j] = fun(x0 + i * dx, y0 + j * dy);
	}
	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
		double v00 = val[i*M + j], v01 = val[i*M + (j + 1)], v10 = val[(i + 1)*M + j], v11 = val[(i + 1)*M + (j + 1)];
		double x = x0 + i * dx, y = y0 + j * dy, x1 = x + dx, y1 = y + dy;
		//vec2 n = vec2((i ? .5*(val[(i + 1)*M + j] - val[(i - 1)*M + j]) : val[M + j] - val[j]) / dx, (j ? .5*(val[i*M + (j + 1)] - val[i*M + (j - 1)]) : val[i*M + 1] - val[i*M]) / dy);
		//drawLineF(vec2(x, y), vec2(x, y) + n, WHITE);
		if ((signbit(v00) + signbit(v01) + signbit(v10) + signbit(v11)) % 4) {
			quadTree(x, y, dx, dy, v00, v01, v10, v11, SEARCH_DPS);
			//drawBox(x, x1, y, y1, RED);
		}
		else {
			if ((i && (Intp2(v10, v00, val[(i - 1)*M + j]) || Intp2(v11, v01, val[(i - 1)*M + (j + 1)]))) ||
				(i + 1 != N && (Intp2(v00, v10, val[(i + 2)*M + j]) || Intp2(v01, v11, val[(i + 2)*M + (j + 1)]))) ||
				(j && (Intp2(v01, v00, val[i*M + (j - 1)]) || Intp2(v11, v10, val[(i + 1)*M + (j - 1)]))) ||
				(j + 1 != N && (Intp2(v00, v01, val[i*M + (j + 2)]) || Intp2(v10, v11, val[(i + 1)*M + (j + 2)])))) {  // I believe there's bug but it's hard to debug
				quadTree(x, y, dx, dy, v00, v01, v10, v11, SEARCH_DPS, false);
				//drawBox(x, x1, y, y1, YELLOW);
			}
		}
	}
}

void render() {
	// debug
	auto t1 = NTime::now();
	double dt = std::chrono::duration<double>(t1 - t0).count();
	dbgprint("[%d×%d] time elapsed: %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, 1000.0*dt, 1. / dt);
	t0 = t1;

	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;

	// "pixel shader" graphing implicit curve
#if 0
	for (int i = 0; i < _WIN_W; i++) for (int j = 0; j < _WIN_H; j++) {
		vec2 p = fromInt(vec2(i, j)); V[i][j] = fun(p.x, p.y);
		double dx = i ? V[i][j] - V[i - 1][j] : NAN;
		double dy = j ? V[i][j] - V[i][j - 1] : NAN;
		double v = V[i][j] / length(vec2(dx, dy));
		COLORREF col = (byte)(50.*clamp(5.0 - abs(v), 0, 1));
		_WINIMG[j*_WIN_W + i] = col | (col << 8) | (col) << 16;
	}
#endif

	// axis and grid
	{
		vec2 LB = fromInt(vec2(0, 0)), RT = fromInt(vec2(_WIN_W, _WIN_H));
		COLORREF GridCol = clamp(0x10, (byte)sqrt(10.0*Unit), 0x20); GridCol = GridCol | (GridCol << 8) | (GridCol) << 16;  // adaptive grid color
		for (int y = (int)round(LB.y), y1 = (int)round(RT.y); y <= y1; y++) drawLine(fromFloat(vec2(LB.x, y)), fromFloat(vec2(RT.x, y)), GridCol);  // horizontal gridlines
		for (int x = (int)round(LB.x), x1 = (int)round(RT.x); x <= x1; x++) drawLine(fromFloat(vec2(x, LB.y)), fromFloat(vec2(x, RT.y)), GridCol);  // vertical gridlines
		drawLine(vec2(0, Center.y), vec2(_WIN_W, Center.y), 0x202080);  // x-axis
		drawLine(vec2(Center.x, 0), vec2(Center.x, _WIN_H), 0x202080);  // y-axis
	}

	int PC = max((int)(0.5*log2(_WIN_W*_WIN_H) + 0.5), 8);
	SEARCH_DPS = PC - 3, PLOT_DPS = PC - 1;
	vec2 p0 = fromInt(vec2(0, 0)), dp = fromInt(vec2(_WIN_W, _WIN_H)) - p0;
	evals = 0, lines = 0;
	marchSquare(p0.x, p0.y, dp.x, dp.y);


	vec2 cursor = fromInt(Cursor);
	sprintf(text, "(%.2f,%.2f)  %d evals, %d segments", cursor.x, cursor.y, evals, lines);
	SetWindowTextA(_HWND, text);
}


// ============================================== User ==============================================


void WindowCreate(int _W, int _H) {
	Center = vec2(_W, _H) * 0.5;
}
void WindowResize(int _oldW, int _oldH, int _W, int _H) {
	if (_W*_H == 0 || _oldW * _oldH == 0) return;
	double pw = _oldW, ph = _oldH, w = _W, h = _H;
	double s = sqrt((w * h) / (pw * ph));
	Unit *= s;
	Center.x *= w / pw, Center.y *= h / ph;
}
void WindowClose() {

}

void MouseMove(int _X, int _Y) {
	vec2 P0 = Cursor, P = vec2(_X, _Y);
	Cursor = P;
	vec2 p0 = fromInt(P0), p = fromInt(P), d = p - p0;

	// click and drag
	if (mouse_down) {
		Center = Center + d * Unit;
	}

}

void MouseWheel(int _DELTA) {
	double s = exp((Alt ? 0.0001 : 0.001)*_DELTA);
	double D = length(vec2(_WIN_W, _WIN_H)), Max = D, Min = 0.02*D;
	if (Unit * s > Max) s = Max / Unit;
	else if (Unit * s < Min) s = Min / Unit;
	Center = mix(Cursor, Center, s);
	Unit *= s;
}

void MouseDownL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	mouse_down = true;
	vec2 p = fromInt(Cursor);
}

void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	mouse_down = false;
	vec2 p = fromInt(Cursor);
}

void MouseDownR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
}

void MouseUpR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	vec2 p = fromInt(Cursor);
}

void KeyDown(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = true;
	else if (_KEY == VK_SHIFT) Shift = true;
	else if (_KEY == VK_MENU) Alt = true;
}

void KeyUp(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = false;
	else if (_KEY == VK_SHIFT) Shift = false;
	else if (_KEY == VK_MENU) Alt = false;
}

