// Graphing Implicit Curve


/* ==================== User Instructions ====================

 *  Move View:                  drag background
 *  Zoom:                       mouse scroll, hold shift to lock center
 */



 // ========================================= Win32 GUI =========================================

#pragma region Windows

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>

#define WIN_NAME "Implicit curve visualization"
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
int main() {
	HINSTANCE hInstance = NULL; int nCmdShow = SW_RESTORE;
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME); if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

// debug
#define dbgprint(format, ...) { wchar_t buf[0x4FFF]; swprintf(buf, 0x4FFF, _T(format), ##__VA_ARGS__); OutputDebugStringW(buf); }

#pragma endregion


// ================================== Vector Classes/Functions ==================================

#pragma region Vector Classes/Functions

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
	vec2 rot() const { return vec2(-y, x); }   // not standard
};


#pragma endregion


// ======================================== Data / Parameters ========================================

#include <stdio.h>

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



namespace combined_function {
	double sdLine(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a, ba = b - a; return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0., 1.)); }
	double u(vec2 p, double t) { return min(min(p.x - 0.16, sdLine(p, 0, vec2(1.22, 0))), min(sdLine(vec2(p.x, abs(p.y)), vec2(0.45, 0), vec2(0.45 + 0.42*cos(t), 0.42*sin(t))), sdLine(vec2(p.x, abs(p.y)), vec2(0.8, 0), vec2(0.8 + 0.35*cos(t), 0.35*sin(t))))); }
	double fun(vec2 p, double t) { double a = abs(fmod(atan2(p.y, p.x) + .5*t, PI / 3.)) - PI / 6.; return u(vec2(cos(a), sin(a))*length(p)*(1.0 + 0.3*cos(t)), 0.9 - 0.3*cos(t)) - 0.05; }
}

int evals;
double fun(double x, double y) {
	evals++;
	//return x * x + y * y - 1;  // 3.141593
	//return hypot(x, y) - 1;  // 3.141593
	//return x * x*x*(x - 2) + y * y*y*(y - 2) + x;  // 5.215079
	//return x * x*x*x + y * y*y*y + x * y;  // 0.785398
	//return abs(abs(max(abs(x) - 0.9876, abs(y + 0.1*sin(3.*x)) - 0.4321)) - 0.3456) - 0.1234;
	//return max(2.*y - sin(10. * x), 4.*x*x + 4.*y * y - 9.);  // 14.13[6-7]
	//return max(abs(x) - abs(y) - 1, x*x + y * y - 2);  // 5.96[7-8]
	//return max(.5 - abs(x*y), x*x + 2 * y*y - 3);  // 1.81[0-3]
	//return max(sin(10*x) + cos(10*y) + 1., x*x + y * y - 1.);  // 0.57[3-4]
	return combined_function::fun(vec2(x, y), 1.7);  // 1.7[6-8]
}

// ============================================ Marching ============================================


// forward declarations of rendering functions
void drawLine(vec2 p, vec2 q, COLORREF col);
void drawBox(double x0, double x1, double y0, double y1, COLORREF col);
#define drawLineF(p,q,col) drawLine(fromFloat(p),fromFloat(q),col)


// global array to store marched segments
#include <vector>
struct segment {
	vec2 a, b;
	segment(vec2 a, vec2 b, bool swap = false) {  // swap: make sure the negative region is on the left
		if (swap) this->a = b, this->b = a;
		else this->a = a, this->b = b;
	}
	void swap() { std::swap(a, b); }
};
std::vector<segment> Segments;


int SEARCH_DPS;
int PLOT_DPS;

// linear interpolation
double Intp1(double a, double b) {
	return a / (a - b);
};


// information about if the boundaries have segments
typedef unsigned char byte;
const byte biR = 0b0001, biT = 0b0010, biL = 0b0100, biB = 0b1000;

// give start position, increasement of position, and pre-calculated values at the edges
// forceMarch: lower 4 bits indicate whether to force search an edge; usually called when all v values are the same sign
// return a byte where its lower 4 bits indicate if there are segments intersecting the boundary
byte quadTree(double x0, double y0, double dx, double dy, double v00, double v01, double v10, double v11, int dps, byte forceMarch = 0) {
	double x1 = x0 + dx, y1 = y0 + dy;
	byte Res = 0;

#define NE(a,b) (abs((b)-(a))>1e-12)
	/*if (NE(fun(x0, y0), v00) || NE(fun(x0, y1), v01) || NE(fun(x1, y0), v10) || NE(fun(x1, y1), v11)) {
		dbgprint("E");
	}*/

	// get the sign of values at the endpoints
	bool s00 = v00 < 0, s01 = v01 < 0, s10 = v10 < 0, s11 = v11 < 0;
	int s = s00 + s01 + s10 + s11;
	if (!forceMarch && s % 4 == 0) return Res;

	if (dps > SEARCH_DPS) drawBox(x0, x0 + dx, y0, y0 + dy, forceMarch ? 0x00FF00 : 0x008080);

	if (dps >= PLOT_DPS) {
		/* recursion depth limit reached */
		vec2 p00(x0, y0), p01(x0, y1), p10(x1, y0), p11(x1, y1), xd(dx, 0), yd(0, dy);
		if (s == 1 || s == 3) {
			// cut one vertex
			if (s == 3) s00 ^= 1, s01 ^= 1, s10 ^= 1, s11 ^= 1;
			if (s00) Segments.push_back(segment(p00 + xd * Intp1(v00, v10), p00 + yd * Intp1(v00, v01), s == 3)), Res |= biB | biL;
			if (s01) Segments.push_back(segment(p01 + xd * Intp1(v01, v11), p01 - yd * Intp1(v01, v00), s == 1)), Res |= biL | biT;
			if (s10) Segments.push_back(segment(p10 - xd * Intp1(v10, v00), p10 + yd * Intp1(v10, v11), s == 1)), Res |= biB | biR;
			if (s11) Segments.push_back(segment(p11 - xd * Intp1(v11, v01), p11 - yd * Intp1(v11, v10), s == 3)), Res |= biR | biT;
			return Res;
		}
		if (s == 2) {
			// two pairs are neighborhood
			if ((s00&&s01) || (s10&&s11))  // vertical split
				Segments.push_back(segment(p00 + xd * Intp1(v00, v10), p01 + xd * Intp1(v01, v11), s10&&s11)), Res |= biB | biT;
			else if ((s00&&s10) || (s01&&s11))  // horizontal split
				Segments.push_back(segment(p00 + yd * Intp1(v00, v01), p10 + yd * Intp1(v10, v11), s00&&s10)), Res |= biL | biR;
			// two segments required, not often
			else {
				// evaluate function at the center of the square
				double vcc = fun(x0 + 0.5*dx, y0 + 0.5*dy);
				bool scc = vcc < 0;
				// in this case, interpolation often do not work well
				if (scc == s00 && scc == s11) {
					Segments.push_back(segment(p00 + xd * Intp1(v00, v10), p10 + yd * Intp1(v10, v11), vcc));
					Segments.push_back(segment(p00 + yd * Intp1(v00, v01), p01 + xd * Intp1(v01, v11), !vcc));
				}
				else if (scc == s01 && scc == s10) {
					Segments.push_back(segment(p00 + xd * Intp1(v00, v10), p00 + yd * Intp1(v00, v01), !vcc));
					Segments.push_back(segment(p10 + yd * Intp1(v10, v11), p01 + xd * Intp1(v01, v11), vcc));
				}
				else throw(__LINE__);
				Res |= biT | biB | biL | biR;
				drawBox(x0, x1, y0, y1, 0xFF00FF);
			}
		}
		return Res;
	}
	else {
		/* recursively */
		dx *= 0.5, dy *= 0.5;
		double xc = x0 + dx, yc = y0 + dy;
		double vc0 = fun(xc, y0), vc1 = fun(xc, y1), v0c = fun(x0, yc), v1c = fun(x1, yc), vcc = fun(xc, yc);
		byte Rtn[2][2];  // return values
		Rtn[0][0] = quadTree(x0, y0, dx, dy, v00, v0c, vc0, vcc, dps + 1, dps >= SEARCH_DPS ? 0 : 0b1111);
		Rtn[1][0] = quadTree(xc, y0, dx, dy, vc0, vcc, v10, v1c, dps + 1, dps >= SEARCH_DPS ? 0 : 0b1111);
		Rtn[0][1] = quadTree(x0, yc, dx, dy, v0c, v01, vcc, vc1, dps + 1, dps >= SEARCH_DPS ? 0 : 0b1111);
		Rtn[1][1] = quadTree(xc, yc, dx, dy, vcc, vc1, v1c, v11, dps + 1, dps >= SEARCH_DPS ? 0 : 0b1111);
		// fix missed zeros
		// doesn't always work (eg. contain both success and failed subtrees; indirectly connected;)
		byte Rem[2][2] = { {0, 0}, {0, 0} };  // remarching parameters
		{
			// case works
			if ((Rtn[0][0] & biR) && !(Rtn[1][0]/* & biL*/)) Rem[1][0] |= biL;
			if ((Rtn[0][0] & biT) && !(Rtn[0][1]/* & biB*/)) Rem[0][1] |= biB;
			if ((Rtn[1][1] & biB) && !(Rtn[1][0]/* & biT*/)) Rem[1][0] |= biT;
			if ((Rtn[1][1] & biL) && !(Rtn[0][1]/* & biR*/)) Rem[0][1] |= biR;
			if ((Rtn[1][0] & biL) && !(Rtn[0][0]/* & biR*/)) Rem[0][0] |= biR;
			if ((Rtn[1][0] & biT) && !(Rtn[1][1]/* & biB*/)) Rem[1][1] |= biB;
			if ((Rtn[0][1] & biB) && !(Rtn[0][0]/* & biT*/)) Rem[0][0] |= biT;
			if ((Rtn[0][1] & biR) && !(Rtn[1][1]/* & biL*/)) Rem[1][1] |= biL;
		}
		{
			if (Rem[1][0]) Rtn[1][0] |= quadTree(xc, y0, dx, dy, vc0, vcc, v10, v1c, dps + 1, Rem[1][0]);
			if (Rem[0][1]) Rtn[0][1] |= quadTree(x0, yc, dx, dy, v0c, v01, vcc, vc1, dps + 1, Rem[0][1]);
			if (Rem[0][0]) Rtn[0][0] |= quadTree(x0, y0, dx, dy, v00, v0c, vc0, vcc, dps + 1, Rem[0][0]);
			if (Rem[1][1]) Rtn[1][1] |= quadTree(xc, yc, dx, dy, vcc, vc1, v1c, v11, dps + 1, Rem[1][1]);
		}
		Rtn[0][0] &= biL | biB;
		Rtn[1][0] &= biR | biB;
		Rtn[0][1] &= biL | biT;
		Rtn[1][1] &= biR | biT;
		return Rtn[0][0] | Rtn[0][1] | Rtn[1][0] | Rtn[1][1];
	}
}
void marchSquare(double x0, double y0, double dx, double dy) {
#if 1
	double x1 = x0 + dx, y1 = y0 + dy;
	double v00 = fun(x0, y0), v10 = fun(x1, y0), v01 = fun(x0, y1), v11 = fun(x1, y1);
	quadTree(x0, y0, dx, dy, v00, v01, v10, v11, 0, 0b1111);
	return;
#else
	int N = 1 << SEARCH_DPS, M = N + 1;
	dx /= N, dy /= N;
	double *val = new double[M*M];
	for (int i = 0; i <= N; i++) for (int j = 0; j <= N; j++) {
		val[i*M + j] = fun(x0 + i * dx, y0 + j * dy);
	}
	for (int i = 0; i < N; i++) {  // x
		for (int j = 0; j < N; j++) {  // y
			double v00 = val[i*M + j], v01 = val[i*M + (j + 1)], v10 = val[(i + 1)*M + j], v11 = val[(i + 1)*M + (j + 1)];
			double x = x0 + i * dx, y = y0 + j * dy, x1 = x + dx, y1 = y + dy;
			if ((signbit(v00) + signbit(v01) + signbit(v10) + signbit(v11)) % 4) {
				byte R = quadTree(x, y, dx, dy, v00, v01, v10, v11, SEARCH_DPS);
				if (R & biL) {
					//if (i != 0 && !((signbit(u00) + signbit(u01) + signbit(u10) + signbit(u11)) % 4)) {
						//quadTree(x - dx, y, dx, dy, u00, u01, u10, u11, SEARCH_DPS);
					drawBox(x, x, y, y1, 0xFF0000);
					//}
				}
				drawBox(x, x1, y, y1, 0xFF0000);
			}
			else {

			}
		}
	}
#endif
}

// ============================================ Rendering ============================================

auto t0 = NTime::now();

double V[1920][1080];

void drawLine(vec2 p, vec2 q, COLORREF col) {
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
void drawBox(double x0, double x1, double y0, double y1, COLORREF col) {
	drawLineF(vec2(x0, y0), vec2(x1, y0), col);
	drawLineF(vec2(x0, y1), vec2(x1, y1), col);
	drawLineF(vec2(x0, y0), vec2(x0, y1), col);
	drawLineF(vec2(x1, y0), vec2(x1, y1), col);
};


void remarch() {
	const bool CONSTANT = 1;
	int PC = (int)(0.5*log2(_WIN_W*_WIN_H) + 0.5);  // depth
	//SEARCH_DPS = max(PC - 3, 6), PLOT_DPS = max(PC - 1, 12);
	SEARCH_DPS = 4, PLOT_DPS = 8;
	vec2 p0 = fromInt(vec2(0, 0)), dp = fromInt(vec2(_WIN_W, _WIN_H)) - p0;  // starting position and increasement
	if (CONSTANT) p0 = vec2(-2.5, -2.5), dp = vec2(5, 5);
	p0 = p0 + vec2(sin(123.4), sin(234.5))*1e-8;  // avoid degenerated cases
	evals = 0;
	Segments.clear();
	marchSquare(p0.x, p0.y, dp.x, dp.y);
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

	//if (Segments.empty())
	remarch();
	int SN = Segments.size();

	// rendering
	for (int i = 0; i < SN; i++) {
		vec2 a = Segments[i].a, b = Segments[i].b, c = (a + b)*0.5;
		drawLineF(a, b, 0xFFFFFF);
		drawLineF(c, c + (b - a).rot(), 0xFFFF00);
	}

	// calculate area
	double Area = 0;
	for (int i = 0; i < SN; i++) {
		Area += det(Segments[i].a, Segments[i].b);
	}
	Area *= 0.5;

	vec2 cursor = fromInt(Cursor);
	sprintf(text, "(%.2f,%.2f)  %d evals, %d segments, Area=%lf", cursor.x, cursor.y, evals, SN, Area);
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
	double D = length(vec2(_WIN_W, _WIN_H)), Max = 10.0*D, Min = 0.01*D;
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

