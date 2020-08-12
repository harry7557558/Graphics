// Test Different Splines and Interpolation Methods
// Incomplete, still updating


/* ==================== User Instructions ====================

 *  Move View:                  drag background
 *  Zoom:                       mouse scroll, hold shift to lock center
 *  Move Point:                 click and drag
 *  Move Shape:                 alt + drag; shift + m
 *  Rotate Shape:               shift + drag
 *  Scale Shape:                S + drag
 *  Add Point:                  ctrl + click
 *  Delete Point:               right click
 *  Switch Endpoint:            left / right
 *  Reverse Point Direction:    space
 *  Center Points:              shift + c (only available in FourierSeries mode)
 *  Hide/Unhide Points:         c
 *  Next Method:                tab
 *  Last Method:                ctrl/shift + tab
 *  Show/Hide Trace:            b
 *  Lock/Unlock Trace:          l
 *  Trace Alpha:                ctrl + mouse scroll
 *  Save File (append):         ctrl + s
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
	float x, y;
	vec2() {}
	vec2(float a) :x(a), y(a) {}
	vec2(float x, float y) :x(x), y(y) {}
	vec2 operator - () const { return vec2(-x, -y); }
	vec2 operator + (const vec2 &v) const { return vec2(x + v.x, y + v.y); }
	vec2 operator - (const vec2 &v) const { return vec2(x - v.x, y - v.y); }
	vec2 operator * (const vec2 &v) const { return vec2(x * v.x, y * v.y); }	// not standard but useful
	vec2 operator * (const float &a) const { return vec2(x*a, y*a); }
	float sqr() const { return x * x + y * y; } 	// not standard
	friend float length(const vec2 &v) { return sqrt(v.x*v.x + v.y*v.y); }
	friend vec2 normalize(const vec2 &v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y)); }
	friend float dot(const vec2 &u, const vec2 &v) { return u.x*v.x + u.y*v.y; }
	friend float det(const vec2 &u, const vec2 &v) { return u.x*v.y - u.y*v.x; } 	// not standard
#if 0
	vec2 operator == (const vec2 &v) const { return x == v.x && y == v.y; }
	vec2 operator != (const vec2 &v) const { return x != v.x || y != v.y; }
	void operator += (const vec2 &v) { x += v.x, y += v.y; }
	void operator -= (const vec2 &v) { x -= v.x, y -= v.y; }
	void operator *= (const vec2 &v) { x *= v.x, y *= v.y; }
	vec2 operator / (const vec2 &v) const { return vec2(x / v.x, y / v.y); }
	void operator /= (const vec2 &v) { x /= v.x, y /= v.y; }
	friend vec2 operator * (const float &a, const vec2 &v) { return vec2(a*v.x, a*v.y); }
	void operator *= (const float &a) { x *= a, y *= a; }
	vec2 operator / (const float &a) const { return vec2(x / a, y / a); }
	void operator /= (const float &a) { x /= a, y /= a; }
#endif
	vec2 rot() const { return vec2(-y, x); }   // not standard
};

float sdSqLine(vec2 p, vec2 a, vec2 b) {	// by iq
	vec2 pa = p - a, ba = b - a;
	float h = dot(pa, ba) / dot(ba, ba);
	return (pa - ba * clamp(h, 0.0, 1.0)).sqr();
}

#pragma endregion


// ======================================== Data / Parameters ========================================

#include <stdio.h>
#pragma warning(disable: 4996)

enum Interpolation {  // or fitting
	Linear, CatmullRom,
	QuadraticB, CubicB,
	QuadraticInt, CubicBInt,
	FourierSeries,
	IntN  // number of supported interpolations
};
const char* IntpName[] = {
	"Linear Interpolation", "Centripetal Catmull-Rom Spline",
	"Quadratic B-Spline", "Standard Cubic B-Spline",
	"Quadratic Interpolation ###", "Cubic B-Spline Interpolation ###",
	"Fourier Series Fitting"
};
Interpolation IntpMethod = Linear;
typedef Interpolation Intp;


#pragma region Global Variables

#include <vector>
std::vector<vec2> CP({ vec2(1,-1), vec2(1,1), vec2(-1,1), vec2(-1,-1) });	// control points
int CP_Selected = -1;	// selected (drag)
int CP_Insert = -1;	// insert index

// window parameters
char text[64];	// window title
vec2 Center = vec2(0, 0);	// origin in screen coordinate
float Unit = 100.0;		// screen unit to object unit
#define fromInt(p) (((p) - Center) * (1.0 / Unit))
#define fromFloat(p) ((p) * Unit + Center)

// rendering parameters
int CPR = 8;	// rendering radius of control point
bool showControl = true;	// control points
bool showBackground = false;	// background image for trace

// user parameters
vec2 Cursor = vec2(0, 0);
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false, S_Key = false;

#pragma endregion

#pragma region Background Image

// image format library, https://github.com/nothings/stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "libraries\stb_image.h"

COLORREF *BKG_ORIGIN = 0, *BKG = 0;	// background image for tracing, where BKG has transparency
int BKG_W, BKG_H;
bool lockBackground = false;
vec2 BKGPos(0.0);		// lower-left corner on screen coordinate
float BKGUnit = 1.0;	// to screen coordinate
float BKGAlpha = 0.5;	// transparency

// load background image from file
bool loadBackground(const wchar_t* filename) {
	FILE *fp = _wfopen(filename, L"rb");
	if (fp == 0) return false;
	COLORREF* temp = (COLORREF*)stbi_load_from_file(fp, &BKG_W, &BKG_H, 0, 4);
	fclose(fp);
	if (temp == 0) return false;
	if (BKG_ORIGIN != 0) delete BKG_ORIGIN;
	BKG_ORIGIN = temp;
	for (int i = 0; i < BKG_H / 2; i++) for (int j = 0; j < BKG_W; j++)
		std::swap(BKG_ORIGIN[i*BKG_W + j], BKG_ORIGIN[(BKG_H - 1 - i)*BKG_W + j]);  // flip back: different image coordinates
	int L = BKG_W * BKG_H;
	for (int i = 0; i < L; i++) std::swap(((byte*)&temp[i])[0], ((byte*)&temp[i])[2]);	// rgb vs bgr
	if (BKG != 0) delete BKG;
	BKG = new COLORREF[L];
	for (int i = 0; i < 4 * L; i++) ((byte*)BKG)[i] = BKGAlpha * ((byte*)BKG_ORIGIN)[i];
	return true;
}
bool selectBackground() {
	OPENFILENAME ofn = { sizeof(OPENFILENAME) };
	WCHAR filename[MAX_PATH] = L"";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;
	if (!GetOpenFileName(&ofn)) return false;
	if (!loadBackground(filename)) return false;
	showBackground = true;
	lockBackground = false;
	return true;
}

// call this when alpha changed
void refreshBackground() {
	if (BKG && BKG_ORIGIN)
		for (int i = 0, l = 4 * BKG_W*BKG_H; i < l; i++)
			((byte*)BKG)[i] = BKGAlpha * ((byte*)BKG_ORIGIN)[i];
}

// get backgroud color from screen coordinate
COLORREF getBackground(int x, int y) {
	x = (x - BKGPos.x)*BKGUnit, y = (y - BKGPos.y)*BKGUnit;
	if (x < 0 || y < 0 || x >= BKG_W || y >= BKG_H) return 0;
	return BKG[y*BKG_W + x];
}

#pragma endregion For Tracing


#define _13 0.33333333f
#define _16 0.16666667f
class spline3 {
public:
	vec2 C3, C2, C1, C0;
	spline3(vec2 C3, vec2 C2, vec2 C1, vec2 C0) :C3(C3), C2(C2), C1(C1), C0(C0) {}
};
#define SpCatmullRom(A,B,C,D) spline3((D-A)*0.5+(B-C)*1.5, A-B*2.5+C*2.0-D*0.5, (C-A)*0.5, B)
#define SpQuadraticB(A,B,C) spline3(vec2(0.), (A+C)*0.5-B, B-A, (A+B)*0.5)
#define SpCubicB(A,B,C,D) spline3((D-A)*_16+(B-C)*0.5, (A+C)*0.5-B, (C-A)*0.5, (A+B*4.+C)*_16)

spline3 getSpline(int i) {
	int n = CP.size();
	switch (IntpMethod) {
	case Linear: {
		return spline3(vec2(0.0), vec2(0.0), CP[(i + 1) % n] - CP[i], CP[i]);
	}
	case CatmullRom: {
		vec2 A = CP[(i + n - 1) % n], B = CP[i], C = CP[(i + 1) % n], D = CP[(i + 2) % n];
		return SpCatmullRom(A, B, C, D);
	}
	case QuadraticB: {
		vec2 A = CP[(i + n - 1) % n], B = CP[i], C = CP[(i + 1) % n];
		return SpQuadraticB(A, B, C);
	}
	case CubicB: {
		vec2 A = CP[(i + n - 1) % n], B = CP[i], C = CP[(i + 1) % n], D = CP[(i + 2) % n];
		return SpCubicB(A, B, C, D);
	}
	default: {
		return spline3(0, 0, 0, 0);
	}
	}
}

spline3 fromFloatSp(spline3 s) {
	return spline3(s.C3*Unit, s.C2*Unit, s.C1*Unit, s.C0*Unit + Center);
}

void calcFourierParameter(vec2 *a, vec2 *b, int N) {
	float t, dt;
	int n = CP.size();
	for (int k = 0; k < N; k++) {
		a[k] = vec2(0.0), b[k] = vec2(0.0);
		t = 0.0, dt = 2.0 * k * PI / n;  // uniform distribute parameter
		for (int i = 0; i < n; i++, t += dt) {
			a[k] = a[k] + CP[i] * cos(t);
			b[k] = b[k] + CP[i] * sin(t);
		}
		a[k] = a[k] * (2.0 / n), b[k] = b[k] * (2.0 / n);
	}
	a[0] = a[0] * 0.5;
}

// return the average of all control points
vec2 calcCenter() {
	vec2 p(0.0);
	for (int i = 0, n = CP.size(); i < n; i++) p = p + CP[i];
	return p * (1.0 / CP.size());
}

// put this at the end because they contain long string code
bool saveFile();
bool readFile();


// ============================================ Rendering ============================================

#define DARKBLUE 0x202080
#define WHITE 0xFFFFFF
#define YELLOW 0xFFFF00
#define RED 0xFF0000
#define RED0 0xFF0080
#define RED1 0xFF0040
#define ORANGE 0xFF8000
#define LIME 0x00FF00
#define MAGENTA 0xFF00FF
#define DARKGRAY 0x404040

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
auto t0 = NTime::now();


void render() {
	// debug
	auto t1 = NTime::now();
	float dt = std::chrono::duration<float>(t1 - t0).count();
	dbgprint("[%d×%d] time elapsed: %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, 1000.0*dt, 1. / dt);
	t0 = t1;

	// initialize window
	if (showBackground && BKG)
		for (int i = 0; i < _WIN_W; i++) for (int j = 0; j < _WIN_H; j++)
			_WINIMG[j*_WIN_W + i] = getBackground(i, j);
	else for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;

	// rendering
	auto drawLine = [&](vec2 p, vec2 q, COLORREF col) {
		vec2 d = q - p;
		float slope = d.y / d.x;
		if (abs(slope) <= 1.0) {
			if (p.x > q.x) std::swap(p, q);
			int x0 = max(0, int(p.x)), x1 = min(_WIN_W - 1, int(q.x)), y;
			float yf = slope * x0 + (p.y - slope * p.x);
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
			float xf = slope * y0 + (p.x - slope * p.y);
			for (int y = y0; y <= y1; y++) {
				x = (int)xf;
				if (x >= 0 && x < _WIN_W) Canvas(x, y) = col;
				xf += slope;
			}
		}
	};
	auto drawCircle = [&](vec2 c, float r, COLORREF col) {
		int x0 = max(0, int(c.x - r)), x1 = min(_WIN_W - 1, int(c.x + r));
		int y0 = max(0, int(c.y - r)), y1 = min(_WIN_H - 1, int(c.y + r));
		int cx = (int)c.x, cy = (int)c.y, r2 = int(r*r), dx, dy;
		for (int x = x0; x <= x1; x++) {
			dx = x - cx;
			for (int y = y0; y <= y1; y++) {
				dy = y - cy;
				if (dx*dx + dy * dy < r2) Canvas(x, y) = col;
			}
		}
	};
	auto drawSpline = [&](spline3 Sp, COLORREF col) {	// screen coordinate, C3 t³ + C2 t² + C1 t + C0
		const float dt = 0.01;
		vec2 p = Sp.C0, q;
		for (float t = dt; t < 1.0; t += dt) {
			q = ((Sp.C3*t + Sp.C2)*t + Sp.C1)*t + Sp.C0;
			drawLine(p, q, col);
			p = q;
		}
	};

	// axis
	drawLine(vec2(0, Center.y), vec2(_WIN_W, Center.y), DARKBLUE);
	drawLine(vec2(Center.x, 0), vec2(Center.x, _WIN_H), DARKBLUE);

	// control polygon
	int n = CP.size();
	if (showControl) for (int i = 0; i < n; i++) {
		drawLine(fromFloat(CP[i]), fromFloat(CP[(i + 1) % n]), i == CP_Insert ? WHITE : DARKGRAY);
	}

	// interpolation curve
	switch (IntpMethod) {
	case Linear: {
		for (int i = 0; i < n; i++)
			drawLine(fromFloat(CP[i]), fromFloat(CP[(i + 1) % n]), WHITE);
		break;
	}
	case FourierSeries: {
		int N = n / 2 + 1;
		vec2 *a = new vec2[N], *b = new vec2[N];
		calcFourierParameter(a, b, N);
		auto eval = [&](float t) ->vec2 {
			vec2 r = a[0];
			for (int k = 1; k < N; k++) r = r + a[k] * cos(k*t) + b[k] * sin(k*t);
			return fromFloat(r);
		};
		const int D = 100 * CP.size();
		float t = 0.0, dt = 1.0 / D;
		vec2 p = eval(t), q;
		for (t = dt; t < 2.0*PI; t += dt) {
			q = eval(t);
			drawLine(p, q, WHITE);
			p = q;
		}
		delete a, b;
		break;
	}
	default: {
		for (int i = 0; i < n; i++)
			drawSpline(fromFloatSp(getSpline(i)), WHITE);
		break;
	}
	}

	// control points
	if (showControl) for (int i = n - 1; i >= 0; i--) {
		vec2 P = fromFloat(CP[i]);
		if (i == CP_Selected) drawCircle(P, CPR, YELLOW);
		else {
			COLORREF col = CP_Insert != -1 && (i == CP_Insert || i == (CP_Insert + 1) % n) ? LIME : RED;	// highlight insert position
			drawCircle(P, CPR - min(i, 2), col);	// the two larger points are startpoint
		}
	}

	// show the center of the figure
	if ((IntpMethod == FourierSeries && showControl) || Alt || Shift || S_Key) {
		vec2 C = fromFloat(calcCenter());
		drawLine(C - vec2(5, 0), C + vec2(5, 0), LIME);
		drawLine(C - vec2(0, 5), C + vec2(0, 5), LIME);
	}

	//if (dt < 0.016) Sleep(16 - int(1000 * dt)), t0 = NTime::now();	// max 60fps reduce CPU usage
	vec2 cursor = fromInt(Cursor);
	sprintf(text, "%s - %d points %s (%.2f,%.2f)", IntpName[IntpMethod], CP.size(),
		showBackground ? (lockBackground ? " locked " : " unlock ") : "  ",
		cursor.x, cursor.y);
	SetWindowTextA(_HWND, text);
}


// ============================================== User ==============================================


void WindowCreate(int _W, int _H) {
	Center = vec2(_W, _H) * 0.5;
	loadBackground(L"D:\\trace.png");
}
void WindowResize(int _oldW, int _oldH, int _W, int _H) {
	if (_W*_H == 0 || _oldW * _oldH == 0) return;
	float pw = _oldW, ph = _oldH, w = _W, h = _H;
	float s = sqrt((w * h) / (pw * ph));
	Unit *= s;
	if (lockBackground) BKGUnit /= s, BKGPos = fromInt(BKGPos) * s;
	Center.x *= w / pw, Center.y *= h / ph;
	if (lockBackground) BKGPos = fromFloat(BKGPos);
}
void WindowClose() {
	delete BKG, BKG_ORIGIN;
}

void MouseMove(int _X, int _Y) {
	vec2 P0 = Cursor, P = vec2(_X, _Y);
	Cursor = P;
	vec2 p0 = fromInt(P0), p = fromInt(P), d = p - p0;
	CP_Insert = -1;

	// click and drag
	if (mouse_down) {
		if (Alt) for (unsigned i = 0; i < CP.size(); i++) CP[i] = CP[i] + d;
		else if (showControl && CP_Selected != -1) CP[CP_Selected] = CP[CP_Selected] + d;	// control point
		else if (Shift || S_Key) {
			vec2 C = calcCenter();
			if (Shift) {
				double s = det(normalize(p0 - C), normalize(p - C)), c = sqrt(1.0 - s * s);
				vec2 R(s, c);
				for (int i = 0, n = CP.size(); i < n; i++) CP[i] = vec2(det(CP[i] - C, R), dot(CP[i] - C, R)) + C;
			}
			if (S_Key) {
				double S = length(p - C) / length(p0 - C);
				for (int i = 0, n = CP.size(); i < n; i++) CP[i] = (CP[i] - C)*S + C;
			}
		}
		else {	// drag axis and grid
			Center = Center + d * Unit;
			if (lockBackground) BKGPos = BKGPos + d * Unit;
		}
	}

	// #2 Ctrl: add a control point
	CP_Insert = -1;
	if (Ctrl) {
		float d, mind = 1e+8;
		for (int i = 0, n = CP.size(); i < n; i++) {
			d = sdSqLine(p, CP[i], CP[(i + 1) % n]);
			if (d < mind) mind = d, CP_Insert = i;
		}
	}
}

void MouseWheel(int _DELTA) {
	if (Ctrl && showBackground) {
		BKGAlpha = clamp(BKGAlpha + 0.0001*_DELTA, 0.01, 0.99);
		refreshBackground();
		return;
	}
	float s = exp((Alt ? 0.0001 : 0.001)*_DELTA);
	float D = length(vec2(_WIN_W, _WIN_H)), Max = D, Min = 0.02*D;
	if (Unit * s > Max) s = Max / Unit;
	else if (Unit * s < Min) s = Min / Unit;
	if (Shift) {
		if (lockBackground) BKGPos = fromFloat(fromInt(BKGPos) * s);
	}
	else {
		Center = mix(Cursor, Center, s);
		if (lockBackground) BKGPos = (BKGPos - Cursor) * s + Cursor;
	}
	if (lockBackground) BKGUnit /= s;
	Unit *= s;
}

void MouseDownL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	mouse_down = true;
	if (!showControl) return;
	vec2 p = fromInt(Cursor);

	// #1 drag: move a control point
	float r2 = CPR / Unit; r2 *= r2;
	for (unsigned i = 0; i < CP.size(); i++) {
		if ((CP[i] - p).sqr() < r2) {
			CP_Selected = i; return;
		}
	}
}

void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	mouse_down = false;
	if (!showControl) return;
	vec2 p = fromInt(Cursor);

	// #1 drag: move a control point
	float r2 = CPR / Unit; r2 *= r2;
	for (unsigned i = 0; i < CP.size(); i++) {
		if ((CP[i] - p).sqr() < r2) {
			CP_Selected = -1;
		}
	}

	// #2 Ctrl: add a control point
	if (Ctrl && CP_Insert != -1) {
		CP.insert(CP.begin() + CP_Insert + 1, p);
		CP_Selected = -1;
	}
}

void MouseDownR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
}

void MouseUpR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	vec2 p = fromInt(Cursor);
	if (!showControl) return;

	// #3 right click: remove a control point
	if (CP.size() > 3) {
		float r2 = CPR / Unit; r2 *= r2;
		for (unsigned i = 0; i < CP.size(); i++) {
			if ((CP[i] - p).sqr() < r2) {
				CP.erase(CP.begin() + i);
				CP_Selected = -1;
			}
		}
	}
}

void KeyDown(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = true, MouseMove(Cursor.x, Cursor.y);		// call MouseMove to calculate insert position
	else if (_KEY == VK_SHIFT) Shift = true;
	else if (_KEY == VK_MENU) Alt = true;
	else if (_KEY == 'S') S_Key = true;
	if (Ctrl && (_KEY >= 'A' && _KEY <= 'Z')) {
		Ctrl = false;
		if (_KEY == 'S')
			if (!saveFile()) MessageBeep(MB_ICONSTOP);
		if (_KEY == 'O')
			if (!readFile()) MessageBeep(MB_ICONSTOP);
		if (_KEY == 'R')
			if (!selectBackground()) MessageBeep(MB_ICONSTOP);
	}
	if (Shift && (_KEY >= 'A' && _KEY <= 'Z')) {
		vec2 c(0.0);
		for (unsigned i = 0; i < CP.size(); i++) c = c + CP[i];
		c = c * (1.0 / CP.size());
		if (_KEY == 'C') {
			for (unsigned i = 0; i < CP.size(); i++) CP[i] = CP[i] - c;
			showControl = !showControl;
		}
		if (_KEY == 'M')
			for (unsigned i = 0; i < CP.size(); i++) CP[i] = CP[i] - c + fromInt(Cursor);
	}
}

void KeyUp(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = false, CP_Insert = -1;
	else if (_KEY == VK_SHIFT) Shift = false;
	else if (_KEY == VK_MENU) Alt = false;
	else if (_KEY == 'S') S_Key = false;
	if (_KEY == VK_LEFT) CP.insert(CP.begin(), CP[CP.size() - 1]), CP.pop_back();		// switch endpoint
	else if (_KEY == VK_RIGHT) CP.push_back(CP[0]), CP.erase(CP.begin());	// switch endpoint
	else if (_KEY == VK_SPACE) for (int i = 1, n = CP.size(); i <= (n - 1) / 2; i++) std::swap(CP[i], CP[n - i]);	// reverse point direction
	else if (_KEY == VK_TAB) IntpMethod = (Ctrl || Shift) ? Intp((IntpMethod + IntN - 1) % IntN) : Intp((IntpMethod + 1) % IntN);	// previous/next interpolation method
	else if (_KEY == 'B') showBackground = !showBackground;
	else if (_KEY == 'C') showControl = !showControl;
	else if (_KEY == 'L') if (showBackground) lockBackground = !lockBackground;
}



// ======================================== File Operations ========================================

bool saveFile() {
	OPENFILENAME ofn = { sizeof(OPENFILENAME) };
	WCHAR filename[MAX_PATH] = L"";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;
	if (!GetSaveFileName(&ofn)) return false;
	FILE *fp = _wfopen(filename, L"ab");
	if (fp == 0) return false;

	// encode binary data
	fprintf(fp, "NO#EDIT#");
	byte d = 137;
	auto encode = [&](const void* p, int l) {
		for (int i = 0; i < l; d += ((byte*)p)[i] + 13, i++) {
			byte c = ~(53 * (((byte*)p)[i] + d));  // just don't want "xx000000" to appear in text
			byte c0 = c & 0xF, c1 = c >> 4;
			fprintf(fp, "%c%c", c1 > 9 ? c1 + 55 : c1 + '0', c0 > 9 ? c0 + 55 : c0 + '0');
		}
	};
	int n = CP.size(), method = IntpMethod;
	encode(&method, sizeof(method));
	encode(&n, sizeof(n));
	for (int i = 0; i < n; i++) encode(&CP[i], sizeof(vec2));
	fprintf(fp, "\n\n");

	// write text expression
	fprintf(fp, "# %s\n\n", IntpName[method]);

	// control points
	fprintf(fp, "// control points\n");
	fprintf(fp, "vec2 path[%d] = {", n);
	for (int i = 0; i < n; i++) fprintf(fp, " vec2(%f,%f)%c", CP[i].x, CP[i].y, i + 1 == n ? ' ' : ',');
	fprintf(fp, "};\n");

	if (IntpMethod == FourierSeries) {	// print Fourier coefficients
		int N = CP.size() / 2 + 1;
		vec2 *a = new vec2[N], *b = new vec2[N];
		calcFourierParameter(a, b, N);

		// computed Fourier parameters
		fprintf(fp, "\n// computed Fourier parameters\n");
		fprintf(fp, "vec2 a[%d] = { ", N);
		for (int i = 0; i < N; i++) fprintf(fp, "vec2(%f,%f)%c", a[i].x, a[i].y, i + 1 == N ? ' ' : ',');
		fprintf(fp, "};\n");
		fprintf(fp, "vec2 b[%d] = { ", N);
		for (int i = 0; i < N; i++) fprintf(fp, "vec2(%f,%f)%c", b[i].x, b[i].y, i + 1 == N ? ' ' : ',');
		fprintf(fp, "};\n");

		// mathematical expression
		fprintf(fp, "\n// mathematical expression\n");
		fprintf(fp, "(");
		bool sign = false;
		for (int i = 0; i < N; i++) {
			if (abs(a[i].x) > 1e-4) {
				if (sign) fprintf(fp, "%+.4g", a[i].x); else { fprintf(fp, "%.4g", a[i].x); sign = true; }
				if (i != 0) { if (i != 1) fprintf(fp, "cos(%dt)", i); else fprintf(fp, "cos(t)"); }
			}
			if (abs(b[i].x) > 1e-4) {
				if (sign) fprintf(fp, "%+.4g", b[i].x); else { fprintf(fp, "%.4g", b[i].x); sign = true; }
				if (i != 1) fprintf(fp, "sin(%dt)", i); else fprintf(fp, "sin(t)");
			}
		}
		fprintf(fp, ", "); sign = false;
		for (int i = 0; i < N; i++) {
			if (abs(a[i].y) > 1e-4) {
				if (sign) fprintf(fp, "%+.4g", a[i].y); else { fprintf(fp, "%.4g", a[i].y); sign = true; }
				if (i != 0) { if (i != 1) fprintf(fp, "cos(%dt)", i); else fprintf(fp, "cos(t)"); }
			}
			if (abs(b[i].y) > 1e-4) {
				if (sign) fprintf(fp, "%+.4g", b[i].y); else { fprintf(fp, "%.4g", b[i].y); sign = true; }
				if (i != 1) fprintf(fp, "sin(%dt)", i); else fprintf(fp, "sin(t)");
			}
		}
		fprintf(fp, ")\n\n");

		delete a, b;
	}
	else {
		// print svg path
		fprintf(fp, "\n// svg path\n");
		fprintf(fp, "<path transform='matrix(%.1f,0,0,%.1f,%.1f,%.1f)' d='", Unit, -Unit, Center.x, _WIN_H - 1.0 - Center.y);

		auto svg_printf = [&](float x) { if (abs(x) < 1e-4) fprintf(fp, "0"); else fprintf(fp, "%.4g", x); };
		auto svg_printv = [&](vec2 p) { svg_printf(p.x); if (p.y > 1e-4) fputc(',', fp); svg_printf(p.y); };
		auto svg_printv2 = [&](vec2 p, vec2 q) { svg_printv(p); if (q.x > 1e-4) fputc(' ', fp); svg_printv(q); };
		auto svg_printv3 = [&](vec2 p, vec2 q, vec2 r) { svg_printv2(p, q); if (r.x > 1e-4) fputc(' ', fp); svg_printv(r); };

		switch (IntpMethod) {
		case Linear: {
			fputc('M', fp), svg_printv(CP[0]);
			for (int i = 1, n = CP.size(); i < n; i++) fputc('L', fp), svg_printv(CP[i]);
			fputc('Z', fp); break;
		}
		case CatmullRom: {
			fputc('M', fp), svg_printv(CP[0]);
			for (int i = 1, n = CP.size(); i < n; i++) {
				fputc('C', fp);
				vec2 A = CP[(i + n - 1) % n], B = CP[i], C = CP[(i + 1) % n], D = CP[(i + 2) % n];
				spline3 Sp = SpCatmullRom(A, B, C, D);
				svg_printv3(B + Sp.C1 * _13, B + (Sp.C2 + Sp.C1 * 2.0)*_13, C);
			}
			break;
		}
		case QuadraticB: {
			fputc('M', fp), svg_printv((CP[0] + CP.back())*0.5);
			fputc('Q', fp), svg_printv2(CP[0], (CP[0] + CP[1])*0.5);
			for (int i = 1, n = CP.size(); i < n; i++) {
				fputc('T', fp);
				svg_printv((CP[i] + CP[(i + 1) % n])*0.5);
			}
			break;
		}
		case CubicB: {
			fputc('M', fp), svg_printv((CP[0] * 4.0 + CP[1] + CP.back())*_16);
			fputc('C', fp), svg_printv3((CP[0] * 2.0 + CP[1])*_13, (CP[0] + CP[1] * 2.0)*_13, (CP[0] + CP[1] * 4.0 + CP[2])*_16);
			for (int i = 1, n = CP.size(); i < n; i++) {
				fputc('S', fp);
				svg_printv2((CP[i] + CP[(i + 1) % n] * 2.0)*_13, (CP[i] + CP[(i + 1) % n] * 4.0 + CP[(i + 2) % n])*_16);
			}
			break;
		}
		}

		fprintf(fp, "'></path>\n\n");
	}
	fprintf(fp, "\n\n");
	fclose(fp);
	return true;
}

bool readFile() {
	OPENFILENAME ofn = { sizeof(OPENFILENAME) };
	WCHAR filename[MAX_PATH] = L"";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;
	if (!GetOpenFileName(&ofn)) return false;
	FILE *fp = _wfopen(filename, L"rb");
	if (fp == 0) return false;

	char k[8]; if (!fread(k, 1, 8, fp)) return false;
	for (int i = 0; i < 8; i++) if (k[i] != "NO#EDIT#"[i]) return false;
	byte d = 137;
	auto decode = [&](void* p, int l) -> bool {
		for (int i = 0; i < l; i++) {
			byte b[2] = { (byte)fgetc(fp), (byte)fgetc(fp) };
			for (int i = 0; i < 2; i++) {
				if (!((b[i] >= '0' && b[i] <= '9') || (b[i] >= 'A' && b[i] <= 'F'))) return false;	// security check
				b[i] = b[i] > '9' ? b[i] - 55 : b[i] - '0';
			}
			((byte*)p)[i] = 29 * (~((b[0] << 4) | b[1])) - d, d += ((byte*)p)[i] + 13;  // "decrypt" data: byte(29*53) = 1
		}
		return true;
	};
	int n, method;
	if (!decode(&method, sizeof(method))) return false;
	if (!decode(&n, sizeof(n))) return false;
	vec2 *P = new vec2[n];
	for (int i = 0; i < n; i++) {
		if (!decode(&P[i], sizeof(vec2))) return false;
		if (0.0*(P[i].x*P[i].y) != 0.0) return false;
	}
	IntpMethod = (Interpolation)method;
	CP.resize(n);
	for (int i = 0; i < n; i++) CP[i] = P[i];
	delete P;
	return true;
}

