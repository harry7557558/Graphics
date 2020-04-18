/* ==================== User Instructions ====================

 *  Move View:                  drag background
 *  Zoom:                       mouse scroll
 *  Add Shape:                  ctrl + click
 *  Delete Shape(s):            backspace, del
 *  Select/Unselect Shape:      mouse click
 *  Select/Unselect Shapes:     shift + click
 *  Select All:                 ctrl + A
 *  Invert Selection:           ctrl + I
 *  Translate Shape(s):         G, click and drag
 *  Rotate Shape(s):            R
 *  Scale Shape(s):             S
 *  Round Shape(s):             T
 *  Object Layer:               shift + mouse scroll
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
#define MAX_WinW 3840
#define MAX_WinH 2160

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
#define setColor(x,y,col) do{if((x)>=0&&(x)<_WIN_W&&(y)>=0&&(y)<_WIN_H)Canvas(x,y)=col;}while(0)

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { HDC hdc = GetDC(_HWND), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); } break;
	switch (message) {
	case WM_CREATE: { RECT Client; GetClientRect(hWnd, &Client); _WIN_W = Client.right, _WIN_H = Client.bottom; WindowCreate(_WIN_W, _WIN_H); break; }
	case WM_CLOSE: { DestroyWindow(hWnd); WindowClose(); return 0; } case WM_DESTROY: { PostQuitMessage(0); return 0; }
	case WM_MOVE:; case WM_SIZE: {
		RECT Client; GetClientRect(hWnd, &Client); WindowResize(_WIN_W, _WIN_H, Client.right, Client.bottom); _WIN_W = Client.right, _WIN_H = Client.bottom;
		BITMAPINFO bmi; bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32; bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0; bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0;
		if (_HIMG != NULL) DeleteObject(_HIMG); HDC hdc = GetDC(hWnd); _HIMG = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&_WINIMG, NULL, 0); DeleteDC(hdc); _RDBK }
	case WM_GETMINMAXINFO: { LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam; lpMMI->ptMinTrackSize.x = MIN_WinW, lpMMI->ptMinTrackSize.y = MIN_WinH, lpMMI->ptMaxTrackSize.x = MAX_WinW, lpMMI->ptMaxTrackSize.y = MAX_WinH; break; }
	case WM_PAINT: { PAINTSTRUCT ps; HDC hdc = BeginPaint(hWnd, &ps), HMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HMem, _HIMG); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HMem, 0, 0, SRCCOPY); SelectObject(HMem, hbmOld); EndPaint(hWnd, &ps); DeleteDC(HMem), DeleteDC(hdc); break; }
#define _USER_FUNC_PARAMS GET_X_LPARAM(lParam), _WIN_H - 1 - GET_Y_LPARAM(lParam)
	case WM_MOUSEMOVE: { MouseMove(_USER_FUNC_PARAMS); _RDBK }
	case WM_MOUSEWHEEL: { MouseWheel(GET_WHEEL_DELTA_WPARAM(wParam)); _RDBK }
	case WM_LBUTTONDOWN: { SetCapture(hWnd); MouseDownL(_USER_FUNC_PARAMS); _RDBK }
	case WM_LBUTTONUP: { ReleaseCapture(); MouseUpL(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONDOWN: { MouseDownR(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONUP: { MouseUpR(_USER_FUNC_PARAMS); _RDBK }
	case WM_SYSKEYDOWN:; case WM_KEYDOWN: { if (wParam >= 0x08) KeyDown(wParam); _RDBK }
	case WM_SYSKEYUP:; case WM_KEYUP: { if (wParam >= 0x08) KeyUp(wParam); _RDBK }
	} return DefWindowProc(hWnd, message, wParam, lParam);
}
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME); if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindowEx(WS_EX_TOPMOST, _T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	//_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
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
#if 1
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
double Unit = 50.0;		// screen unit to object unit
#define fromInt(p) (((p) - Center) * (1.0 / Unit))
#define fromFloat(p) ((p) * Unit + Center)

// user parameters
vec2 Cursor = vec2(0, 0), clickCursor;
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;
bool Translate = false, Rotate = false, Scale = false, Round = false;
vec2 oldCursor;  // transformation control, world coordinate

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
NTime::time_point _Global_Timer = NTime::now();
#define iTime std::chrono::duration<double>(NTime::now()-_Global_Timer).count()

#pragma endregion


#include <vector>

struct Node {
	vec2 p; double r;
	bool selected = false;
};
double sdNode(vec2 p, const Node& n) {
	return length(p - n.p) - max(n.r, 0.);
}

double newNode_r = 1.0;
std::vector<Node> PL = { Node{vec2(0,0), newNode_r} };

int fittedNode = 0;
int calcClosestNode(vec2 p, bool selected_only = false) {
	int c = -1; double md = 2.0 / Unit;
	for (int i = 0, n = PL.size(); i < n; i++) {
		if (!selected_only || PL[i].selected) {
			double d = sdNode(p, PL[i]);
			if (d < md) c = i, md = d;
		}
	}
	return c;
}
int calcHoverNode(vec2 p) {
	int mm = 2.0 / Unit;
	for (int n = PL.size() - 1; n >= 0; n--) {
		if (sdNode(p, PL[n]) < mm) return n;
	}
	return -1;
}
double calcSDF(vec2 p, bool selected_only = false) {
	double md = INFINITY;
	for (int i = 0, n = PL.size(); i < n; i++) {
		if (!selected_only || PL[i].selected) {
			double d = sdNode(p, PL[i]);
			if (d < md) md = d;
		}
	}
	return md;
}

vec2 calcSelectedCOM() {
	double A = 0; vec2 R(0.0);
	for (int i = 0, n = PL.size(); i < n; i++) {
		if (PL[i].selected && PL[i].r > 0) {
			double a = PI * PL[i].r*PL[i].r;
			A += a, R += PL[i].p*a;
		}
	}
	return R * (1. / A);
}




// ============================================ Rendering ============================================

auto t0 = NTime::now();


#define WHITE 0xFFFFFF
#define RED 0xFF0000
#define YELLOW 0xFFFF00
#define BLUE 0x0000FF
#define GRAY 0x808080
#define LIME 0x00FF00

auto drawLine = [](vec2 p, vec2 q, COLORREF col) {
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
auto drawCross = [&](vec2 p, double r, COLORREF Color) {
	drawLine(p - vec2(r, 0), p + vec2(r, 0), Color);
	drawLine(p - vec2(0, r), p + vec2(0, r), Color);
};
auto drawCircle = [&](vec2 c, double r, COLORREF Color) {
	int s = int(r / sqrt(2) + 0.5);
	int cx = (int)c.x, cy = (int)c.y;
	for (int i = 0, im = min(s, max(_WIN_W - cx, cx)) + 1; i < im; i++) {
		int u = sqrt(r*r - i * i) + 0.5;
		setColor(cx + i, cy + u, Color); setColor(cx + i, cy - u, Color); setColor(cx - i, cy + u, Color); setColor(cx - i, cy - u, Color);
		setColor(cx + u, cy + i, Color); setColor(cx + u, cy - i, Color); setColor(cx - u, cy + i, Color); setColor(cx - u, cy - i, Color);
	}
};
auto fillCircle = [&](vec2 c, double r, COLORREF Color) {
	int x0 = max(0, int(c.x - r)), x1 = min(_WIN_W - 1, int(c.x + r));
	int y0 = max(0, int(c.y - r)), y1 = min(_WIN_H - 1, int(c.y + r));
	int cx = (int)c.x, cy = (int)c.y, r2 = int(r*r);
	for (int x = x0, dx = x - cx; x <= x1; x++, dx++) {
		for (int y = y0, dy = y - cy; y <= y1; y++, dy++) {
			if (dx * dx + dy * dy < r2) Canvas(x, y) = Color;
		}
	}
};
#define drawLineF(p,q,col) drawLine(fromFloat(p),fromFloat(q),col)
#define drawCrossF(p,r,col) drawCross(fromFloat(p),r,col)
#define drawCircleF(p,r,col) drawCircle(fromFloat(p),(r)*Unit,col)
#define fillCircleF(p,r,col) fillCircle(fromFloat(p),(r)*Unit,col)


void render() {
	// debug
	auto t1 = NTime::now();
	double dt = std::chrono::duration<double>(t1 - t0).count();
	dbgprint("[%d×%d] time elapsed: %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, 1000.0*dt, 1. / dt);
	t0 = t1;

	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;

	// axis and grid
	{
		vec2 LB = fromInt(vec2(0, 0)), RT = fromInt(vec2(_WIN_W, _WIN_H));
		COLORREF GridCol = clamp(0x10, (byte)sqrt(10.0*Unit), 0x20); GridCol = GridCol | (GridCol << 8) | (GridCol) << 16;  // adaptive grid color
		for (int y = (int)round(LB.y), y1 = (int)round(RT.y); y <= y1; y++) drawLine(fromFloat(vec2(LB.x, y)), fromFloat(vec2(RT.x, y)), GridCol);  // horizontal gridlines
		for (int x = (int)round(LB.x), x1 = (int)round(RT.x); x <= x1; x++) drawLine(fromFloat(vec2(x, LB.y)), fromFloat(vec2(x, RT.y)), GridCol);  // vertical gridlines
		drawLine(vec2(0, Center.y), vec2(_WIN_W, Center.y), 0x202080);  // x-axis
		drawLine(vec2(Center.x, 0), vec2(Center.x, _WIN_H), 0x202080);  // y-axis
	}

	// draw basic shapes
	for (int i = 0, n = PL.size(); i < n; i++) {
		double r = max(PL[i].r, 0);
		fillCircleF(PL[i].p, r, 0x000080);
		drawCircleF(PL[i].p, r, WHITE);
	}
	// highlight selected shapes
	for (int i = 0, n = PL.size(); i < n; i++) {
		if (PL[i].selected) {
			drawCrossF(PL[i].p, 4, LIME);
			drawCircleF(PL[i].p, max(PL[i].r, 0), LIME);
		}
	}
	// highlight fitted object
	if (fittedNode != -1) {
		drawCrossF(PL[fittedNode].p, 6, LIME);
	}
	// highlight transformation center
	if (Rotate || Scale) {
		drawCrossF(oldCursor, 6, RED);
	}

	vec2 cursor = fromInt(Cursor);
	sprintf(text, "(%.2f,%.2f)", cursor.x, cursor.y);
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

	if (mouse_down) {
		if (calcClosestNode(p0, true) != -1) {
			for (int i = 0, n = PL.size(); i < n; i++) if (PL[i].selected) PL[i].p += d;
		}
		else Center = Center + d * Unit;
	}
	else {
		vec2 po = oldCursor;
		if (Translate) {
			for (int i = 0, n = PL.size(); i < n; i++) if (PL[i].selected) PL[i].p += d;
		}
		else if (Rotate) {
			double s = det(normalize(p0 - po), normalize(p - po)), c = sqrt(1.0 - s * s);
			if (0.0*c == 0.0) {
				vec2 r(s, c), q;
				for (int i = 0, n = PL.size(); i < n; i++) if (PL[i].selected)
					q = PL[i].p - po, PL[i].p = po + vec2(det(q, r), dot(r, q));
			}
		}
		else if (Scale) {
			double sc = length(p - po) / length(p0 - po);
			if (0.0*sc == 0.0) {
				for (int i = 0, n = PL.size(); i < n; i++) if (PL[i].selected) {
					PL[i].p = po + (PL[i].p - po)*sc;
					newNode_r = PL[i].r *= sc;
				}
			}
		}
		else if (Round) {
			double dr = calcSDF(p, true) - calcSDF(p0, true);
			if (0.0*dr == 0.0) {
				for (int i = 0, n = PL.size(); i < n; i++) if (PL[i].selected) {
					PL[i].r += dr;
					newNode_r = max(PL[i].r, 0.01);
				}
			}
		}
	}

	fittedNode = calcHoverNode(p);
}

void MouseWheel(int _DELTA) {
	if (Shift) {
		int c = 0, d = -1;
		for (int i = 0, n = PL.size(); i < n; i++) if (PL[i].selected) c++, d = i;
		if (c == 1) {
			if (_DELTA > 0 && d + 1 < PL.size()) std::swap(PL[d], PL[d + 1]);
			if (_DELTA < 0 && d > 0) std::swap(PL[d], PL[d - 1]);
			return;
		}
	}
	double s = exp(0.001*_DELTA);
	double D = length(vec2(_WIN_W, _WIN_H)), Max = D, Min = 0.02*D;
	if (Unit * s > Max) s = Max / Unit;
	else if (Unit * s < Min) s = Min / Unit;
	Center = mix(Cursor, Center, s);
	Unit *= s;
}

void MouseDownL(int _X, int _Y) {
	clickCursor = Cursor = vec2(_X, _Y);
	mouse_down = true;
	vec2 p = fromInt(Cursor);
}

void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	bool moved = (int)length(clickCursor - Cursor) != 0;
	mouse_down = false;
	if (moved);  // drag
	else {  // click
		if (Translate || Rotate || Scale || Round) {
			Translate = Rotate = Scale = Round = false;
			return;
		}
		vec2 p = fromInt(Cursor);
		if (Ctrl) {
			PL.push_back(Node{ p, newNode_r });
		}
		else {
			if (!Shift) for (int i = 0, n = PL.size(); i < n; i++) PL[i].selected = false;
			if (fittedNode != -1) PL[fittedNode].selected ^= 1;
		}
	}
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
	Translate = Rotate = Scale = Round = false;
	if (_KEY == VK_CONTROL) Ctrl = false;
	else if (_KEY == VK_SHIFT) Shift = false;
	else if (_KEY == VK_MENU) Alt = false;
	else if (_KEY == VK_BACK || _KEY == VK_DELETE) {
		for (int i = 0; i < PL.size(); i++) {
			if (PL[i].selected) PL.erase(PL.begin() + i), i--;
		}
		fittedNode = calcHoverNode(fromInt(Cursor));
	}
	else if (Ctrl) {
		if (_KEY == 'A') for (int i = 0, n = PL.size(); i < n; i++) PL[i].selected = true;
		else if (_KEY == 'I') for (int i = 0, n = PL.size(); i < n; i++) PL[i].selected ^= 1;
	}
	else {
		if (_KEY == VK_ESCAPE || _KEY == VK_RETURN) {
			for (int i = 0, n = PL.size(); i < n; i++) PL[i].selected = false;
			return;
		}
		oldCursor = fromInt(Cursor);
		if (_KEY == 'G') Translate = true;
		else if (_KEY == 'R') Rotate = true;
		else if (_KEY == 'S') Scale = true, oldCursor = calcSelectedCOM();
		else if (_KEY == 'T') Round = true;
		if (Translate || Rotate || Scale || Round) {
			bool m = false;
			for (int i = 0, n = PL.size(); i < n; i++) if (PL[i].selected) { m = true; break; }
			if (!m) Translate = Rotate = Scale = Round = false;
		}
	}
}

