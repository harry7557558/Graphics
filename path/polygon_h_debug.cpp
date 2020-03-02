// debug this header file using Win32 GUI
#include "D:\\polygon.h"

// ========================================= Win32 Standard =========================================

// some compressed copy-and-paste code
#pragma region Windows

#define WIN_NAME "DEMO"
#define WinW_Default 600
#define WinH_Default 400
#define MIN_WinW 400
#define MIN_WinH 300

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>

HWND _HWND; int _WIN_W, _WIN_H;
HBITMAP _HIMG; COLORREF *_WINIMG;	// image
#define Canvas(x,y) _WINIMG[(y)*_WIN_W+(x)]

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
	WNDCLASSEX wc;
	wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance;
	wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME);
	if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL);
	ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) TranslateMessage(&message), DispatchMessage(&message);
	return (int)message.wParam;
}

void render();
void WindowCreate(int _W, int _H); void WindowResize(int _oldW, int _oldH, int _W, int _H); void WindowClose();
void MouseMove(int _X, int _Y); void MouseWheel(int _DELTA); void MouseDownL(int _X, int _Y); void MouseUpL(int _X, int _Y); void MouseDownR(int _X, int _Y); void MouseUpR(int _X, int _Y);
void KeyDown(WPARAM _KEY); void KeyUp(WPARAM _KEY);
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { HDC hdc = GetDC(_HWND), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); \
	render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); } break;
	switch (message) {
	case WM_CREATE: { RECT Client; GetClientRect(hWnd, &Client); _WIN_W = Client.right, _WIN_H = Client.bottom; WindowCreate(_WIN_W, _WIN_H); break; }
	case WM_CLOSE: { DestroyWindow(hWnd); WindowClose(); return 0; }
	case WM_DESTROY: { PostQuitMessage(0); return 0; }
	case WM_MOVE:; case WM_SIZE: {
		RECT Client; GetClientRect(hWnd, &Client);
		WindowResize(_WIN_W, _WIN_H, Client.right, Client.bottom);
		_WIN_W = Client.right, _WIN_H = Client.bottom;
		BITMAPINFO bmi;
		bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32;
		bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0;
		bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0;
		if (_HIMG != NULL) DeleteObject(_HIMG);
		HDC hdc = GetDC(hWnd);
		_HIMG = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&_WINIMG, NULL, 0);
		DeleteDC(hdc);
		_RDBK
	}
	case WM_GETMINMAXINFO: {
		LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam;
		lpMMI->ptMinTrackSize.x = MIN_WinW, lpMMI->ptMinTrackSize.y = MIN_WinH;
		break;
	}
	case WM_PAINT: {
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hWnd, &ps), HMem = CreateCompatibleDC(hdc);
		HBITMAP hbmOld = (HBITMAP)SelectObject(HMem, _HIMG);
		BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HMem, 0, 0, SRCCOPY);
		SelectObject(HMem, hbmOld); EndPaint(hWnd, &ps);
		DeleteDC(HMem), DeleteDC(hdc); break;
	}
#define _USER_FUNC_PARAMS GET_X_LPARAM(lParam), _WIN_H - 1 - GET_Y_LPARAM(lParam)
	case WM_MOUSEMOVE: { MouseMove(_USER_FUNC_PARAMS); _RDBK }
	case WM_LBUTTONDOWN: { SetCapture(hWnd); MouseDownL(_USER_FUNC_PARAMS); _RDBK }
	case WM_LBUTTONUP: { ReleaseCapture(); MouseUpL(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONDOWN: { MouseDownR(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONUP: { MouseUpR(_USER_FUNC_PARAMS); _RDBK }
	case WM_MOUSEWHEEL: { MouseWheel(GET_WHEEL_DELTA_WPARAM(wParam)); _RDBK }
	case WM_SYSKEYDOWN:; case WM_KEYDOWN: { KeyDown(wParam); _RDBK }
	case WM_SYSKEYUP:; case WM_KEYUP: { KeyUp(wParam); _RDBK }
	}
	return DefWindowProc(hWnd, message, wParam, lParam);
}

// debug
#define dbgprint(format, ...) { wchar_t buf[0x4FFF]; swprintf(buf, 0x4FFF, _T(format), ##__VA_ARGS__); OutputDebugStringW(buf); }

#pragma endregion



// ======================================== Data / Parameters ========================================

#include <stdio.h>
#pragma warning(disable: 4996)


#pragma region Global Variables

polygon CP1({ vec2(-1,-1), vec2(.5,-1), vec2(.5,.5), vec2(-1,.5) });
polygon CP2({ vec2(1,1), vec2(-.5,1), vec2(-.5,-.5), vec2(1,-.5) });
int CP1_Selected = -1, CP1_Insert = -1;
int CP2_Selected = -1, CP2_Insert = -1;
vec2 CP1_Center = calcCOM(CP1), CP2_Center = calcCOM(CP2);
double CP1_Dist = NAN, CP2_Dist = NAN;

// window parameters
char text[256];	// window title
vec2 Center = vec2(0, 0);	// origin in screen coordinate
double Unit = 100.0;		// screen unit to object unit
#define fromInt(p) (((p) - Center) * (1.0 / Unit))
#define fromFloat(p) ((p) * Unit + Center)

// rendering parameters
int CPR = 8;	// rendering radius of control point
bool showControl = true;	// control points

// user parameters
vec2 Cursor = vec2(0, 0);
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;

#pragma endregion


#include <fstream>
#include <algorithm>
bool saveFile() {
	std::ofstream os("D:\\ps.txt", std::ios_base::out | std::ios_base::app);
	if (os.fail()) return false;
	printPolygon(os, CP1);
	printPolygon(os, CP2);
	os.close();
	return true;
}



// ============================================ Rendering ============================================

#define DARKBLUE 0x202080
#define DARKGRAY 0x181818
#define WHITE 0xFFFFFF
#define YELLOW 0xFFFF00
#define RED 0xFF0000
#define ORANGE 0xFF8000
#define LIME 0x00FF00
#define MAGENTA 0xFF00FF
#define GRAY 0xA0A0A0

#define byteBlend(a,b,alpha) (byte)(((256-(int)(alpha))*byte(a)+(int)(alpha)*byte(b))>>8)
#define COLORREFBlend(c0,c1,alpha) COLORREF(byteBlend(c0,c1,alpha))|((COLORREF)byteBlend((c0)>>8,(c1)>>8,alpha)<<8)|((COLORREF)byteBlend((c0)>>16,(c1)>>16,alpha)<<16)

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
auto time_0 = NTime::now();

void render() {
	// debug
	auto t0 = NTime::now();

	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;

	// rendering
	auto drawLine = [&](vec2 p, vec2 q, COLORREF col) {  // dda
		vec2 d = q - p;
		double slope = d.y / d.x;
		if (abs(slope) <= 1.0) {
			if (p.x > q.x) std::swap(p, q);
			int x0 = max(0, int(p.x)), x1 = min(_WIN_W - 1, int(q.x)), y;
			double yf = slope * x0 + (p.y - slope * p.x);
			for (int x = x0; x <= x1; x++, yf += slope) {
				y = (int)yf;
				if (y >= 0 && y < _WIN_H) Canvas(x, y) = col;
			}
		}
		else {
			slope = d.x / d.y;
			if (p.y > q.y) std::swap(p, q);
			int y0 = max(0, int(p.y)), y1 = min(_WIN_H - 1, int(q.y)), x;
			double xf = slope * y0 + (p.x - slope * p.y);
			for (int y = y0; y <= y1; y++, xf += slope) {
				x = (int)xf;
				if (x >= 0 && x < _WIN_W) Canvas(x, y) = col;
			}
		}
	};
	auto drawCross = [&](vec2 p, double r, COLORREF col) {
		p = fromFloat(p);
		drawLine(p - vec2(r, 0), p + vec2(r, 0), col);
		drawLine(p - vec2(0, r), p + vec2(0, r), col);
	};
	auto drawCircle = [&](vec2 c, double r, COLORREF col) {
		int x0 = max(0, int(c.x - r)), x1 = min(_WIN_W - 1, int(c.x + r));
		int y0 = max(0, int(c.y - r)), y1 = min(_WIN_H - 1, int(c.y + r));
		int cx = (int)c.x, cy = (int)c.y, r2 = int(r*r);
		for (int x = x0, dx = x - cx; x <= x1; x++, dx++) {
			for (int y = y0, dy = y - cy; y <= y1; y++, dy++) {
				if (dx * dx + dy * dy < r2) Canvas(x, y) = col;
			}
		}
	};
	auto drawPolygon = [&](const polygon &p, COLORREF Stroke, COLORREF Fill) {
		int n = p.size(), y0 = _WIN_H - 1, y1 = 0;
		polygon P = p; for (int i = 0; i < n; i++) P[i] = fromFloat(p[i]);
		for (int i = 0; i < n; i++) {
			if ((int)P[i].y < y0) y0 = (int)P[i].y; if ((int)P[i].y > y1) y1 = (int)P[i].y;
		}
		if (y0 < 0) y0 = 0; if (y1 >= _WIN_H) y1 = _WIN_H - 1;

		// scan-line filling - I think it has a lot of room for optimization
		int alpha = Fill >> 24;
		if (alpha != 0) {
			std::vector<int> k;
			for (int y = y0; y <= y1; y++) {
				k.clear();
				for (int i = 0; i < n; i++) {
					if (y > P[i].y != y > P[(i + 1) % n].y) {
						double t = (y - P[i].y) / (P[(i + 1) % n].y - P[i].y);
						k.push_back((1 - t)*P[i].x + t * P[(i + 1) % n].x);
					}
				}
				std::sort(k.begin(), k.end());
				for (int i = 1; i < k.size(); i += 2) {
					for (int x = max(k[i - 1], 0), x1 = min(k[i], _WIN_W - 1); x <= x1; x++) {
						Canvas(x, y) = COLORREFBlend(Canvas(x, y), Fill, alpha);
					}
				}
			}
		}
		// stroke
		for (int i = 0; i < n; i++) drawLine(P[i], P[(i + 1) % n], Stroke);
	};

	// grid and axis
	vec2 LB = fromInt(vec2(0, 0)), RT = fromInt(vec2(_WIN_W, _WIN_H));
	COLORREF GridCol = clamp(0x10, (byte)sqrt(10.0*Unit), 0x20); GridCol = GridCol | (GridCol << 8) | (GridCol) << 16;  // adaptive grid color
	for (int y = (int)round(LB.y), y1 = (int)round(RT.y); y <= y1; y++)
		drawLine(fromFloat(vec2(LB.x, y)), fromFloat(vec2(RT.x, y)), GridCol);  // horizontal gridlines
	for (int x = (int)round(LB.x), x1 = (int)round(RT.x); x <= x1; x++)
		drawLine(fromFloat(vec2(x, LB.y)), fromFloat(vec2(x, RT.y)), GridCol);  // vertical gridlines
	drawLine(vec2(0, Center.y), vec2(_WIN_W, Center.y), DARKBLUE);  // x-axis
	drawLine(vec2(Center.x, 0), vec2(Center.x, _WIN_H), DARKBLUE);  // y-axis

	// polygons
	//drawPolygon(CP1, CP1_Dist < CP2_Dist ? WHITE : GRAY);
	//drawPolygon(CP2, CP1_Dist > CP2_Dist ? WHITE : GRAY);
	drawPolygon(CP1, isSelfIntersecting(CP1) ? RED : CP1_Dist < CP2_Dist ? WHITE : GRAY, 0x20FF8000);
	drawPolygon(CP2, isSelfIntersecting(CP2) ? RED : CP1_Dist > CP2_Dist ? WHITE : GRAY, 0x20FF8000);

	// center of polygons
	CP1_Center = calcCOM(CP1), CP2_Center = calcCOM(CP2);
	if (Shift || Alt || mouse_down) {
		drawCross(calcCenter(CP1), 6, MAGENTA), drawCross(calcCenter(CP2), 6, MAGENTA);
		drawCross(CP1_Center, 6, LIME), drawCross(CP2_Center, 6, LIME);
	}

	// control points
	if (showControl) {
		int n1 = CP1.size(), n2 = CP2.size();
		for (int i = n1 - 1; i >= 0; i--) {
			vec2 P = fromFloat(CP1[i]);
			if (i == CP1_Selected) drawCircle(P, CPR, YELLOW);
			else drawCircle(P, CPR - min(i, 2), CP1_Insert != -1 && (i == CP1_Insert || i == (CP1_Insert + 1) % n1) ? LIME : RED);
		}
		for (int i = n2 - 1; i >= 0; i--) {
			vec2 P = fromFloat(CP2[i]);
			if (i == CP2_Selected) drawCircle(P, CPR, YELLOW);
			else drawCircle(P, CPR - min(i, 2), CP2_Insert != -1 && (i == CP2_Insert || i == (CP2_Insert + 1) % n2) ? LIME : RED);
		}
	}

	auto t1 = NTime::now();
	double dt = std::chrono::duration<double>(t1 - t0).count();
	double fps = 1.0 / std::chrono::duration<double>(t1 - time_0).count();
	time_0 = t0;
	vec2 cursor = fromInt(Cursor);
	sprintf(text, "P1=%.2lf, P2=%.2lf, A1=%.3lf, A2=%.3lf    (%.2lf,%.2lf)    %.2lfms (%.1lffps)",
		calcPerimeter(CP1), calcPerimeter(CP2), calcArea(CP1), calcArea(CP2), cursor.x, cursor.y, 1000.0*dt, fps);
	SetWindowTextA(_HWND, text);
}


// ============================================== User ==============================================


void WindowCreate(int _W, int _H) {
	Center = vec2(_W, _H) * 0.5;

	// test perfermance
	/*{
		vec2 a[21] = { vec2(0.285528,0.441513),vec2(-0.120229,-0.827634),vec2(0.362113,-0.659311),vec2(-0.418880,-0.131927),vec2(-0.011258,0.239604),vec2(0.046156,-0.084705),vec2(0.251175,0.175976),vec2(-0.109171,-0.075648),vec2(-0.021327,0.036144),vec2(-0.084662,-0.028716),vec2(-0.036200,-0.025998),vec2(0.081147,-0.067187),vec2(0.041652,-0.070299),vec2(0.073204,-0.015663),vec2(-0.012340,-0.029848),vec2(-0.075957,0.008711),vec2(-0.043883,-0.027078),vec2(0.043032,-0.074274),vec2(-0.028872,-0.046507),vec2(-0.029267,-0.010998),vec2(0.018142,-0.073762) },
			b[21] = { vec2(0.000000,0.000000),vec2(0.895598,0.180168),vec2(0.373547,0.372804),vec2(0.088225,0.004491),vec2(-0.049367,-0.524011),vec2(-0.211382,0.153020),vec2(0.042170,-0.053641),vec2(-0.120638,-0.029826),vec2(0.108581,0.017927),vec2(0.043941,0.001481),vec2(0.034297,0.046155),vec2(0.014269,0.052746),vec2(0.001654,0.079682),vec2(-0.073158,0.020828),vec2(-0.021989,0.080393),vec2(0.010579,0.023657),vec2(0.086257,-0.018580),vec2(0.074851,0.002612),vec2(-0.054539,0.001691),vec2(0.008488,-0.007624),vec2(0.000000,0.000000) };
		CP1.clear(), CP2.clear();
		for (double t = 0; t < 2.0*PI; t += 0.03) {
			vec2 v(-1.0);
			for (int i = 0; i < 21; i++) v = v + a[i] * cos(i*t) + b[i] * sin(i*t);
			CP1.push_back(v + vec2(sin(7393.43*t), cos(9972.56*t))*0.05), CP2.push_back(-v + vec2(cos(6231.97*t), sin(8622.36*t))*0.05);
		}
	}*/
}
void WindowResize(int _oldW, int _oldH, int _W, int _H) {
	if (_W*_H == 0 || _oldW * _oldH == 0) return;
	double pw = _oldW, ph = _oldH, w = _W, h = _H;
	double s = sqrt((w * h) / (pw * ph));
	Unit *= s;
	Center.x *= w / pw, Center.y *= h / ph;
}
void WindowClose() {
	return;
}

void MouseMove(int _X, int _Y) {
	vec2 P0 = Cursor, P = vec2(_X, _Y);
	Cursor = P;
	vec2 p0 = fromInt(P0), p = fromInt(P), d = p - p0;
	CP1_Insert = CP2_Insert = -1;

	if (mouse_down) {
		if (Alt) {  // drag object
			if (CP1_Dist < CP2_Dist) for (unsigned i = 0; i < CP1.size(); i++) CP1[i] = CP1[i] + d;
			else for (unsigned i = 0; i < CP2.size(); i++) CP2[i] = CP2[i] + d;
		}
		else if (showControl && CP1_Selected != -1 || CP2_Selected != -1) {  // drag control point
			if (CP1_Dist < CP2_Dist && CP1_Selected != -1) CP1[CP1_Selected] = CP1[CP1_Selected] + d;
			else if (CP2_Selected != -1) CP2[CP2_Selected] = CP2[CP2_Selected] + d;
		}
		else if (Shift) {  // rotate shape
			CP1_Center = calcCOM(CP1), CP2_Center = calcCOM(CP2);
			auto rotate = [&](polygon &CP, vec2 C) {
				vec2 R = vec2(det(normalize(p0 - C), normalize(p - C)), dot(normalize(p0 - C), normalize(p - C)));
				for (int i = 0; i < CP.size(); i++) {
					CP[i] = CP[i] - C;
					CP[i] = vec2(det(CP[i], R), dot(R, CP[i])) + C;
				}
			};
			if (CP1_Dist < CP2_Dist) rotate(CP1, CP1_Center);
			else rotate(CP2, CP2_Center);
		}
		else {	// coordinate
			Center = Center + d * Unit;
		}
	}

	if (!Alt && !Shift && CP1_Selected == -1 && CP2_Selected == -1) {  // refresh distance to polygons
		CP1_Insert = CP2_Insert = -1, CP1_Dist = CP2_Dist = 1e8;
		for (int i = 0, n = CP1.size(); i < n; i++) {
			double d = sdSqLine(p, CP1[i], CP1[(i + 1) % n]);
			if (d < CP1_Dist) CP1_Dist = d, CP1_Insert = i;
		}
		for (int i = 0, n = CP2.size(); i < n; i++) {
			double d = sdSqLine(p, CP2[i], CP2[(i + 1) % n]);
			if (d < CP2_Dist) CP2_Dist = d, CP2_Insert = i;
		}
		if (Ctrl) {  // calculate fitted point
			if (CP1_Dist < CP2_Dist) CP2_Insert = -1;
			else CP1_Insert = -1;
		}
		else CP1_Insert = CP2_Insert = -1;
	}
}

void MouseWheel(int _DELTA) {
	if (Ctrl) {
		if (_DELTA > 0) CPR++;
		else if (_DELTA < 0 && CPR > 4) CPR--;
		return;
	}
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
	if (!showControl) return;
	vec2 p = fromInt(Cursor);

	// #1 drag: move a control point
	double r2 = CPR / Unit; r2 *= r2;
	for (unsigned i = 0; i < CP1.size(); i++) {
		if ((CP1[i] - p).sqr() < r2) { CP1_Selected = i; return; }
	}
	for (unsigned i = 0; i < CP2.size(); i++) {
		if ((CP2[i] - p).sqr() < r2) { CP2_Selected = i; return; }
	}
}

void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	mouse_down = false;
	if (!showControl) return;
	vec2 p = fromInt(Cursor);

	// #1 drag: move a control point
	CP1_Selected = CP2_Selected = -1;

	// #2 Ctrl: add a control point
	if (Ctrl) {
		if (CP1_Insert != -1) CP1.insert(CP1.begin() + CP1_Insert + 1, p), CP1_Selected = -1;
		if (CP2_Insert != -1) CP2.insert(CP2.begin() + CP2_Insert + 1, p), CP2_Selected = -1;
	}
}

void MouseDownR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
}

void MouseUpR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	vec2 p = fromInt(Cursor);
	if (!showControl) return;

	// #3 rightclick: remove a control point
	if (CP1_Dist < CP2_Dist && CP1.size() > 3) {
		double r2 = CPR / Unit; r2 *= r2;
		for (unsigned i = 0; i < CP1.size(); i++) {
			if ((CP1[i] - p).sqr() < r2) {
				CP1.erase(CP1.begin() + i);
				CP1_Selected = -1; return;
			}
		}
	}
	else if (CP2.size() > 3) {
		double r2 = CPR / Unit; r2 *= r2;
		for (unsigned i = 0; i < CP2.size(); i++) {
			if ((CP2[i] - p).sqr() < r2) {
				CP2.erase(CP2.begin() + i);
				CP2_Selected = -1; return;
			}
		}
	}
}

void KeyDown(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = true, MouseMove(Cursor.x, Cursor.y);		// call MouseMove to calculate insert position
	else if (_KEY == VK_SHIFT) Shift = true;
	else if (_KEY == VK_MENU) Alt = true;
	if (Ctrl && (_KEY >= 'A' && _KEY <= 'Z')) {
		Ctrl = false;
		if (_KEY == 'S') {
			if (!saveFile()) MessageBeep(MB_ICONERROR);
		}
	}
}

void KeyUp(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = false, CP1_Insert = CP2_Insert = -1;
	else if (_KEY == VK_SHIFT) Shift = false;
	else if (_KEY == VK_MENU) Alt = false;
	else if (_KEY == VK_LEFT) {		// shift endpoint
		if (CP1_Dist < CP2_Dist) CP1.insert(CP1.begin(), CP1[CP1.size() - 1]), CP1.pop_back();
		else CP2.insert(CP2.begin(), CP2[CP2.size() - 1]), CP2.pop_back();
	}
	else if (_KEY == VK_RIGHT) {	// shift endpoint
		if (CP1_Dist < CP2_Dist) CP1.push_back(CP1[0]), CP1.erase(CP1.begin());
		else CP2.push_back(CP2[0]), CP2.erase(CP2.begin());
	}
	else if (_KEY == VK_SPACE) {	// reverse point direction
		if (CP1_Dist < CP2_Dist) for (int i = 1, n = CP1.size(); i <= (n - 1) / 2; i++) std::swap(CP1[i], CP1[n - i]);
		else for (int i = 1, n = CP2.size(); i <= (n - 1) / 2; i++) std::swap(CP2[i], CP2[n - i]);
	}
	else if (_KEY == 'C') showControl = !showControl;
}

