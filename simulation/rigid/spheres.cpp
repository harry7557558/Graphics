
#include <cmath>
#include <stdio.h>
#include <algorithm>

// ========================================= Win32 Standard =========================================

#pragma region Windows

#ifndef UNICODE
#define UNICODE
#endif

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>



#pragma region Window Macros / Forward Declarations


#define WIN_NAME "3D GUI Template"
#define WinW_Padding 100
#define WinH_Padding 100
#define WinW_Default 640
#define WinH_Default 400
#define WinW_Min 400
#define WinH_Min 300
#define WinW_Max 3840
#define WinH_Max 2160

void Init();  // called before window is created
void render();
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

double _DEPTHBUF[WinW_Max][WinH_Max];  // how you use this depends on you

bool Render_Needed = true;

#pragma endregion  // Windows global variables and forward declarations


// Win32 Entry

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { if (!Render_Needed) break; HDC hdc = GetDC(hWnd), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); Render_Needed = false; break; }
	switch (message) {
	case WM_NULL: { _RDBK }
	case WM_CREATE: { if (!_HWND) Init(); break; } case WM_CLOSE: { WindowClose(); DestroyWindow(hWnd); return 0; } case WM_DESTROY: { PostQuitMessage(0); return 0; }
	case WM_MOVE:; case WM_SIZE: { RECT Client; GetClientRect(hWnd, &Client); WindowResize(_WIN_W, _WIN_H, Client.right, Client.bottom); _WIN_W = Client.right, _WIN_H = Client.bottom; BITMAPINFO bmi; bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32; bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0; bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0; if (_HIMG != NULL) DeleteObject(_HIMG); HDC hdc = GetDC(hWnd); _HIMG = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&_WINIMG, NULL, 0); DeleteDC(hdc); _RDBK }
	case WM_GETMINMAXINFO: { LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam; lpMMI->ptMinTrackSize.x = WinW_Min, lpMMI->ptMinTrackSize.y = WinH_Min, lpMMI->ptMaxTrackSize.x = WinW_Max, lpMMI->ptMaxTrackSize.y = WinH_Max; break; }
	case WM_PAINT: { PAINTSTRUCT ps; HDC hdc = BeginPaint(hWnd, &ps), HMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HMem, _HIMG); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HMem, 0, 0, SRCCOPY); SelectObject(HMem, hbmOld); EndPaint(hWnd, &ps); DeleteDC(HMem), DeleteDC(hdc); break; }
#define _USER_FUNC_PARAMS GET_X_LPARAM(lParam), _WIN_H - 1 - GET_Y_LPARAM(lParam)
	case WM_MOUSEMOVE: { MouseMove(_USER_FUNC_PARAMS); _RDBK } case WM_MOUSEWHEEL: { MouseWheel(GET_WHEEL_DELTA_WPARAM(wParam)); _RDBK }
	case WM_LBUTTONDOWN: { SetCapture(hWnd); MouseDownL(_USER_FUNC_PARAMS); _RDBK } case WM_LBUTTONUP: { ReleaseCapture(); MouseUpL(_USER_FUNC_PARAMS); _RDBK }
	case WM_RBUTTONDOWN: { MouseDownR(_USER_FUNC_PARAMS); _RDBK } case WM_RBUTTONUP: { MouseUpR(_USER_FUNC_PARAMS); _RDBK }
	case WM_SYSKEYDOWN:; case WM_KEYDOWN: { if (wParam >= 0x08) KeyDown(wParam); _RDBK } case WM_SYSKEYUP:; case WM_KEYUP: { if (wParam >= 0x08) KeyUp(wParam); _RDBK }
	} return DefWindowProc(hWnd, message, wParam, lParam);
}
#if 1
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
#else
int main() {
	HINSTANCE hInstance = NULL; int nCmdShow = SW_RESTORE; if (!_USE_CONSOLE) FreeConsole();
#endif
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME); if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, WinW_Padding, WinH_Padding, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL); ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

#pragma endregion  // WIN32


// ================================== Vector Classes/Functions ==================================

#pragma region Vector & Matrix

#include "numerical/geometry.h"
const vec3 vec0(0, 0, 0), veci(1, 0, 0), vecj(0, 1, 0), veck(0, 0, 1);
#define SCRCTR vec2(0.5*_WIN_W,0.5*_WIN_H)

// 4x4 matrix
struct Affine {
	vec3 u, v, w;  // first row, second row, third row
	vec3 t, p;  // translation, perspective
	double s;  // scaling
};
vec3 operator * (Affine T, vec3 p) {
	vec3 q = vec3(dot(T.u, p), dot(T.v, p), dot(T.w, p)) + T.t;
	double d = 1.0 / (dot(T.p, p) + T.s);
	return d < 0.0 ? vec3(NAN) : q * d;
	//return q * d;
}
Affine operator * (const Affine &A, const Affine &B) {
	Affine R;
	R.u = A.u.x*B.u + A.u.y*B.v + A.u.z*B.w + A.t.x*B.p;
	R.v = A.v.x*B.u + A.v.y*B.v + A.v.z*B.w + A.t.y*B.p;
	R.w = A.w.x*B.u + A.w.y*B.v + A.w.z*B.w + A.t.z*B.p;
	R.t = vec3(dot(A.u, B.t), dot(A.v, B.t), dot(A.w, B.t)) + A.t*B.s;
	R.p = vec3(A.p.x*B.u.x + A.p.y*B.v.x + A.p.z*B.w.x, A.p.x*B.u.y + A.p.y*B.v.y + A.p.z*B.w.y, A.p.x*B.u.z + A.p.y*B.v.z + A.p.z*B.w.z) + B.p*A.s;
	R.s = dot(A.p, B.t) + A.s*B.s;
	return R;
}


#pragma endregion  // Vector & Matrix



// ======================================== Data / Parameters ========================================


#pragma region General Global Variables

// viewport
vec3 Center(0.0, 0.0, 0.0);  // view center in world coordinate
double rz = 0.2*PI, rx = 0.15*PI, ry = 0.0, dist = 12.0, Unit = 100.0;  // yaw, pitch, roll, camera distance, scale to screen

// window parameters
Affine Tr;  // matrix
vec3 CamP, ScrO, ScrA, ScrB;  // camera and screen
auto scrDir = [](vec2 pixel) { return normalize(ScrO + (pixel.x / _WIN_W)*ScrA + (pixel.y / _WIN_H)*ScrB - CamP); };

// user parameters
vec2 Cursor = vec2(0, 0), clickCursor;  // current cursor and cursor position when mouse down
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;

#pragma endregion Camera/Screen, Mouse/Key


#pragma region Global Variable Related Functions

// projection - matrix vs camera/screen
Affine axisAngle(vec3 axis, double a) {
	axis = normalize(axis); double ct = cos(a), st = sin(a);
	return Affine{
		vec3(ct + axis.x*axis.x*(1 - ct), axis.x*axis.y*(1 - ct) - axis.z*st, axis.x*axis.z*(1 - ct) + axis.y*st),
		vec3(axis.y*axis.x*(1 - ct) + axis.z*st, ct + axis.y*axis.y*(1 - ct), axis.y*axis.z*(1 - ct) - axis.x*st),
		vec3(axis.z*axis.x*(1 - ct) - axis.y*st, axis.z*axis.y*(1 - ct) + axis.x*st, ct + axis.z*axis.z*(1 - ct)),
		vec3(0), vec3(0), 1.0
	};
}
void calcMat() {
	double cx = cos(rx), sx = sin(rx), cz = cos(rz), sz = sin(rz), cy = cos(ry), sy = sin(ry);
	Affine D{ veci, vecj, veck, -Center, vec3(0), 1.0 };  // world translation
	Affine R{ vec3(-sz, cz, 0), vec3(-cz * sx, -sz * sx, cx), vec3(-cz * cx, -sz * cx, -sx), vec3(0), vec3(0), 1.0 };  // rotation
	R = Affine{ vec3(cy, -sy, 0), vec3(sy, cy, 0), vec3(0, 0, 1), vec3(0), vec3(0), 1.0 } *R;  // camera roll (ry)
	Affine P{ veci, vecj, veck, vec3(0), vec3(0, 0, 1.0 / dist), 1.0 };  // perspective
	Affine S{ veci, vecj, veck, vec3(0), vec3(0), 1.0 / Unit };  // scaling
	Affine T{ veci, vecj, veck, vec3(SCRCTR, 0.0), vec3(0), 1.0 };  // screen translation
	Tr = T * S * P * R * D;
}

#pragma endregion


// ============================================ Rendering ============================================

#pragma region Rasterization functions

typedef unsigned char byte;
COLORREF toCOLORREF(vec3 c) {
	COLORREF r = 0; byte *k = (byte*)&r;
	k[0] = byte(255 * clamp(c.z, 0, 1));
	k[1] = byte(255 * clamp(c.y, 0, 1));
	k[2] = byte(255 * clamp(c.x, 0, 1));
	return r;
}

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
}
void drawCross(vec2 p, double r, COLORREF Color = 0xFFFFFF) {
	drawLine(p - vec2(r, 0), p + vec2(r, 0), Color);
	drawLine(p - vec2(0, r), p + vec2(0, r), Color);
}
void drawCircle(vec2 c, double r, COLORREF Color) {
	int s = int(r / sqrt(2) + 0.5);
	int cx = (int)c.x, cy = (int)c.y;
	for (int i = 0, im = min(s, max(_WIN_W - cx, cx)) + 1; i < im; i++) {
		int u = (int)(sqrt(r*r - i * i) + 0.5);
		setColor(cx + i, cy + u, Color); setColor(cx + i, cy - u, Color); setColor(cx - i, cy + u, Color); setColor(cx - i, cy - u, Color);
		setColor(cx + u, cy + i, Color); setColor(cx + u, cy - i, Color); setColor(cx - u, cy + i, Color); setColor(cx - u, cy - i, Color);
	}
}
void fillCircle(vec2 c, double r, COLORREF Color) {
	int x0 = max(0, int(c.x - r)), x1 = min(_WIN_W - 1, int(c.x + r));
	int y0 = max(0, int(c.y - r)), y1 = min(_WIN_H - 1, int(c.y + r));
	int cx = (int)c.x, cy = (int)c.y, r2 = int(r*r);
	for (int x = x0, dx = x - cx; x <= x1; x++, dx++) {
		for (int y = y0, dy = y - cy; y <= y1; y++, dy++) {
			if (dx * dx + dy * dy < r2) Canvas(x, y) = Color;
		}
	}
}
void drawTriangle(vec3 A, vec3 B, vec3 C, COLORREF col, bool stroke = false, COLORREF strokecol = 0xFFFFFF) {
	vec2 a = A.xy(), b = B.xy(), c = C.xy();
	vec3 ab = B - A, bc = C - B, ca = A - C, n = cross(ab, ca);
	double k = 1.0 / det(ca.xy(), ab.xy());
	int x0 = max((int)(min(min(a.x, b.x), c.x) + 1), 0), x1 = min((int)(max(max(a.x, b.x), c.x) + 1), _WIN_W);
	int y0 = max((int)(min(min(a.y, b.y), c.y) + 1), 0), y1 = min((int)(max(max(a.y, b.y), c.y) + 1), _WIN_H);
	for (int i = y0; i < y1; i++) {
		for (int j = x0; j < x1; j++) {
			vec2 p(j, i);
			vec2 ap = p - a, bp = p - b, cp = p - c;
			if (((det(ap, bp) < 0) + (det(bp, cp) < 0) + (det(cp, ap) < 0)) % 3 == 0) {
				double z = k * dot(n, vec3(p.x, p.y, 0.) - A);
				if (z < _DEPTHBUF[j][i]) {
					Canvas(j, i) = col;
					_DEPTHBUF[j][i] = z;
				}
			}
		}
	}
	if (stroke) {
		drawLine(a, b, strokecol); drawLine(a, c, strokecol); drawLine(b, c, strokecol);
	}
}

void drawLine_F(vec3 A, vec3 B, COLORREF col = 0xFFFFFF) {
	double u = dot(Tr.p, A) + Tr.s, v = dot(Tr.p, B) + Tr.s;
	if (u > 0 && v > 0) { drawLine((Tr*A).xy(), (Tr*B).xy(), col); return; }
	if (u < 0 && v < 0) return;
	if (u < v) std::swap(A, B), std::swap(u, v);
	double t = u / (u - v) - 1e-6;
	B = A + (B - A)*t;
	drawLine((Tr*A).xy(), (Tr*B).xy(), col);
}
void drawTriangle_F(vec3 A, vec3 B, vec3 C, COLORREF col) {
	double u = dot(Tr.p, A) + Tr.s, v = dot(Tr.p, B) + Tr.s, w = dot(Tr.p, C) + Tr.s;
	if (u > 0 && v > 0 && w > 0) { drawTriangle((Tr*A), (Tr*B), (Tr*C), col); return; }
}
void drawCross3D(vec3 P, double r, COLORREF col = 0xFFFFFF) {
	r *= dot(Tr.p, P) + Tr.s;
	drawLine_F(P - vec3(r, 0, 0), P + vec3(r, 0, 0), col);
	drawLine_F(P - vec3(0, r, 0), P + vec3(0, r, 0), col);
	drawLine_F(P - vec3(0, 0, r), P + vec3(0, 0, r), col);
}


#pragma endregion


// ============================================ Simulation ============================================


#pragma region Rigid body simulation


#include <vector>
#include <string>

#include "UI/3d_reader.h"
COLORREF toCOLORREF(vec3f c) {
	COLORREF r = 0; byte *k = (byte*)&r;
	k[0] = byte(255.99f * clamp(c.z, 0.f, 1.f));
	k[1] = byte(255.99f * clamp(c.y, 0.f, 1.f));
	k[2] = byte(255.99f * clamp(c.x, 0.f, 1.f));
	return r;
}



mat3 cross(vec3 w, mat3 R) {
	return mat3(cross(w, R.column(0)), cross(w, R.column(1)), cross(w, R.column(2)));
}

struct quaternion {
	double s; vec3 t;
	quaternion operator * (double a) const {
		return quaternion{ s*a, t*a };
	}
	quaternion operator * (quaternion q) const {
		return quaternion{ s*q.s - dot(t, q.t), s*q.t + q.s*t + cross(t, q.t) };
	}
};

#include "numerical/ode.h"

// sphere with orientation
struct RigidBody {

	/* constant quantities */
	static double r;  // radius
	static double m, inv_m;  // mass and its reciprocal
	static mat3 I0, inv_I0;  // moment of inertia when right oriented
	vec3f color;

	/* state variables */
	static double t;  // time
	vec3 x;  // position
	quaternion q;  // orientation
	vec3 P;  // momentum
	vec3 L;  // angular momentum

	/* derived quantities */
	mat3 R;  // orientation
	mat3 I, inv_I;  // moment of inertia
	vec3 v;  // velocity
	vec3 w;  // angular velocity

	/* computed quantities */
	vec3 force;
	vec3 torque;

	vec3 getAbsolutePos(vec3 r) const {
		return x + R * r;
	}
	vec3 getAbsoluteVelocity(vec3 r, bool rotated = false) const {
		return v + cross(w, rotated ? r : R * r);
	}

	/* constructors */
	RigidBody() {}
	RigidBody(vec3 x, quaternion q, vec3 v, vec3 w, vec3f color = vec3f(1.0)) : x(x), q(q), color(color) {
		calcDerivedQuantities();
		P = m * v, L = I * w;
	}

	// calculate mass and moment of inertia
	static void calcConstants(double density, double radius);

	// set variables to default
	void calcDerivedQuantities() {
		// quaternion to rotation matrix
		double s = q.s, x = q.t.x, y = q.t.y, z = q.t.z;
		R = mat3(
			1.0 - 2.0*(y*y + z * z), 2.0*(x*y + s * z), 2.0*(x*z - s * y),
			2.0*(x*y - s * z), 1.0 - 2.0*(x*x + z * z), 2.0*(y*z + s * x),
			2.0*(x*z + s * y), 2.0*(y*z - s * x), 1.0 - 2.0*(x*x + y * y)
		);
		// moment of inertia, linear and angular velocity
		I = R * I0 * transpose(R);
		inv_I = R * inv_I0 * transpose(R);
		v = inv_m * P;
		w = inv_I * L;
	}
	void resetVariables() {
		t = 0.0, x = vec3(0.0), q = quaternion{ 1.0, vec3(0, 0, 0) }, P = vec3(0.0), L = vec3(0.0);
		calcDerivedQuantities();
		force = vec3(0.0), torque = vec3(0.0);
	}

	// conversion between state and vector
	static int vectorSize() {
		return 3 + 4 + 3 + 3;  // 13
	}
	void toVector(double *vec) const {
		*(vec3*)(vec + 0) = this->x;
		*(quaternion*)(vec + 3) = this->q;
		*(vec3*)(vec + 7) = this->P;
		*(vec3*)(vec + 10) = this->L;
	}
	void toVectorDerivative(double *vec) const {
		*(vec3*)(vec + 0) = this->v;
		*(quaternion*)(vec + 3) = quaternion{ 0, this->w } *this->q * 0.5;
		*(vec3*)(vec + 7) = this->force;
		*(vec3*)(vec + 10) = this->torque;
	}
	void fromVector(const double *vec) {
		this->x = *(vec3*)(vec + 0);
		this->q = *(quaternion*)(vec + 3);
		this->P = *(vec3*)(vec + 7);
		this->L = *(vec3*)(vec + 10);
		if (1) q = q * (1.0 / sqrt(q.s*q.s + q.t.sqr()));
	}

};

double RigidBody::r = 0.0;
double RigidBody::m = 0.0; double RigidBody::inv_m = INFINITY;
mat3 RigidBody::I0 = mat3(0.0); mat3 RigidBody::inv_I0 = mat3(INFINITY);
double RigidBody::t = 0.0;

void RigidBody::calcConstants(double density, double radius) {
	r = radius;
	m = 1.333333 * PI * r*r*r * density, inv_m = 1.0 / m;
	I0 = mat3(0.4*m*r*r), inv_I0 = inverse(I0);
}


vec3 B0, B1;  // simulation box
const bool has_top = true;

std::vector<RigidBody> bodies;

const vec3 g = vec3(0, 0, -9.8);


void calc_force_and_torque() {
	const int BN = (int)bodies.size();

	// gravity
	for (int i = 0; i < BN; i++) {
		bodies[i].force = bodies[i].m * g;
		bodies[i].torque = vec3(0.0);
	}

	const double k_c = 10000.0 * RigidBody::m;  // collision force coefficient
	const double k_d = 50.0 * RigidBody::m;  // damping coefficient
	const double mu_b = 0.2;  // friction coefficient between sphere and boundary
	const double mu_s = 0.1;  // friction coefficient between spheres

	// boundary collision
	auto addCollisionForce = [&](RigidBody &body, vec3 n, vec3 p0) {
		double depth = -dot(n, body.x - p0) + body.r;
		if (depth > 0.0) {
			vec3 f_c = k_c * depth * n;
			vec3 f_d = -k_d * dot(body.v, n) * n;  // not sure if this is physically correct
			vec3 v_f = body.getAbsoluteVelocity(-body.r*n, true);
			vec3 f_f = -mu_b * (k_c * depth) * normalize(v_f - dot(v_f, n) * n);
			if (isnan(f_f.x)) f_f = vec3(0.0);
			body.force += f_c + f_d + f_f;
			body.torque += cross(-body.r*n, f_f);
		}
	};
	for (int i = 0; i < BN; i++) {
		addCollisionForce(bodies[i], vec3(1, 0, 0), B0);
		addCollisionForce(bodies[i], vec3(0, 1, 0), B0);
		addCollisionForce(bodies[i], vec3(0, 0, 1), B0);
		addCollisionForce(bodies[i], vec3(-1, 0, 0), B1);
		addCollisionForce(bodies[i], vec3(0, -1, 0), B1);
		if (has_top) addCollisionForce(bodies[i], vec3(0, 0, -1), B1);
	}

	// collision between objects
	auto add_force = [&](int i, int j) {
		if (i == j) return;
		vec3 xij = bodies[j].x - bodies[i].x, n = normalize(xij);
		double penetrate = -(length(xij) - 2.0 * RigidBody::r);
		if (penetrate > 0.0) {
			vec3 f_c = -k_c * penetrate * n;
			vec3 f_d = -k_d * dot(bodies[i].v, n) * n;
			vec3 v_f = bodies[i].getAbsoluteVelocity(bodies[i].r*n, true);
			vec3 f_f = -mu_s * (k_c * penetrate) * normalize(v_f - dot(v_f, n) * n);
			if (isnan(f_f.x)) f_f = vec3(0.0);
			//f_f = vec3(0.0);
			bodies[i].force += f_c + f_d + f_f;
			bodies[i].torque += cross(bodies[i].r*n, f_f);
		}
	};
#if 0
	// O(N^2)
	for (int i = 0; i < BN; i++) {
		for (int j = 0; j < BN; j++) {
			add_force(i, j);
		}
	}
#else
	// grid construction
	vec3 P0(INFINITY), P1(-INFINITY), dP(2.0 * RigidBody::r);
	for (int i = 0; i < BN; i++)
		P0 = min(P0, bodies[i].x), P1 = max(P1, bodies[i].x);
	if (1) P0 -= vec3(RigidBody::r), P1 += vec3(RigidBody::r);
	ivec3 GN = ivec3(ceil((P1 - P0) / dP));
	const int MAX_I = 8;
	struct cell { int i[MAX_I] = { -1, -1, -1, -1, -1, -1, -1, -1 }; };
	cell *_G = new cell[GN.x*GN.y*GN.z];
	auto G = [&](int i, int j, int k) -> cell& { return _G[i + GN.x*(j + GN.y*k)]; };
	// add spheres to grid
	for (int i = 0; i < BN; i++) {
		vec3 p = bodies[i].x;
		ivec3 pi = ivec3((p - P0) / dP);
		cell *g = &G(pi.x, pi.y, pi.z);
		for (int u = 0; u < MAX_I; u++) {
			if (g->i[u] == -1) { g->i[u] = i; break; }
		}
	}
	// collision detection
	auto add_cell = [&](int i0, int i, int j, int k) {
		if (i < 0 || i >= GN.x || j < 0 || j >= GN.y || k < 0 || k >= GN.z) return;
		cell *g = &G(i, j, k);
		for (int u = 0; u < MAX_I; u++) {
			if (g->i[u] != -1) add_force(i0, g->i[u]);
			else break;
		}
	};
	for (int k = 0; k < GN.z; k++) for (int j = 0; j < GN.y; j++) for (int i = 0; i < GN.x; i++) {
		cell *g = &G(i, j, k);
		for (int u = 0; u < MAX_I; u++) {
			if (g->i[u] != -1) {
				for (int _i = i - 1; _i <= i + 1; _i++)
					for (int _j = j - 1; _j <= j + 1; _j++)
						for (int _k = k - 1; _k <= k + 1; _k++)
							add_cell(g->i[u], _i, _j, _k);
			}
			else break;
		}
	}
	delete _G;
#endif

}

void update_scene(double dt) {

	const double max_dt = 0.001;
	if (dt > max_dt) {
		int N = (int)ceil(dt / max_dt);
		for (int i = 0; i < N; i++)
			update_scene(dt / N);
		return;
	}

	//Sleep(20);

	double t0 = RigidBody::t;

	int N0 = RigidBody::vectorSize();
	int BN = bodies.size();
	int N = N0 * BN;
	double *x = new double[N];
	for (int i = 0; i < BN; i++) bodies[i].toVector(&x[N0*i]);
	double *temp0 = new double[N], *temp1 = new double[N], *temp2 = new double[N];

	RungeKuttaMethod([&](const double* x, double t, double* dxdt) {
		for (int i = 0; i < BN; i++) {
			bodies[i].fromVector(&x[N0*i]);
			bodies[i].calcDerivedQuantities();
		}
		RigidBody::t = t;
		calc_force_and_torque();
		for (int i = 0; i < BN; i++) {
			bodies[i].toVectorDerivative(&dxdt[N0*i]);
		}
	}, x, N, t0, dt, temp0, temp1, temp2);

	for (int i = 0; i < BN; i++) bodies[i].fromVector(&x[N0*i]);
	RigidBody::t = t0 + dt;

	delete x; delete temp0; delete temp1; delete temp2;
}


#pragma endregion


#include <chrono>

void render() {

	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;
	for (int i = 0; i < _WIN_W; i++) for (int j = 0; j < _WIN_H; j++) _DEPTHBUF[i][j] = INFINITY;
	calcMat();

	// axis and grid
	{
		const int R = 20;
		for (int i = -R; i <= R; i++) {
			drawLine_F(vec3(-R, i, 0), vec3(R, i, 0), 0x404040);
			drawLine_F(vec3(i, -R, 0), vec3(i, R, 0), 0x404040);
		}
		drawLine_F(vec3(0, -R, 0), vec3(0, R, 0), 0x409040);
		drawLine_F(vec3(-R, 0, 0), vec3(R, 0, 0), 0xC04040);
		//drawLine_F(vec3(0, 0, -.6*R), vec3(0, 0, .6*R), 0x4040FF);
	}

	// shape
	{
		const triangle_3d octa[8] = {
			triangle_3d(vec3(0, 0, 1), vec3(1, 0, 0), vec3(0, 1, 0)),
			triangle_3d(vec3(0, 0, 1), vec3(0, 1, 0), vec3(-1, 0, 0)),
			triangle_3d(vec3(0, 0, 1), vec3(-1, 0, 0), vec3(0, -1, 0)),
			triangle_3d(vec3(0, 0, 1), vec3(0, -1, 0), vec3(1, 0, 0)),
			triangle_3d(vec3(0, 0, -1), vec3(0, 1, 0), vec3(1, 0, 0)),
			triangle_3d(vec3(0, 0, -1), vec3(-1, 0, 0), vec3(0, 1, 0)),
			triangle_3d(vec3(0, 0, -1), vec3(0, -1, 0), vec3(-1, 0, 0)),
			triangle_3d(vec3(0, 0, -1), vec3(1, 0, 0), vec3(0, -1, 0))
		};
		auto draw_triangle = [](triangle_3d t, vec3f color) {
			vec3 n = t.unit_normal();
			double c = 0.6 + 0.4*dot(n, normalize(vec3(-0.1, 0.3, 1)));
			drawTriangle_F(t[0], t[1], t[2], toCOLORREF((float)c * color));
		};
		const int SUBDIV = 4;
		for (int i = 0; i < (int)bodies.size(); i++) {
			vec3f color = bodies[i].color;
			const RigidBody b = bodies[i];
			for (int f = 0; f < 8; f++) {
				for (int ui = 0; ui < SUBDIV; ui++) for (int vi = 0; ui + vi < SUBDIV; vi++) {
					double u0 = ui / (double)SUBDIV, u1 = (ui + 1) / (double)SUBDIV;
					double v0 = vi / (double)SUBDIV, v1 = (vi + 1) / (double)SUBDIV;
					draw_triangle(triangle_3d(
						b.getAbsolutePos(b.r * normalize((1 - u0 - v0)*octa[f][0] + u0 * octa[f][1] + v0 * octa[f][2])),
						b.getAbsolutePos(b.r * normalize((1 - u1 - v0)*octa[f][0] + u1 * octa[f][1] + v0 * octa[f][2])),
						b.getAbsolutePos(b.r * normalize((1 - u0 - v1)*octa[f][0] + u0 * octa[f][1] + v1 * octa[f][2]))
					), color);
					if (ui + vi + 2 <= SUBDIV) draw_triangle(triangle_3d(
						b.getAbsolutePos(b.r * normalize((1 - u1 - v1)*octa[f][0] + u1 * octa[f][1] + v1 * octa[f][2])),
						b.getAbsolutePos(b.r * normalize((1 - u0 - v1)*octa[f][0] + u0 * octa[f][1] + v1 * octa[f][2])),
						b.getAbsolutePos(b.r * normalize((1 - u1 - v0)*octa[f][0] + u1 * octa[f][1] + v0 * octa[f][2]))
					), color);
				}
			}
		}
	}

	// simulation box
	{
		const COLORREF col = 0xffffff;
		const vec3 b0 = B0, b1 = has_top ? B1 : vec3(B1.x, B1.y, mix(B0.z, B1.z, 10.0));
		drawLine_F(vec3(b0.x, b0.y, b0.z), vec3(b0.x, b1.y, b0.z), col);
		drawLine_F(vec3(b0.x, b1.y, b0.z), vec3(b1.x, b1.y, b0.z), col);
		drawLine_F(vec3(b1.x, b1.y, b0.z), vec3(b1.x, b0.y, b0.z), col);
		drawLine_F(vec3(b1.x, b0.y, b0.z), vec3(b0.x, b0.y, b0.z), col);
		drawLine_F(vec3(b0.x, b0.y, b0.z), vec3(b0.x, b0.y, b1.z), col);
		drawLine_F(vec3(b0.x, b1.y, b0.z), vec3(b0.x, b1.y, b1.z), col);
		drawLine_F(vec3(b1.x, b0.y, b0.z), vec3(b1.x, b0.y, b1.z), col);
		drawLine_F(vec3(b1.x, b1.y, b0.z), vec3(b1.x, b1.y, b1.z), col);
		if (has_top) {
			drawLine_F(vec3(b0.x, b0.y, b1.z), vec3(b0.x, b1.y, b1.z), col);
			drawLine_F(vec3(b0.x, b1.y, b1.z), vec3(b1.x, b1.y, b1.z), col);
			drawLine_F(vec3(b1.x, b1.y, b1.z), vec3(b1.x, b0.y, b1.z), col);
			drawLine_F(vec3(b1.x, b0.y, b1.z), vec3(b0.x, b0.y, b1.z), col);
		}
	}

	// drawCross3D(Center, 6, 0xFF8000);

	// statistics
	double E = 0.0;
	for (int i = 0; i < (int)bodies.size(); i++) {
		double Eg = bodies[i].m * -dot(bodies[i].x, g);
		double Ev = 0.5 * bodies[i].m * bodies[i].v.sqr();
		double Er = 0.5 * dot(bodies[i].w, bodies[i].I * bodies[i].w);
		E += Eg + Ev + Er;
	}
	//printf("(%lg,%.4lg),", RigidBody::t, E);

	// window title (display time)
	char text[1024];
	sprintf(text, "%d bodies   %.3lfs", (int)bodies.size(), RigidBody::t);
	SetWindowTextA(_HWND, text);
}


// ============================================== User ==============================================


#include <thread>
#include "numerical/random.h"
#include "UI/colors/ColorFunctions.h"

void Init() {
	rz = PI - 1.2, rx = 0.2;

	// initial configuration
	const double sc = 0.2;
	B0 = vec3(-20, -12, 0) * sc, B1 = vec3(20, 12, 18) * sc;
	RigidBody::calcConstants(1.0, 1.0 * sc);

	for (int k = 0; k < 8; k++) {
		for (int j = -5; j <= 5; j++) {
			for (int i = 0; i < 4; i++) {
				bodies.push_back(RigidBody(
					vec3(2 * i, 2 * j, 2 * k + 1)*RigidBody::r,
					quaternion{ 1.0, vec3(0.0) },
					vec3(0, 0, 0),
					vec3(0.0),
					ColorFunctions<vec3f, float>::LightTerrain(k / 8.0)
				));
			}
		}
	}
	bodies.push_back(RigidBody(
		vec3(-3, 0, RigidBody::r),
		quaternion{ 1.0, vec3(0.0) },
		2.0 * vec3(10, 0.1, 5),
		vec3(0.0),
		vec3f(1.0f, 0.5f, 0.0f)
	));

	double zoom_out = 10.0 * sc;
	dist *= zoom_out, Unit /= zoom_out;
	Center = 0.5*(B0 + B1);

	// simulation thread
	new std::thread([]() {
		auto t0 = std::chrono::high_resolution_clock::now();
		const double frame_delay = 0.01;
		for (int i = 0;; i++) {
			double time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
			double time_next = ceil(time / frame_delay)*frame_delay;
			double dt = time_next - RigidBody::t;

			update_scene(dt);
			double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() - time;
			printf("(%lf,%lf),", time, time_elapsed / dt);

			double t_remain = time_next - std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	});

	// rendering thread
	new std::thread([]() {
		for (;;) {
			if (_WINIMG) {
				Render_Needed = true;
				SendMessage(_HWND, WM_NULL, NULL, NULL);
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
		}
	});
}




void WindowResize(int _oldW, int _oldH, int _W, int _H) {
	if (_W*_H == 0 || _oldW * _oldH == 0) return;  // window is minimized
	double pw = _oldW, ph = _oldH, w = _W, h = _H;
	double s = sqrt((w * h) / (pw * ph));
	Unit *= s, dist /= s;
	Render_Needed = true;
}
void WindowClose() {
}

void MouseWheel(int _DELTA) {
	Render_Needed = true;
	if (Ctrl) Center.z += 0.1 * _DELTA / Unit;
	else if (Shift) dist *= exp(-0.001*_DELTA);
	else {
		double s = exp(0.001*_DELTA);
		double D = length(vec2(_WIN_W, _WIN_H)), Max = D, Min = 0.015*D;
		//if (Unit * s > Max) s = Max / Unit; else if (Unit * s < Min) s = Min / Unit;
		Unit *= s, dist /= s;
	}
}
void MouseDownL(int _X, int _Y) {
	clickCursor = Cursor = vec2(_X, _Y);
	mouse_down = true;
	Render_Needed = true;

}
void MouseMove(int _X, int _Y) {
	vec2 P0 = Cursor, P = vec2(_X, _Y), D = P - P0;
	Cursor = P;

	// drag to rotate scene
	if (mouse_down) {
		Render_Needed = true;
		vec2 d = 0.01*D;
		rz -= cos(ry)*d.x + sin(ry)*d.y, rx -= -sin(ry)*d.x + cos(ry)*d.y;  // doesn't work very well
		//rz -= d.x, rx -= d.y;
	}

}
void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	bool moved = (int)length(clickCursor - Cursor) != 0;   // be careful about coincidence
	mouse_down = false;
	Render_Needed = true;
}
void MouseDownR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	Render_Needed = true;
}
void MouseUpR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	bool topmost = GetWindowLong(_HWND, GWL_EXSTYLE) & WS_EX_TOPMOST;
	SetWindowPos(_HWND, topmost ? HWND_NOTOPMOST : HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
	Render_Needed = true;
}
void KeyDown(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Render_Needed = !Ctrl, Ctrl = true;
	else if (_KEY == VK_SHIFT) Render_Needed = !Shift, Shift = true;
	else if (_KEY == VK_MENU) Render_Needed = !Alt, Alt = true;
}
void KeyUp(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = false;
	else if (_KEY == VK_SHIFT) Shift = false;
	else if (_KEY == VK_MENU) Alt = false;
	Render_Needed = true;

	if (_KEY == VK_HOME || (Ctrl && (_KEY == '0' || _KEY == VK_NUMPAD0))) {
		Center = vec3(0.0, 0.0, 0.0);
		rz = 0.2*PI, rx = 0.15*PI, ry = 0.0, dist = 12.0, Unit = 100.0;
	}
}

