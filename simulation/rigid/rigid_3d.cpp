
#include <cmath>
#include <stdio.h>
#include <algorithm>
#pragma warning(disable: 4244)  // data type conversion warning

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
		int u = sqrt(r*r - i * i) + 0.5;
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
				float z = k * dot(n, vec3(p.x, p.y, 0.) - A);
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

struct RigidBody {
	/* geometry */
	std::vector<vec3> vertices;  // vertices
	std::vector<ivec3> faces;  // triangle faces with ccw normal

	/* constant quantities */
	double m, inv_m;  // mass and its reciprocal
	mat3 I0, inv_I0;  // moment of inertia when right oriented

	/* state variables */
	double t;  // time
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

	vec3 getAbsolutePos(vec3 r) {
		return x + R * r;
	}
	vec3 getAbsoluteVelocity(vec3 r) {
		return v + cross(w, R*r);
	}

	/* constructors */
	RigidBody() {}
	RigidBody(const char* filename, bool isSolid, double normalize_mass = NAN) {
		FILE* fp = fopen(filename, "rb");
		if (!fp) return;
		vec3f *Vs = 0; ply_triangle *Fs = 0;
		int VN, FN;
		COLORREF *v_col = 0, *f_col = 0;
		if (read3DFile(fp, Vs, Fs, VN, FN, v_col, f_col)) {
			for (int i = 0; i < VN; i++)
				vertices.push_back(vec3(Vs[i]));
			for (int i = 0; i < FN; i++)
				faces.push_back(ivec3(Fs[i][0], Fs[i][1], Fs[i][2]));
			calcConstants(1.0, isSolid);
			if (!isnan(normalize_mass)) {
				double s = isSolid ? cbrt(normalize_mass / m) : sqrt(normalize_mass / m);
				for (int i = 0; i < VN; i++) vertices[i] *= s;
				calcConstants(1.0, isSolid);
			}
		}
		fclose(fp);
		if (Vs) delete Vs; if (Fs) delete Fs;
		if (v_col) delete v_col; if (f_col) delete f_col;
		resetVariables();
	}

	// calculate mass and moment of inertia, translate the shape so its center of mass is the origin
	void calcConstants(double density, bool isSolid);

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
	}

};

void RigidBody::calcConstants(double density, bool isSolid) {
	double Volume = 0; vec3 COM = vec3(0.0); mat3 Inertia = mat3(0.0);
	for (int i = 0; i < (int)faces.size(); i++) {
		vec3 a = vertices[faces[i].x], b = vertices[faces[i].y], c = vertices[faces[i].z];
		if (isSolid) {
			double dV = det(a, b, c) / 6.;
			Volume += dV;
			COM += dV * (a + b + c) / 4.;
			Inertia += dV * 0.1*(mat3(dot(a, a) + dot(b, b) + dot(c, c) + dot(a, b) + dot(a, c) + dot(b, c)) -
				(tensor(a, a) + tensor(b, b) + tensor(c, c) + 0.5*(tensor(a, b) + tensor(a, c) + tensor(b, a) + tensor(b, c) + tensor(c, a) + tensor(c, b))));
		}
		else {
			double dV = 0.5*length(cross(b - a, c - a));
			Volume += dV;
			COM += dV * (a + b + c) / 3.;
			Inertia += dV / 6. *(mat3(dot(a, a) + dot(b, b) + dot(c, c) + dot(a, b) + dot(a, c) + dot(b, c)) -
				(tensor(a, a) + tensor(b, b) + tensor(c, c) + 0.5*(tensor(a, b) + tensor(a, c) + tensor(b, a) + tensor(b, c) + tensor(c, a) + tensor(c, b))));
		}
	}
	COM /= Volume;
	Inertia = Inertia - Volume * (mat3(dot(COM, COM)) - tensor(COM, COM));
	this->inv_m = 1.0 / (this->m = density * Volume);
	this->inv_I0 = inverse(this->I0 = density * Inertia);
	for (int i = 0; i < (int)vertices.size(); i++) vertices[i] -= COM;
}


vec3 B0, B1;  // simulation box
const bool has_top = false;

RigidBody body;

const vec3 g = vec3(0, 0, -9.8);


void calc_force_and_torque() {

	// gravity
	body.force = body.m * g;
	body.torque = vec3(0.0);

	// boundary collision
	double sc = body.m / body.vertices.size();
	const double k_c = 10000.0 * sc;  // collision force coefficient
	const double k_d = 100.0 * sc;  // damping coefficient
	const double mu = 0.2;  // friction coefficient
	auto addCollisionForce = [&](vec3 n, vec3 p0, vec3 p, vec3 v) {
		double depth = -dot(n, p - p0);
		if (depth > 0.0) {
			vec3 f_c = k_c * depth * n;
			vec3 f_d = -k_d * dot(v, n) * n;  // not sure if this is physically correct
			vec3 f_f = -mu * (k_c * depth) * normalize(v - dot(v, n) * n + vec3(1e-100));
			vec3 f = f_c + f_d + f_f;
			body.force += f;
			body.torque += cross(p - body.x, f);
		}
	};
	for (int i = 0; i < (int)body.vertices.size(); i++) {
		vec3 p = body.getAbsolutePos(body.vertices[i]);
		vec3 v = body.getAbsoluteVelocity(body.vertices[i]);
		addCollisionForce(vec3(1, 0, 0), B0, p, v);
		addCollisionForce(vec3(0, 1, 0), B0, p, v);
		addCollisionForce(vec3(0, 0, 1), B0, p, v);
		addCollisionForce(vec3(-1, 0, 0), B1, p, v);
		addCollisionForce(vec3(0, -1, 0), B1, p, v);
		if (has_top) addCollisionForce(vec3(0, 0, -1), B1, p, v);
	}
}

void update_scene(double dt) {

	const double max_dt = 0.001;
	if (dt > max_dt) {
		int N = (int)ceil(dt / max_dt);
		for (int i = 0; i < N; i++)
			update_scene(dt / N);
		return;
	}

	double t0 = body.t;

	//const int N = body.vectorSize();
	const int N = 13;
	double x[N]; body.toVector(x);
	double temp0[N], temp1[N], temp2[N];

	RungeKuttaMethod([&](const double* x, double t, double* dxdt) {
		body.fromVector(x);
		body.calcDerivedQuantities();
		body.t = t;
		calc_force_and_torque();
		body.toVectorDerivative(dxdt);
	}, x, N, t0, dt, temp0, temp1, temp2);

	body.fromVector(x);
	body.t = t0 + dt;

	if (1) body.q = body.q * (1.0 / sqrt(body.q.s*body.q.s + body.q.t.sqr()));
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
		const double R = 20.0;
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
		for (ivec3 T : body.faces) {
			triangle_3d t(
				body.getAbsolutePos(body.vertices[T.x]),
				body.getAbsolutePos(body.vertices[T.y]),
				body.getAbsolutePos(body.vertices[T.z])
			);
			vec3 n = t.unit_normal();
			double c = 0.6 + 0.4*dot(n, normalize(vec3(-0.1, 0.3, 1)));
			drawTriangle_F(t[0], t[1], t[2], toCOLORREF(vec3f(c)));
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
	double Eg = body.m * -dot(body.x, g);
	double Ev = 0.5 * body.m * body.v.sqr();
	double Er = 0.5 * dot(body.w, body.I * body.w);
	double E = Eg + Ev + Er;
	printf("(%lg,%.4lg),", body.t, E);
	//printf("(%lg,%lf),", body.t, sqrt(body.q.s*body.q.s + body.q.t.sqr()));

	// window title (display time)
	char text[1024];
	sprintf(text, "%d vertice   %.3lfs", (int)body.vertices.size(), body.t);
	SetWindowTextA(_HWND, text);
}


// ============================================== User ==============================================


#include <thread>

void Init() {

	// initial configuration
	const double sc = 0.2;
	body = RigidBody("D:\\dragon_res4.ply", false, 20.0 * sc*sc);
	//body.x = vec3(0, 0, 2) * sc, body.P = body.m * vec3(5, 5, 8), body.L = body.I0 * vec3(0, 1, 1);
	body.x = vec3(8, -8, 4) * sc, body.P = body.m * vec3(-3, 3, -1), body.L = body.I0 * vec3(1, 1, 1);
	B0 = vec3(-10, -10, 0) * sc, B1 = vec3(10, 10, 10) * sc;

	double zoom_out = 6.0 * sc;
	dist *= zoom_out, Unit /= zoom_out;
	Center = 0.5*(B0 + B1);

	// simulation thread
	new std::thread([]() {
		auto t0 = std::chrono::high_resolution_clock::now();
		const double frame_delay = 0.01;
		for (;;) {
			double time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
			double time_next = ceil(time / frame_delay)*frame_delay;
			double dt = time_next - body.t;

			update_scene(dt);

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

