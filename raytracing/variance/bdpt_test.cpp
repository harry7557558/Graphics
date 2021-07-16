// path tracing test

#include <cmath>
#include <stdio.h>
#include <algorithm>
#pragma warning(disable: 4244 4305 4996)


#pragma region Windows

#ifndef UNICODE
#define UNICODE
#endif

#include <Windows.h>
#include <windowsx.h>
#include <tchar.h>

#define WIN_NAME "UI"
#define WinW_Padding 100
#define WinH_Padding 100
#define WinW_Default 640
#define WinH_Default 400
#define WinW_Min 300
#define WinH_Min 200
#define WinW_Max 1920
#define WinH_Max 1080

void Init();  // only use this function to initialize variables (or test)
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



// Win32 Entry

bool Render_Needed = true;

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
#define _RDBK { if (!Render_Needed) break; HDC hdc = GetDC(hWnd), HImgMem = CreateCompatibleDC(hdc); HBITMAP hbmOld = (HBITMAP)SelectObject(HImgMem, _HIMG); render(); BitBlt(hdc, 0, 0, _WIN_W, _WIN_H, HImgMem, 0, 0, SRCCOPY); SelectObject(HImgMem, hbmOld), DeleteDC(HImgMem), DeleteDC(hdc); Render_Needed = false; break; }
	switch (message) {
	case WM_CREATE: { if (!_HWND) Init(); break; }
	case WM_CLOSE: { WindowClose(); DestroyWindow(hWnd); return 0; }
	case WM_DESTROY: { PostQuitMessage(0); return 0; }
	case WM_MOVE:; case WM_SIZE: {
		RECT Client; GetClientRect(hWnd, &Client); WindowResize(_WIN_W, _WIN_H, Client.right, Client.bottom); _WIN_W = Client.right, _WIN_H = Client.bottom;
		BITMAPINFO bmi; bmi.bmiHeader.biSize = sizeof(BITMAPINFO), bmi.bmiHeader.biWidth = Client.right, bmi.bmiHeader.biHeight = Client.bottom, bmi.bmiHeader.biPlanes = 1, bmi.bmiHeader.biBitCount = 32; bmi.bmiHeader.biCompression = BI_RGB, bmi.bmiHeader.biSizeImage = 0, bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0, bmi.bmiHeader.biClrUsed = bmi.bmiHeader.biClrImportant = 0; bmi.bmiColors[0].rgbBlue = bmi.bmiColors[0].rgbGreen = bmi.bmiColors[0].rgbRed = bmi.bmiColors[0].rgbReserved = 0;
		if (_HIMG != NULL) DeleteObject(_HIMG); HDC hdc = GetDC(hWnd); _HIMG = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&_WINIMG, NULL, 0); DeleteDC(hdc); _RDBK
	}
	case WM_GETMINMAXINFO: { LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam; lpMMI->ptMinTrackSize.x = WinW_Min, lpMMI->ptMinTrackSize.y = WinH_Min, lpMMI->ptMaxTrackSize.x = WinW_Max, lpMMI->ptMaxTrackSize.y = WinH_Max; break; }
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
	WNDCLASSEX wc; wc.cbSize = sizeof(WNDCLASSEX), wc.style = 0, wc.lpfnWndProc = WndProc, wc.cbClsExtra = wc.cbWndExtra = 0, wc.hInstance = hInstance; wc.hIcon = wc.hIconSm = 0, wc.hCursor = LoadCursor(NULL, IDC_ARROW), wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)), wc.lpszMenuName = NULL, wc.lpszClassName = _T(WIN_NAME);
	if (!RegisterClassEx(&wc)) return -1;
	_HWND = CreateWindow(_T(WIN_NAME), _T(WIN_NAME), WS_OVERLAPPEDWINDOW, WinW_Padding, WinH_Padding, WinW_Default, WinH_Default, NULL, NULL, hInstance, NULL);
	ShowWindow(_HWND, nCmdShow); UpdateWindow(_HWND);
	MSG message; while (GetMessage(&message, 0, 0, 0)) { TranslateMessage(&message); DispatchMessage(&message); } return (int)message.wParam;
}

#pragma endregion







#include "numerical/geometry.h"

const vec3 vec0(0, 0, 0), veci(1, 0, 0), vecj(0, 1, 0), veck(0, 0, 1);
#define SCRCTR vec2(0.5*_WIN_W,0.5*_WIN_H)



#pragma region Global Variables and Functions

// viewport
vec3 Center(0.0, 0.0, 0.0);  // view center in world coordinate
double rz = -0.8, rx = 0.3, ry = 0.0, dist = 8.0, Unit = 60.0;  // yaw, pitch, row, camera distance, scale to screen

// window parameters
char text[64];	// window title
vec3 CamP, ScrO, ScrA, ScrB;  // camera and screen
auto scrPos = [](vec2 pixel) { return ScrO + ((pixel.x + 0.5) / _WIN_W)*ScrA + ((pixel.y + 0.5) / _WIN_H)*ScrB; };
auto scrDir = [](vec2 pixel) { return normalize(scrPos(pixel) - CamP); };

// user parameters
vec2 Cursor = vec2(0, 0), clickCursor;  // current cursor and cursor position when mouse down
bool mouse_down = false;
bool Ctrl = false, Shift = false, Alt = false;  // these variables are shared by both windows


// projection
void getRay(vec2 Cursor, vec3 &p, vec3 &d) {
	p = CamP;
	d = normalize(ScrO + (Cursor.x / _WIN_W)*ScrA + (Cursor.y / _WIN_H)*ScrB - CamP);
}
void getScreen(vec3 &P, vec3 &O, vec3 &A, vec3 &B) {  // O+uA+vB
	double cx = cos(rx), sx = sin(rx), cz = cos(rz), sz = sin(rz);
	vec3 u(-sz, cz, 0), v(-cz * sx, -sz * sx, cx), w(cz * cx, sz * cx, sx);
	mat3 Y = axis_angle(w, -ry); u = Y * u, v = Y * v;
	u *= 0.5*_WIN_W / Unit, v *= 0.5*_WIN_H / Unit, w *= dist;
	P = Center + w;
	O = Center - (u + v), A = u * 2.0, B = v * 2.0;
}


#pragma endregion







// ============================================ Scene ============================================


#include <functional>
#include <vector>
#include "numerical/random.h"
#include "raytracing/brdf.h"



// result of a ray-surface intersection test
struct IntersectionObject {
	double t = INFINITY;  // minimum distance encountered
	vec3 n;  // unit normal
	vec3 color;  // surface color, light, absorption, etc.
};
struct DLS_Object {
	double weight;  // should be cos(φ)/(π*pdf(rd))
	vec3 rd;  // random light direction from ro
};

typedef std::function<bool(vec3 ro, vec3 rd, IntersectionObject&)> Intersector;
typedef std::function<void(vec3 ro, vec3 n, DLS_Object&, uint32_t& seed)> DLS_Sampler;
typedef std::function<void(vec3& ro, vec3& rd, vec3& col, uint32_t& seed)> LightEmitter;


// intersectors

bool intersectSphere(double r, vec3 ro, vec3 rd, IntersectionObject &io) {
	double b = -dot(ro, rd), c = dot(ro, ro) - r * r;
	double delta = b * b - c;
	if (delta < 0.0) return false;
	delta = sqrt(delta);
	double t1 = b - delta, t2 = b + delta;
	if (t1 > t2) std::swap(t1, t2);
	if (t1 > io.t || t2 < 0.) return false;
	io.t = t1 > 0. ? t1 : t2;
	io.n = normalize(ro + rd * io.t);
	return true;
}

bool intersectEllipsoid(vec3 r, vec3 ro, vec3 rd, double &t, vec3 &n) {
	double a = dot(rd / r, rd / r);
	double b = -dot(rd / r, ro / r);
	double c = dot(ro / r, ro / r) - 1.0;
	double delta = b * b - a * c;
	if (delta < 0.0) return false;
	delta = sqrt(delta);
	double t1 = (b - delta) / a, t2 = (b + delta) / a;
	if (t1 > t2) std::swap(t1, t2);
	if (t1 > t || t2 < 0.) return false;
	t = t1 > 0. ? t1 : t2;
	n = normalize((ro + rd * t) / (r*r));
	return true;
}

bool intersectParallelogram(vec3 p, vec3 a, vec3 b, vec3 ro, vec3 rd, IntersectionObject &io) {  // relative with precomputer normal cross(a,b)
	vec3 rp = ro - p;
	vec3 n = cross(a, b);
	double d = 1.0 / dot(rd, n);
	vec3 q = cross(rp, rd);
	double u = -d * dot(q, b); if (u<0. || u>1.) return false;
	double v = d * dot(q, a); if (v<0. || v>1.) return false;
	double t = -d * dot(n, rp);
	if (t > 0.0 && t < io.t) {
		io.t = t, io.n = normalize(n);
		return true;
	}
	return false;
}


// direct light sampling

void DLS_Sphere(double r, vec3 ro, vec3 n, DLS_Object &dls_obj, uint32_t &seed) {
	double a = asin(r / length(ro));
	if (isnan(a)) {  // ray origin is inside the sphere
		dls_obj.weight = 1.0;
		dls_obj.rd = randdir_cosWeighted(n, seed);
		return;
	}

	//dls_obj.rd = normalize(r*randdir_uniform(seed) - ro);  // incorrect
	double z = 1.0 - rand01(seed) * (1.0 - cos(a));
	vec2 xy = sqrt(1.0 - z * z) * cossin(2.0*PI*rand01(seed));
	vec3 w = normalize(-ro), u = normalize(cross(w, vec3(1e-3, 1e-4, 1.0))), v = cross(w, u);
	dls_obj.rd = normalize(z * w + xy.x * u + xy.y * v);

	dls_obj.weight = (2.0*PI * (1.0 - cos(a))) / PI;
	dls_obj.weight *= dot(dls_obj.rd, n);
}

void DLS_Parallelogram(vec3 p, vec3 a, vec3 b, vec3 n, DLS_Object &dls_obj, uint32_t &seed) {
	double u = rand01(seed), v = rand01(seed);
	vec3 q = p + u * a + v * b;
	dls_obj.rd = normalize(q);

	vec3 w = dls_obj.rd;
	vec3 proj_a = a - dot(a, w)*w;
	vec3 proj_b = b - dot(b, w)*w;
	double pdf = q.sqr() / length(cross(proj_a, proj_b));
	dls_obj.weight = dot(dls_obj.rd, n) / (PI*pdf);
}



// an object in the scene
class SceneObject {
public:
	enum SceneObjectType {
		sotNone = -1,
		sotLight,  // light, black body, constant in all directions
		sotDiffuseLambert,  // diffuse surface with constant BRDF
		sotSpecular,  // specular, non-refractive
		sotRefractive,  // refractive
	};
	SceneObjectType type;

	// object
	Intersector intersector;

	// for lights only
	DLS_Sampler dls_sampler;
	LightEmitter light_emitter;

	SceneObject(SceneObjectType type, Intersector intersector,
		DLS_Sampler dls_sampler = nullptr, LightEmitter light_emittor = nullptr) {
		this->type = type;
		this->intersector = intersector;
		if (this->type == sotLight) {
			this->dls_sampler = dls_sampler;
			this->light_emitter = light_emitter;
		}
	}

};


// all objects in the scene to be rendered
std::vector<SceneObject> scene;

void initScene() {

	// lights
	SceneObject ceiling_light(SceneObject::sotLight,
		[](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		double t = -(ro.z - 3.0) / rd.z;
		if (t > 0.0 && t < io.t) {
			vec2 p = ro.xy() + t * rd.xy();
			if (max(abs(p.x) - 2., abs(p.y) - 3.) < 0.0) {
				io.t = t, io.n = vec3(0, 0, 1);
				io.color = vec3(2.0);
				return true;
			}
		}
		return false;
	}, [](vec3 ro, vec3 n, DLS_Object &dlso, uint32_t &seed) {
		DLS_Parallelogram(vec3(-2, -3, 3) - ro, vec3(4, 0, 0), vec3(0, 6, 0),
			n, dlso, seed);
	});
	SceneObject square_light(SceneObject::sotLight,
		[](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		if (!intersectParallelogram(vec3(3, 0, 4), vec3(0, 0.7, 0), vec3(0.5, 0, -0.5), ro, rd, io)) return false;
		io.color = vec3(80.0); return true;
		return false;
	}, [](vec3 ro, vec3 n, DLS_Object &dlso, uint32_t &seed) {
		DLS_Parallelogram(vec3(3, 0, 4) - ro, vec3(0, 0.7, 0), vec3(0.5, 0, -0.5), n, dlso, seed);
	});
	SceneObject bulb_middle(SceneObject::sotLight,
		[](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		if (!intersectSphere(1.0, ro - vec3(0, 0, 2), rd, io)) return false;
		io.color = vec3(1.8); return true;
	}, [](vec3 ro, vec3 n, DLS_Object &dlso, uint32_t &seed) {
		DLS_Sphere(1.0, ro - vec3(0, 0, 2), n, dlso, seed);
	});
	SceneObject bulb_large(SceneObject::sotLight,
		[](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		if (!intersectSphere(5.0, ro - vec3(5, -5, 10), rd, io)) return false;
		io.color = vec3(2.5); return true;
	}, [](vec3 ro, vec3 n, DLS_Object &dlso, uint32_t &seed) {
		DLS_Sphere(5.0, ro - vec3(5, -5, 10), n, dlso, seed);
	});
	SceneObject bulb_far = SceneObject(SceneObject::sotLight,
		[](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		if (!intersectSphere(2.0, ro - vec3(8, 1, 10), rd, io)) return false;
		io.color = vec3(12.0); return true;
	}, [](vec3 ro, vec3 n, DLS_Object &dlso, uint32_t &seed) {
		DLS_Sphere(2.0, ro - vec3(8, 1, 10), n, dlso, seed);
	}, [](vec3 &ro, vec3 &rd, vec3 &col, uint32_t &seed) {
		vec3 n = randdir_uniform(seed);
		ro = vec3(8, 1, 10) + 2.0*n;
		rd = randdir_hemisphere(n, seed);
		col = vec3(12.0);
	});
	SceneObject bulb_dot = SceneObject(SceneObject::sotLight,
		[](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		if (!intersectSphere(0.1, ro - vec3(8, 1, 10), rd, io)) return false;
		io.color = vec3(5000.0); return true;
	}, [](vec3 ro, vec3 n, DLS_Object &dlso, uint32_t &seed) {
		DLS_Sphere(0.1, ro - vec3(8, 1, 10), n, dlso, seed);
	});
	SceneObject bulb_sun = SceneObject(SceneObject::sotLight,
		[](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		if (!intersectSphere(6.957e+8, ro - 152.09e+9*normalize(vec3(8, -1, 10)), rd, io)) return false;
		io.color = vec3(20000.0); return true;
	}, [](vec3 ro, vec3 n, DLS_Object &dlso, uint32_t &seed) {
		DLS_Sphere(6.957e+8, ro - 152.09e+9*normalize(vec3(8, -1, 10)), n, dlso, seed);
	});

	// "uneven" lights
	SceneObject bulb_cosine = SceneObject(SceneObject::sotLight,
		[](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		if (!intersectSphere(2.0, ro - vec3(5, -3, 6), rd, io)) return false;
		io.color = vec3(12.0)*max(-dot(rd, io.n), 0.0); return true;  // cosine-weighted light
	}, [](vec3 ro, vec3 n, DLS_Object &dlso, uint32_t &seed) {
		DLS_Sphere(2.0, ro - vec3(5, -3, 6), n, dlso, seed);
	});
	SceneObject bulb_spotlight = SceneObject(SceneObject::sotLight,
		[](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		const vec3 pos = vec3(10, 0, 2);
		const vec3 dir = normalize(vec3(0, 0, 1) - pos);
		if (!intersectSphere(1.0, ro - pos, rd, io)) return false;
		double s = pow(max(dot(io.n, dir), 0.0), 40.0) * pow(max(-dot(rd, dir), 0.0), 40.0);  // focus in one direction
		io.color = vec3(2000.0)*s; return true;
	}, [](vec3 ro, vec3 n, DLS_Object &dlso, uint32_t &seed) {
		const vec3 pos = vec3(10, 0, 2);
		DLS_Sphere(1.0, ro - pos, n, dlso, seed);
	});

	// diffuse surfaces
	SceneObject plane(SceneObject::sotDiffuseLambert, [](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		double t = -(ro.z) / rd.z;
		if (t > 0.0 && t < io.t) {
			io.t = t, io.n = vec3(0, 0, 1);
			io.color = vec3(1.0);
			return true;
		}
		return false;
	});
	SceneObject wall_x0(SceneObject::sotDiffuseLambert, [](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		double t = -(ro.x + 3.) / rd.x;
		if (!(t > 0.0 && t < io.t)) return false;
		if (ro.z + rd.z*t > 100.0) return false;
		io.t = t, io.n = vec3(1, 0, 0), io.color = vec3(1.0, 0.9, 0.9);
		return true;
	});
	SceneObject wall_y0(SceneObject::sotDiffuseLambert, [](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		double t = -(ro.y - 4.) / rd.y;
		if (!(t > 0.0 && t < io.t)) return false;
		if (ro.z + rd.z*t > 100.0) return false;
		io.t = t, io.n = vec3(0, 1, 0), io.color = vec3(0.9, 1.0, 0.9);
		return true;
	});
	SceneObject wall_x1(SceneObject::sotDiffuseLambert, [](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		double t = -(ro.x - 12.) / rd.x;
		if (!(t > 0.0 && t < io.t)) return false;
		if (ro.z + rd.z*t > 10.0) return false;
		io.t = t, io.n = vec3(-1, 0, 0), io.color = vec3(0.9, 0.9, 1.0);
		return true;
	});
	SceneObject ball(SceneObject::sotDiffuseLambert, [](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		if (intersectSphere(1.0, ro - vec3(0, 0, 1), rd, io)) {
			io.color = sin(dot(io.n, vec3(8, 8, 12))) < 0.0 ? vec3(1.0, 0.5, 0.5) : vec3(0.5, 0.5, 1.0);
			return true;
		}
		return false;
	});
	SceneObject oval(SceneObject::sotDiffuseLambert, [](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		double t = io.t; vec3 n;
		if (intersectEllipsoid(vec3(1.0, 1.618, 1.0), ro - vec3(0, 0, 1), rd, t, n)) {
			io.t = t, io.n = n;
			io.color = sin(dot(n, vec3(8, 8, 12))) < 0.0 ? vec3(1.0, 0.95, 0.2) : vec3(0.4, 0.8, 0.2);
			return true;
		}
		return false;
	});

	// refractive surfaces
	SceneObject ball_glass(SceneObject::sotRefractive, [](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		if (intersectSphere(1.0, ro - vec3(0, 0, 1), rd, io)) {
			io.color = vec3(0.0);  // no absorption
			return true;
		}
		return false;
	});
	SceneObject ball_glass_stained(SceneObject::sotRefractive, [](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		if (intersectSphere(1.0, ro - vec3(0, 0, 1), rd, io)) {
			io.color = vec3(0.0, 0.0, 1.0);  // yellow
			return true;
		}
		return false;
	});
	SceneObject egg_glass(SceneObject::sotRefractive, [](vec3 ro, vec3 rd, IntersectionObject &io)->bool {
		double t = io.t; vec3 n;
		if (intersectEllipsoid(vec3(1.0, 1.0, 1.4), ro - vec3(0, 0, 1.4), rd, t, n)) {
			io.t = t, io.n = n;
			io.color = vec3(0.1, 0.05, 0.0);  // slightly blue
			return true;
		}
		return false;
	});

	//scene.push_back(ceiling_light);
	//scene.push_back(square_light);
	//scene.push_back(bulb_middle);
	//scene.push_back(bulb_large);
	scene.push_back(bulb_far);
	//scene.push_back(bulb_dot);
	//scene.push_back(bulb_sun);
	//scene.push_back(bulb_cosine);
	//scene.push_back(bulb_spotlight);

	scene.push_back(plane);
	//scene.push_back(SceneObject(SceneObject::sotSpecular, plane.intersector));
	scene.push_back(wall_x0);
	scene.push_back(wall_y0);
	//scene.push_back(SceneObject(SceneObject::sotSpecular, wall_x0.intersector));
	//scene.push_back(SceneObject(SceneObject::sotSpecular, wall_y0.intersector));
	scene.push_back(wall_x1);
	//scene.push_back(SceneObject(SceneObject::sotSpecular, wall_x1.intersector));

	scene.push_back(ball);
	//scene.push_back(oval);
	//scene.push_back(SceneObject(SceneObject::sotSpecular, ball.intersector));
	//scene.push_back(ball_glass);
	//scene.push_back(ball_glass_stained);
	//scene.push_back(egg_glass);

}




// ============================================ Rendering ============================================


#define MULTITHREAD 1


#include "variance.h"


#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;

COLORREF toCOLORREF(const vec3 &col) {
	COLORREF C; byte* c = (byte*)&C;
	c[2] = (byte)(255 * clamp(col.x, 0., 1.));
	c[1] = (byte)(255 * clamp(col.y, 0., 1.));
	c[0] = (byte)(255 * clamp(col.z, 0., 1.));
	return C;
}




VarianceObject<vec3> colorBuffer[WinW_Max][WinH_Max];
int colorBuffer_N = 0;


// count what path length is appropriate, prepare for BDPT
// disable multithreading to avoid conflicting
int ITER_COUNT[256];



// intersection test
bool intersectScene(vec3 ro, vec3 rd, int &intersect_id, IntersectionObject &io) {
	io.t = INFINITY;
	intersect_id = -1;
	for (int id = 0; id < (int)scene.size(); id++) {
		SceneObject object = scene[id];
		if (object.intersector(ro, rd, io)) {
			intersect_id = id;
		}
	}
	return intersect_id != -1;
}

// test if the direct path from p to q is occluded
bool isOccluded(vec3 p, vec3 q) {
	vec3 rd = normalize(q - p);
	p += 1e-6*length(p)*rd, q -= 1e-6*length(q)*rd;

	IntersectionObject io;
	io.t = length(q - p);

	for (int id = 0; id < (int)scene.size(); id++) {
		SceneObject object = scene[id];
		if (object.intersector(p, rd, io)) {
			return true;
		}
	}
	return false;
}



// path tracing, with or without direct light sampling
vec3 CalcCol_PT(vec3 ro, vec3 rd, uint32_t &seed, bool dls) {

	vec3 m_col = vec3(1.0), t_col = vec3(0.0);
	bool sample_lightsource = true;  // false if already DLS
	bool is_inside = false;  // whether it is inside an object or not, assume false initially

	for (int iter = 0; iter < 64; iter++) {
		ITER_COUNT[iter]++;
		ro += 1e-6*rd;

		IntersectionObject io;
		int intersect_id;
		if (!intersectScene(ro, rd, intersect_id, io)) {
			return t_col;
		}
		SceneObject::SceneObjectType intersect_type = scene[intersect_id].type;

		double t = io.t;
		vec3 n = io.n;
		vec3 col = io.color;

		// update ray
		n = dot(rd, n) < 0. ? n : -n;  // ray hits into the surface
		ro = ro + rd * t;
		if (intersect_type == SceneObject::sotLight) {
			if (sample_lightsource) return t_col + m_col * col;
			return t_col;
		}
		if (intersect_type == SceneObject::sotDiffuseLambert) {
			m_col *= col;
			// direct light sample
			if (dls) {
				for (SceneObject light : scene) {
					if (light.type == SceneObject::sotLight) {
						DLS_Object dlso;
						light.dls_sampler(ro, n, dlso, seed);
						if (dot(dlso.rd, n) > 0.0) {
							IntersectionObject io;
							light.intersector(ro, dlso.rd, io);
							if (!isOccluded(ro, ro + dlso.rd*io.t)) {
								vec3 light_col = io.color*dlso.weight;
								t_col += m_col * light_col;
							}
						}
					}
				}
				sample_lightsource = false;
			}
			else sample_lightsource = true;
			// update ray
			rd = randdir_cosWeighted(n, seed);
		}
		if (intersect_type == SceneObject::sotSpecular) {
			m_col *= col;
			rd = rd - 2.0*dot(rd, n)*n;
			sample_lightsource = true;
		}
		if (intersect_type == SceneObject::sotRefractive) {
			if (is_inside) m_col *= exp(-col * t);
			double n0 = 1.0, n1 = 1.0;
			if (is_inside) n0 = 1.5, n1 = 1.0;
			else n0 = 1.0, n1 = 1.5;
			rd = randdir_Fresnel<vec3>(rd, n, n0, n1, seed);
			sample_lightsource = true;
		}
		if (dot(rd, n) < 0.0) is_inside ^= true;

		ITER_COUNT[iter]--;
	}
	return t_col;
}



// bidirectional path tracing
vec3 CalcCol_BDPT(vec3 ro, vec3 rd, uint32_t &seed) {

	struct Vertex {
		int id;  // object id
		vec3 p;  // position
		vec3 n;  // normal
		vec3 rd;  // incoming eye/light path, goes "into" the surface
		vec3 w;  // product of weights
	};

	Vertex eyepath[32], lightpath[32];
	int eyepath_length = 0, lightpath_length = 0;

	// pass 1 - eyepath

	vec3 col_weight = vec3(1.0);
	for (; eyepath_length < 32; eyepath_length++) {
		ro += 1e-6*rd;

		IntersectionObject io; int intersect_id;
		if (!intersectScene(ro, rd, intersect_id, io)) {
			break;
		}
		SceneObject::SceneObjectType intersect_type = scene[intersect_id].type;

		if (intersect_type == SceneObject::sotLight) {
			// not implemented
			break;
		}
		else if (intersect_type == SceneObject::sotDiffuseLambert) {
			ro += rd * io.t;
			col_weight *= io.color;
			eyepath[eyepath_length].id = intersect_id;
			eyepath[eyepath_length].p = ro;
			eyepath[eyepath_length].n = io.n;
			eyepath[eyepath_length].rd = rd;
			eyepath[eyepath_length].w = col_weight;
			rd = randdir_cosWeighted(io.n, seed);
		}
		else break;  // not implemented
	}

	// pass 2 - light path

	int light_id = -1;
	for (int id = 0; id < (int)scene.size(); id++) {
		if (scene[id].type == SceneObject::sotLight) {
			light_id = id;
			break;
		}
	}
	scene[light_id].light_emitter(ro, rd, col_weight, seed);
	lightpath[0].id = light_id;
	lightpath[0].p = ro;
	lightpath[0].rd = rd;
	lightpath[0].w = col_weight;
	for (lightpath_length = 1; lightpath_length < 32; lightpath_length++) {
		ro += 1e-6 * rd;

		IntersectionObject io; int intersect_id;
		if (!intersectScene(ro, rd, intersect_id, io)) {
			break;
		}
		SceneObject::SceneObjectType intersect_type = scene[intersect_id].type;

		if (intersect_type == SceneObject::sotLight) {
			break;
		}
		else if (intersect_type == SceneObject::sotDiffuseLambert) {
			ro += rd * io.t;
			col_weight *= io.color;
			lightpath[lightpath_length].id = intersect_id;
			lightpath[lightpath_length].p = ro;
			lightpath[lightpath_length].n = io.n;
			lightpath[lightpath_length].rd = rd;
			lightpath[lightpath_length].w = col_weight;
			rd = randdir_cosWeighted(io.n, seed);
		}
		else break;  // not implemented
	}

	// connect paths

	double cols[32][32];  // eyepath_i, lightpath_j
	for (int i = 0; i < eyepath_length; i++) {
		for (int j = 0; j < lightpath_length; j++) {
			// incomplete
		}
	}

}





void Render_Exec(void(*task)(int, int, int, bool*), int Max);

void render_RT() {
	static uint32_t call_time = 0;
	call_time = lcg_next(call_time);

	colorBuffer_N++;

	Render_Exec([](int beg, int end, int step, bool* sig) {
		const int WIN_SIZE = _WIN_W * _WIN_H;
		for (int k = beg; k < end; k += step) {
			int i = k % _WIN_W, j = k / _WIN_W;
			const int N = 1;
			vec3 col(0.);
			for (int u = 0; u < N; u++) {
				uint32_t seed = hashu(u*WIN_SIZE + k + call_time);
				vec3 rd = scrDir(vec2(i + rand01(seed), j + rand01(seed)));
				vec3 ro = CamP + 1.0*rd;
				//col += CalcCol_PT(ro, rd, seed, false);
				col += CalcCol_PT(ro, rd, seed, true);
			}
			if (colorBuffer_N == 1) colorBuffer[i][j] = VarianceObject<vec3>();
			colorBuffer[i][j].addElement(col / N);

			col = abs(colorBuffer[i][j].getMean());
			//col = 0.5 * pow(col, 1.0 / 2.2);
			Canvas(i, j) = toCOLORREF(col);

			//if (colorBuffer_N > 1) Canvas(i, j) = toCOLORREF(colorBuffer[i][j].getVariance());
		}
		if (sig) *sig = true;
	}, _WIN_W*_WIN_H);
}



#include <thread>

#if MULTITHREAD
void Render_Exec(void(*task)(int, int, int, bool*), int Max) {
	const int MAX_THREADS = std::thread::hardware_concurrency();
	bool* fn = new bool[MAX_THREADS];
	std::thread** T = new std::thread*[MAX_THREADS];
	for (int i = 0; i < MAX_THREADS; i++) {
		fn[i] = false;
		T[i] = new std::thread(task, i, Max, MAX_THREADS, &fn[i]);
	}
	int count; do {
		count = 0;
		for (int i = 0; i < MAX_THREADS; i++) count += fn[i];
	} while (count < MAX_THREADS);
	//for (int i = 0; i < MAX_THREADS; i++) delete T[i];
	delete fn; delete T;
}
#else
void Render_Exec(void(*task)(int, int, int, bool*), int Max) {
	task(0, Max, 1, NULL);
}
#endif



void render() {
	auto t0 = NTime::now();
	// initialize window
	for (int i = 0, l = _WIN_W * _WIN_H; i < l; i++) _WINIMG[i] = 0;
	for (int i = 0; i < _WIN_W; i++) for (int j = 0; j < _WIN_H; j++) _DEPTHBUF[i][j] = INFINITY;
	getScreen(CamP, ScrO, ScrA, ScrB);
	//printf("W=%d,H=%d; CamP=vec3(%lf,%lf,%lf),ScrO=vec3(%lf,%lf,%lf),ScrA=vec3(%lf,%lf,%lf),ScrB=vec3(%lf,%lf,%lf);\n", \
		_WIN_W, _WIN_H, CamP.x, CamP.y, CamP.z, ScrO.x, ScrO.y, ScrO.z, ScrA.x, ScrA.y, ScrA.z, ScrB.x, ScrB.y, ScrB.z);


	render_RT();


	for (int i = 0; i < 64; i++) printf("(%d,%d),", i, ITER_COUNT[i]), ITER_COUNT[i] = 0; printf("\n");


	double t = fsec(NTime::now() - t0).count();
	sprintf(text, "[%d×%d, %d]  %.1fms (%.1ffps)\n", _WIN_W, _WIN_H, colorBuffer_N, 1000.0*t, 1. / t);
	SetWindowTextA(_HWND, text);
}





// ============================================== Testing ==============================================

// test if the rendering functions with and without DLS are consistent


std::vector<vec3> testConverge(std::function<vec3(uint32_t)> sample) {
	const int N = 1 << 20;
	VarianceObject<vec3> vl;

	std::vector<vec3> avrs;

	printf("[");
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 1; i <= N; i++) {
		vec3 col = sample(uint32_t(i));
		vl.addElement(col);
		if ((i > (2 << 7)) && ((i & (i - 1)) == 0)) {  // integer power of 2
			int power = (int)(log2(i) + 1e-6);
			vec3 avr = vl.getMean();
			printf("(%lf,%lf,%lf),", avr.x, avr.y, avr.z);
			avrs.push_back(avr);
		}
	}
	double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
	printf("\b]\n");
	printf("%.2lfsecs\n", time_elapsed);

	//vec3 sigma = vl.getStandardDeviation();
	//printf("sigma = (%lf,%lf,%lf)\n", sigma.x, sigma.y, sigma.z);
	vec3 var = vl.getVariance();
	printf("var = (%lf,%lf,%lf)\n", var.x, var.y, var.z);

	// time needed for <0.001 variance
	double variance = dot(var, vec3(0.3, 0.59, 0.11));
	double time_per_sample = time_elapsed / N;
	double time_needed = (time_per_sample * _WIN_W * _WIN_H) * (variance / 0.001);
	printf("time needed for <0.001 variance: %.1lfmin\n", time_needed / 60);

	return avrs;
}


void testConsistency(vec3 ro, vec3 rd) {

	std::function<vec3(uint32_t)> without_dls = [&](uint32_t seed) {
		return CalcCol_PT(ro, rd, seed, false);
	};
	std::function<vec3(uint32_t)> with_dls = [&](uint32_t seed) {
		return CalcCol_PT(ro, rd, seed, true);
	};

	printf("Without DLS\n");
	std::vector<vec3> avr_without = testConverge(without_dls);
	printf("\nWith DLS\n");
	std::vector<vec3> avr_with = testConverge(with_dls);

	printf("\n[");
	for (int i = 0; i < (int)avr_without.size(); i++) {
		printf("%lf,", length(avr_with[i] - avr_without[i]));
	}
	printf("\b]\n");

	printf("\n\n");
}



// ============================================== User ==============================================


bool inited = false;
void Init() {
	if (inited) return; inited = true;

	initScene();
	Center = vec3(0, 0, 1);

	//testConsistency();

}


void keyDownShared(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = true;
	else if (_KEY == VK_SHIFT) Shift = true;
	else if (_KEY == VK_MENU) Alt = true;
}
void keyUpShared(WPARAM _KEY) {
	if (_KEY == VK_CONTROL) Ctrl = false;
	else if (_KEY == VK_SHIFT) Shift = false;
	else if (_KEY == VK_MENU) Alt = false;
}

void WindowResize(int _oldW, int _oldH, int _W, int _H) {
	if (_W*_H == 0 || _oldW * _oldH == 0) return;  // window is minimized
	double pw = _oldW, ph = _oldH, w = _W, h = _H;
	double s = sqrt((w * h) / (pw * ph));
	Unit *= s;
	Render_Needed = true;
	colorBuffer_N = 0;
}
void WindowClose() {}

void MouseWheel(int _DELTA) {
	Render_Needed = true;
	if (_DELTA) colorBuffer_N = 0;
	if (Ctrl) Center.z += 0.1 * _DELTA / Unit;
	else if (Shift) {
		double sc = exp(0.001*_DELTA);
		dist *= sc, Unit *= sc;
	}
	else {
		double s = exp(0.001*_DELTA);
		double D = length(vec2(_WIN_W, _WIN_H)), Max = 1000.0*D, Min = 0.001*D;
		if (Unit * s > Max) s = Max / Unit; else if (Unit * s < Min) s = Min / Unit;
		Unit *= s;
		dist /= s;
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

	Render_Needed = true;

	if (mouse_down) {
		colorBuffer_N = 0;
		if (Ctrl) {
			vec3 d = scrDir(P0);
			vec3 p = CamP.z / d.z * d;
			d = scrDir(P);
			vec3 q = CamP.z / d.z * d;
			Center += q - p;
		}
		else if (Shift) {
			ry += 0.005*D.y;
		}
		else {
			vec2 d = 0.01*D;
			rz -= cos(ry)*d.x + sin(ry)*d.y, rx -= -sin(ry)*d.x + cos(ry)*d.y;  // doesn't work very well
			//rz -= d.x, rx -= d.y;
		}
	}

}
void MouseUpL(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	bool moved = (int)length(clickCursor - Cursor) != 0;
	mouse_down = false;
	Render_Needed = true;
}
void MouseDownR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);
	Render_Needed = true;
}
void MouseUpR(int _X, int _Y) {
	Cursor = vec2(_X, _Y);

	// test consistency
	vec3 rd = normalize(scrDir(Cursor));
	vec3 ro = CamP + 1.0 * rd;
	testConsistency(ro, rd);
}
void KeyDown(WPARAM _KEY) {
	keyDownShared(_KEY);
}
void KeyUp(WPARAM _KEY) {
	keyUpShared(_KEY);
}

