// parse SVG path data

// bool svg_path_read::parse_path(std::string S, std::vector<svg_path_read::bezier3> &V)

// return true: succeeded
// return false: fail, interpreted data before the first error may be added to V

// set `svg_path_read::fit_ellipse` to false to return false when encounter `a` command
// otherwise, curve fitting will be applied



#include <string>
#include <vector>

#include "numerical/geometry.h"

namespace svg_path_read {
	bool fit_ellipse = true;

	// 2x3 matrix, linear transform
	class mat2x3 {
	public:
		double a, b, c, d, x, y;
		mat2x3() :a(1), b(0), c(0), d(1), x(0), y(0) {}
		mat2x3(double a, double b, double c, double d) :a(a), b(b), c(c), d(d), x(0), y(0) {}
		mat2x3(double x, double y) :a(1), b(0), c(0), d(1), x(x), y(y) {}
		mat2x3(double a, double b, double c, double d, double x, double y) :a(a), b(b), c(c), d(d), x(x), y(y) {}
		mat2x3 operator * (const mat2x3 &m) const {
			return mat2x3(a*m.a + b * m.c, a*m.b + b * m.d, c*m.a + d * m.c, c*m.b + d * m.d,
				a*m.x + b * m.y + x, c*m.x + d * m.y + y);
		}
		vec2 applyTo(const vec2 &v) const {
			return vec2(a*v.x + b * v.y + x, c*v.x + d * v.y + y);
		}
		vec2 operator * (const vec2 &v) const {
			return vec2(a*v.x + b * v.y, c*v.x + d * v.y);
		}
	};

	// cubic spline class
	class bezier3 {
	public:
		vec2 A, B, C, D;  // 4 control points in order

		bezier3() {}
		bezier3(vec2 A, vec2 B, vec2 C, vec2 D) :A(A), B(B), C(C), D(D) {}
		void translate(vec2 p) {
			A += p, B += p, C += p, D += p;
		}
		void scale(vec2 s) {
			A *= s, B *= s, C *= s, D *= s;
		}
		void applyMatrix(const mat2x3 &M) {
			A = M * A, B = M * B, C = M * C, D = M * D;
		}
	};
	bezier3 fromSegment(vec2 A, vec2 B) {
		return bezier3(A, .618*A + .382*B, .382*A + .618*B, B);
	}
	bezier3 fromBezier2(vec2 A, vec2 B, vec2 C) {
		return bezier3(A, A + (2. / 3.)*(B - A), C + (2. / 3.)*(B - C), C);
	}
	bezier3 fromBezier3(vec2 A, vec2 B, vec2 C, vec2 D) {
		return bezier3(A, B, C, D);
	}

	bezier3 fromArc(double theta, mat2x3 T) {
		// unit arc from angle 0 to theta, 0 < theta < pi, least square fitting
		// explanation: https://www.shadertoy.com/view/wly3WG  (not a perfect loss function)
		// consider replacing the following by numerical optimization + polynomial fitting
		double s1 = sin(theta), s2 = s1 * s1, s3 = s2 * s1, s4 = s3 * s1,
			c1 = cos(theta), c2 = c1 * c1, c3 = c2 * c1, c4 = c3 * c1;
		double a = 756 * s4 + (1512 * c2 - 1890 * c1 + 810)*s2 + 756 * c4 - 1890 * c3 + 2430 * c2 - 1890 * c1 + 756,
			b = (3996 * s3 + (3996 * c2 - 6750 * c1 + 3996)*s1) / a,
			c = (2520 * s4 + (5040 * c2 - 507 * c1 + 2736)*s2 + 2520 * c4 - 507 * c3 - 6600 * c2 + 7215 * c1 - 2628) / a,
			d = (3439 * s3 + (3439 * c2 + 4276 * c1 - 7715)*s1) / a;
		double p = (c - b * b / 3) / 3, q = -0.5 * ((b*b / 13.5 - c / 3) * b + d);
		a = q * q + p * p * p;
		double r = a > 0 ? cbrt(q + sqrt(a)) + cbrt(q - sqrt(a)) - b / 3
			: 2 * pow(q*q - a, 1.0 / 6) * cos(atan2(sqrt(-a), q) / 3) - b / 3;
		bezier3 R = fromBezier3(vec2(1, 0), vec2(1, r), vec2(c1 + r * s1, s1 - r * c1), vec2(c1, s1));
		R.applyMatrix(T);
		return R;
	}
	void fromArc(vec2 c, vec2 r, double t0, double t1, double rot, std::vector<bezier3> &v) {
		double dt = t1 - t0;
		int n = abs(dt) < 0.5 ? 1 : int((abs(dt) - 0.5) / (PI / 2)) + 1;
		dt /= n;
		mat2x3 B(cos(t0), -sin(t0), sin(t0), cos(t0));
		if (dt < 0) B = B * mat2x3(1, 0, 0, -1);
		bezier3 s = fromArc(abs(dt), B), d;
		mat2x3 R = mat2x3(cos(dt), -sin(dt), sin(dt), cos(dt));
		mat2x3 T = mat2x3(c.x, c.y) * mat2x3(cos(rot), -sin(rot), sin(rot), cos(rot)) * mat2x3(r.x, 0, 0, r.y);
		for (int i = 0; i < n; i++) {
			d = s; d.applyMatrix(T);
			v.push_back(d);
			s.applyMatrix(R);
		}
	}

	bool parse_path(const std::string S, std::vector<bezier3> &V) {

		// macros
#define isFloat(c) ((c >= '0' && c <= '9') || c == '-' || c == '.')
#define readFloat(r) do { \
			while (d < S.size() && (S[d] == ' ' || S[d] == ',')) d++; \
			if (d >= S.size() || !isFloat(S[d])) return false; \
			unsigned sz; \
			(r) = std::stod(&S[d], &sz); \
			d += sz; \
		} while (0)
#define readPoint(v) \
		do { readFloat((v).x); readFloat((v).y); } while(0)

		char cmd = '\0';
		vec2 P(0, 0), P0(0, 0), P1(NAN);
		for (unsigned d = 0; d < S.size();) {
			while (d < S.size() && (S[d] == ' ' || S[d] == ',')) d++;

			if (std::string(fit_ellipse ? "MZLHVCSQT" : "MZLHVCSQTA").find(S[d] >= 'a' ? S[d] - 32 : S[d]) != -1) cmd = S[d], d++;
			else if (!isFloat(S[d])) return false;

			switch (cmd) {
			case 'M':;
			case 'm': {
				vec2 tmp; readPoint(tmp);
				if (cmd == 'm') P = P0 + tmp;
				else P = tmp;
				P0 = P, P1 = vec2(NAN);
				break;
			}
			case 'Z':;
			case 'z': {
				if (P.x != P0.x || P.y != P0.y) V.push_back(fromSegment(P, P0));
				P1 = vec2(NAN);
				break;
			}
			case 'L':;
			case 'l': {
				vec2 Q; readPoint(Q);
				if (cmd == 'l') Q = P + Q;
				if (P.x != Q.x || P.y != Q.y) V.push_back(fromSegment(P, Q));
				P1 = P, P = Q;
				break;
			}
			case 'H':;
			case 'h': {
				double c; readFloat(c);
				vec2 tmp = P;
				if (cmd == 'H') tmp.x = c;
				else tmp.x += c;
				V.push_back(fromSegment(P, tmp));
				P1 = P, P = tmp;
				break;
			}
			case 'V':;
			case 'v': {
				double c; readFloat(c);
				vec2 tmp = P;
				if (cmd == 'V') tmp.y = c;
				else tmp.y += c;
				V.push_back(fromSegment(P, tmp));
				P1 = P, P = tmp;
				break;
			}
			case 'C':;
			case 'c': {
				vec2 B, C, D;
				readPoint(B); readPoint(C); readPoint(D);
				if (cmd == 'c') B = B + P, C = C + P, D = D + P;
				V.push_back(fromBezier3(P, B, C, D));
				P1 = C, P = D;
				break;
			}
			case 'S':;
			case 's': {
				if (isnan(P1.x)) return 0;
				vec2 B = P * 2.0 - P1;
				vec2 C, D;
				readPoint(C); readPoint(D);
				if (cmd == 's') C = P + C, D = P + D;
				V.push_back(fromBezier3(P, B, C, D));
				P1 = C, P = D;
				break;
			}
			case 'Q':;
			case 'q': {
				vec2 B, C;
				readPoint(B); readPoint(C);
				if (cmd == 'q') B = B + P, C = C + P;
				V.push_back(fromBezier2(P, B, C));
				P1 = B, P = C;
				break;
			}
			case 'T':;
			case 't': {
				if (isnan(P1.x)) return 0;
				vec2 B = P * 2.0 - P1;
				vec2 C; readPoint(C);
				if (cmd == 't') C = P + C;
				V.push_back(fromBezier2(P, B, C));
				P1 = B, P = C;
				break;
			}
			case 'A':;
			case 'a': {		// possibly have bug
				if (!fit_ellipse) return false;

				vec2 r; readPoint(r);
				double theta; readFloat(theta); theta *= PI / 180;
				bool laf, sf; readFloat(laf); readFloat(sf);
				vec2 Q; readPoint(Q);
				if (cmd == 'a') Q = P + Q;

				mat2x3 T = mat2x3(1. / r.x, 0, 0, 1. / r.y) * mat2x3(cos(theta), sin(theta), -sin(theta), cos(theta));
				vec2 p = T * P, q = T * Q, d = q - p;
				if (length(d) >= 2.0) {
					double s = (2.0 - 1e-12) / length(d);
					r = r * (1. / s), p = p * s, q = q * s, d = d * s;
				}
				double a = acos(0.5*length(d)), b;
				if (isnan(a)) {
					P = Q, P1 = P;
					break;
				}
				vec2 C = p + mat2x3(cos(a), -sin(a), sin(a), cos(a)) * normalize(d);
				T = mat2x3(r.x, 0, 0, r.y);
				C = T * C, p = T * p, q = T * q;
				if (!sf ^ laf) C = p + q - C;

				T = mat2x3(cos(theta), -sin(theta), sin(theta), cos(theta));
				a = atan2((p.y - C.y) / r.y, (p.x - C.x) / r.x), b = atan2((q.y - C.y) / r.y, (q.x - C.x) / r.x);
				if (sf && b < a) b += 2 * PI;
				if (!sf && a < b) a += 2 * PI;
				fromArc(T*C, r, a, b, theta, V);

				P = Q;
				break;
			}
			default: {
				return false;
			}
			}
		}
		return true;
#undef isFloat
#undef readFloat
#undef readPoint
	}


}  // namespace svg_path_read



