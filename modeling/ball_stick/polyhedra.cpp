#include "ball_stick.h"

std::vector<ball> Balls;
std::vector<stick> Sticks;

const float ball_radius = 0.18f;
const float stick_radius = 0.09f;


void tetrahedron(vec2 pos) {
	const double rt3_2 = sqrt(3) / 2, rt3_6 = sqrt(3) / 6, rt6_3 = sqrt(6) / 3;
	std::vector<vec3> vertices({
		vec3(0, 0, 0),
		vec3(1, 0, 0),
		vec3(0.5, rt3_2, 0),
		vec3(0.5, rt3_6, rt6_3)
		});
	vec3 d = -vec3(0.5, rt3_6, 0.0) + vec3(pos, ball_radius);
	for (int i = 0; i < 4; i++)
		Balls.push_back(ball{ vec3f(vertices[i] + d) });
}

void cube(vec2 pos) {
	std::vector<vec3> vertices({
		vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1),
		vec3(0, 1, 1), vec3(1, 0, 1), vec3(1, 1, 0), vec3(1, 1, 1)
		});
	vec3 d = -vec3(0.5, 0.5, 0.0) + vec3(pos, ball_radius);
	for (int i = 0; i < 8; i++)
		Balls.push_back(ball{ vec3f(vertices[i] + d) });
}

void octahedron(vec2 pos) {
	const double rt2_2 = sqrt(2) / 2;
	std::vector<vec3> vertices({
		vec3(0, 0, -rt2_2),
		vec3(0.5, 0.5, 0), vec3(-0.5, 0.5, 0), vec3(-0.5, -0.5, 0), vec3(0.5, -0.5, 0),
		vec3(0, 0, rt2_2)
		});
	vec3 d = -vec3(0.0, 0.0, -rt2_2) + vec3(pos, ball_radius);
	for (int i = 0; i < 6; i++)
		Balls.push_back(ball{ vec3f(vertices[i] + d) });
}

void octahedron_balanced(vec2 pos) {
	const double rt3_2 = sqrt(3) / 2, rt3_3 = sqrt(3) / 3, rt3_6 = sqrt(3) / 6, rt6_3 = sqrt(6) / 3;
	std::vector<vec3> vertices({
		vec3(0, 0, 0), vec3(1, 0, 0), vec3(0.5, rt3_2, 0),
		vec3(0, rt3_3, rt6_3), vec3(1, rt3_3, rt6_3), vec3(0.5, -rt3_6, rt6_3)
		});
	vec3 d = -vec3(0.5, rt3_6, 0.0) + vec3(pos, ball_radius);
	for (int i = 0; i < 6; i++)
		Balls.push_back(ball{ vec3f(vertices[i] + d) });
}

void icosahedron(vec2 pos) {
	const double phi = sin(0.3*PI);
	std::vector<vec3> vertices({
		vec3(0, 0.5, phi), vec3(0, 0.5, -phi), vec3(0, -0.5, phi), vec3(0, -0.5, -phi),
		vec3(0.5, phi, 0), vec3(0.5, -phi, 0), vec3(-0.5, phi, 0), vec3(-0.5, -phi, 0),
		vec3(phi, 0, 0.5), vec3(-phi, 0, 0.5), vec3(phi, 0, -0.5), vec3(-phi, 0, -0.5),
		});
	vec3 d = -vec3(0.0, 0.0, -phi) + vec3(pos, ball_radius);
	for (int i = 0; i < 12; i++)
		Balls.push_back(ball{ vec3f(vertices[i] + d) });
}

void dodecahedron(vec2 pos) {
	const double phi = (sqrt(5) + 1) / 2, psi = (sqrt(5) - 1) / 2;
	std::vector<vec3> vertices({
		vec3(1,1,1), vec3(1,1,-1), vec3(1,-1,1), vec3(1,-1,-1), vec3(-1,1,1), vec3(-1,1,-1), vec3(-1,-1,1), vec3(-1,-1,-1),
		vec3(0,phi,psi), vec3(0,phi,-psi), vec3(0,-phi,psi), vec3(0,-phi,-psi),
		vec3(psi,0,phi), vec3(psi,0,-phi), vec3(-psi,0,phi), vec3(-psi,0,-phi),
		vec3(phi,psi,0), vec3(phi,-psi,0), vec3(-phi,psi,0), vec3(-phi,-psi,0),
		});
	double s = sqrt((3 + sqrt(5)) / 8);
	vec3 d = -vec3(0.0, 0.0, -s * phi) + vec3(pos, ball_radius);
	for (vec3 v : vertices)
		Balls.push_back(ball{ vec3f(s * v + d) });
}

void dodecasphere(vec2 pos) {
	const double phi = (sqrt(5) + 1) / 2, psi = (sqrt(5) - 1) / 2;
	std::vector<vec3> vertices({
		vec3(1,1,1), vec3(1,1,-1), vec3(1,-1,1), vec3(1,-1,-1), vec3(-1,1,1), vec3(-1,1,-1), vec3(-1,-1,1), vec3(-1,-1,-1),
		vec3(0,phi,psi), vec3(0,phi,-psi), vec3(0,-phi,psi), vec3(0,-phi,-psi),
		vec3(psi,0,phi), vec3(psi,0,-phi), vec3(-psi,0,phi), vec3(-psi,0,-phi),
		vec3(phi,psi,0), vec3(phi,-psi,0), vec3(-phi,psi,0), vec3(-phi,-psi,0),
		});
	double s = sqrt((3 + sqrt(5)) / 8);
	for (int i = 0; i < 20; i++) vertices[i] *= s;
	std::vector<int> face_list({
		0, 12, 2, 17, 16,
		2, 10, 11, 3, 17,
		2, 12, 14, 6, 10,
		7, 11, 10, 6, 19,
		7, 15, 13, 3, 11,
		13, 1, 16, 17, 3,
		0, 8, 4, 14, 12,
		1, 9, 8, 0, 16,
		5, 9, 1, 13, 15,
		4, 8, 9, 5, 18,
		7, 19, 18, 5, 15,
		14, 4, 18, 6, 19,
		});
	for (int i = 0; i < 60; i += 5) {
		int *f = &face_list[i];
		vec3 p = vec3(0.0);
		for (int u = 0; u < 5; u++) p += vertices[f[u]];
		double a = p.sqr(), b = dot(vertices[f[0]], p), c = vertices[f[0]].sqr() - 1.0;
		double k = (sqrt(b * b - a * c) + b) / a;
		vertices.push_back(k * p);
	}
	vec3 d = -vec3(0.0, 0.0, -s * phi) + vec3(pos, ball_radius);
	for (vec3 v : vertices)
		Balls.push_back(ball{ vec3f(v + d) });
}

void icosasphere(vec2 pos) {
	const double phi = 2.0*sin(0.3*PI);
	std::vector<vec3> vertices({
		vec3(0, 1.0, phi), vec3(0, 1.0, -phi), vec3(0, -1.0, phi), vec3(0, -1.0, -phi),
		vec3(1.0, phi, 0), vec3(1.0, -phi, 0), vec3(-1.0, phi, 0), vec3(-1.0, -phi, 0),
		vec3(phi, 0, 1.0), vec3(-phi, 0, 1.0), vec3(phi, 0, -1.0), vec3(-phi, 0, -1.0),
		});
	for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 12; j++) {
			if (abs(length(vertices[j] - vertices[i]) - 2.0) < 1e-6) {
				vertices.push_back(0.5*(vertices[i] + vertices[j]));
			}
		}
	}
	vec3 d = -vec3(0.0, 0.0, -phi) + vec3(pos, ball_radius);
	for (vec3 p : vertices)
		Balls.push_back(ball{ vec3f(p + d) });
}


int main(int argc, char* argv[]) {

	tetrahedron(vec2(0, 0));

	cube(vec2(2, 0));

	octahedron(vec2(4, 2));
	octahedron_balanced(vec2(4, 0));

	icosahedron(vec2(7, 0));

	dodecahedron(vec2(7, 3));
	dodecasphere(vec2(1, 3));

	icosasphere(vec2(4, 6));

	connect_sticks(Balls, Sticks, 1.0f, 0.001f, 0xff8020);

	printf("%d balls\n", (int)Balls.size());
	printf("%d sticks\n", (int)Sticks.size());
	write_file(argv[1], Balls, Sticks, ball_radius, stick_radius);
	return 0;
}
