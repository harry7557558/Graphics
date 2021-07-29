// https://www.shadertoy.com/view/4djSRW
#include "numerical/geometry.h"


// Hash without Sine
// MIT License...
/* Copyright (c)2014 David Hoskins.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

//----------------------------------------------------------------------------------------
//  1 out, 1 in...
float hash11(float p)
{
	p = fract(p * .1031f);
	p *= p + 33.33f;
	p *= p + p;
	return fract(p);
}

//----------------------------------------------------------------------------------------
//  1 out, 2 in...
float hash12(vec2f p)
{
	vec3f p3 = fract(vec3f(p.x, p.y, p.x) * .1031f);
	p3 += dot(p3, p3.yzx() + 33.33f);
	return fract((p3.x + p3.y) * p3.z);
}

//----------------------------------------------------------------------------------------
//  1 out, 3 in...
float hash13(vec3f p3)
{
	p3 = fract(p3 * .1031f);
	p3 += dot(p3, p3.zyx() + 31.32f);
	return fract((p3.x + p3.y) * p3.z);
}

//----------------------------------------------------------------------------------------
//  2 out, 1 in...
vec2f hash21(float p)
{
	vec3f p3 = fract(vec3f(p) * vec3f(.1031f, .1030f, .0973f));
	p3 += dot(p3, p3.yzx() + 33.33f);
	return fract((p3.xx() + p3.yz())*p3.zy());
}

//----------------------------------------------------------------------------------------
///  2 out, 2 in...
vec2f hash22(vec2f p)
{
	vec3f p3 = fract(vec3f(p.x, p.y, p.x) * vec3f(.1031f, .1030f, .0973f));
	p3 += dot(p3, p3.yzx() + 33.33f);
	return fract((p3.xx() + p3.yz())*p3.zy());
}

//----------------------------------------------------------------------------------------
///  2 out, 3 in...
vec2f hash23(vec3f p3)
{
	p3 = fract(p3 * vec3f(.1031f, .1030f, .0973f));
	p3 += dot(p3, p3.yzx() + 33.33f);
	return fract((p3.xx() + p3.yz())*p3.zy());
}

//----------------------------------------------------------------------------------------
//  3 out, 1 in...
vec3f hash31(float p)
{
	vec3f p3 = fract(vec3f(p) * vec3f(.1031f, .1030f, .0973f));
	p3 += dot(p3, p3.yzx() + 33.33f);
	return fract((vec3f(p3.x, p3.x, p3.y) + vec3f(p3.y, p3.z, p3.z))*p3.zyx());
}

//----------------------------------------------------------------------------------------
///  3 out, 2 in...
vec3f hash32(vec2f p)
{
	vec3f p3 = fract(vec3f(p.x, p.y, p.x) * vec3f(.1031f, .1030f, .0973f));
	p3 += dot(p3, p3.yxz() + 33.33f);
	return fract((vec3f(p3.x, p3.x, p3.y) + vec3f(p3.y, p3.z, p3.z))*p3.zyx());
}

//----------------------------------------------------------------------------------------
///  3 out, 3 in...
vec3f hash33(vec3f p3)
{
	p3 = fract(p3 * vec3f(.1031f, .1030f, .0973f));
	p3 += dot(p3, p3.yxz() + 33.33f);
	return fract((vec3f(p3.x, p3.x, p3.y) + vec3f(p3.y, p3.z, p3.z))*p3.zyx());
}

