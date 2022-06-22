"use strict";


// rendering related
var renderer = {
    canvas: null,
    gl: null,
    width: -1,
    height: -1,
    eps: NaN,
    dt: 0.005,
    iFrame: 0,
    iMouse: [0, 0, -1],
    vsSource: "",
    fsAdvectU: "",
    programAdvectU: null,
    targetVelocity: null,
    fsDivU: "",
    programDivU: null,
    fsPressure: "",
    programPressure: null,
    targetPressure: null,
    fsGradP: null,
    programGradP: null,
    fsAdvectC: null,
    programAdvectC: null,
    targetColor: null,
    fsDisplay: "",
    programDisplay: null,
};

// request shader sources
function loadShaderSource(path) {
    var request = new XMLHttpRequest();
    request.open("GET", path, false);
    request.send(null);
    if (request.status != 200) return "";
    var source = request.responseText;
    return source;
}

// compile shaders and create a shader program
function createShaderProgram(vsSource, fsSource) {
    let gl = renderer.gl;
    function loadShader(gl, type, source) {
        var shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
            throw new Error("Shader compile error: " + gl.getShaderInfoLog(shader));
        return shader;
    }
    var vShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
    var fShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);
    var shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vShader);
    gl.attachShader(shaderProgram, fShader);
    gl.linkProgram(shaderProgram);
    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS))
        throw new Error(gl.getProgramInfoLog(shaderProgram));
    return shaderProgram;
}

// create texture/framebuffer
function createSampleTexture(width, height) {
    let gl = renderer.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const level = 0;
    const internalFormat = gl.RGBA32F;
    const border = 0;
    const format = gl.RGBA;
    const type = gl.FLOAT;
    const data = null;
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
        width, height, border,
        format, type, data);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    // gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    return tex;
}
function createRenderTarget(width, height) {
    let gl = renderer.gl;
    const tex = createSampleTexture(width, height);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const sampler = createSampleTexture(gl, width, height);
    return {
        texture: tex,
        framebuffer: framebuffer,
        sampler: sampler
    };
}
function destroyRenderTarget(target) {
    let gl = renderer.gl;
    gl.deleteTexture(target.texture);
    gl.deleteFramebuffer(target.framebuffer);
}


// call this function to re-render
async function drawScene() {
    let gl = renderer.gl;

    // set position buffer for vertex shader
    function setPositionBuffer(program) {
        var vpLocation = gl.getAttribLocation(program, "vertexPosition");
        const numComponents = 2;
        const type = gl.FLOAT;
        const normalize = false;
        const stride = 0, offset = 0;
        gl.bindBuffer(gl.ARRAY_BUFFER, renderer.positionBuffer);
        gl.vertexAttribPointer(
            vpLocation,
            numComponents, type, normalize, stride, offset);
        gl.enableVertexAttribArray(vpLocation);
    }

    // set uniforms
    function setUniforms(program) {
        gl.uniform2f(gl.getUniformLocation(program, "iResolution"),
            renderer.width, renderer.height);
        gl.uniform1i(gl.getUniformLocation(program, "iFrame"), renderer.iFrame);
        gl.uniform1f(gl.getUniformLocation(program, "eps"), renderer.eps);
        gl.uniform1f(gl.getUniformLocation(program, "dt"), renderer.dt);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, renderer.targetVelocity.sampler);
        gl.uniform1i(gl.getUniformLocation(program, "samplerU"), 0);
    }

    // clear the canvas
    gl.viewport(0, 0, renderer.width, renderer.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // initialize + advect u=u-u*∇u
    gl.useProgram(renderer.programAdvectU);
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.targetVelocity.framebuffer);
    setPositionBuffer(renderer.programAdvectU);
    setUniforms(renderer.programAdvectU);
    gl.uniform3f(gl.getUniformLocation(renderer.programAdvectU, "iMouse"),
        renderer.iMouse[0], renderer.iMouse[1], renderer.iMouse[2]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindTexture(gl.TEXTURE_2D, renderer.targetVelocity.sampler);
    gl.copyTexImage2D(gl.TEXTURE_2D,
        0, gl.RGBA32F, 0, 0, renderer.width, renderer.height, 0);

    // calculate the divergence of advected velocity
    gl.useProgram(renderer.programDivU);
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.targetVelocity.framebuffer);
    setPositionBuffer(renderer.programDivU);
    setUniforms(renderer.programDivU);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindTexture(gl.TEXTURE_2D, renderer.targetVelocity.sampler);
    gl.copyTexImage2D(gl.TEXTURE_2D,
        0, gl.RGBA32F, 0, 0, renderer.width, renderer.height, 0);

    // iteratively solve for pressure
    gl.useProgram(renderer.programPressure);
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.targetPressure.framebuffer);
    for (var iter = 0; iter < 20; iter++) {
        setPositionBuffer(renderer.programAdvectU);
        setUniforms(renderer.programPressure);
        gl.uniform1i(gl.getUniformLocation(renderer.programPressure, "iterIndex"), iter);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, renderer.targetPressure.sampler);
        gl.uniform1i(gl.getUniformLocation(renderer.programPressure, "samplerP"), 1);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.bindTexture(gl.TEXTURE_2D, renderer.targetPressure.sampler);
        gl.copyTexImage2D(gl.TEXTURE_2D,
            0, gl.RGBA32F, 0, 0, renderer.width, renderer.height, 0);
    }

    // apply the pressure update
    gl.useProgram(renderer.programGradP);
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.targetVelocity.framebuffer);
    setPositionBuffer(renderer.programGradP);
    setUniforms(renderer.programGradP);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, renderer.targetPressure.sampler);
    gl.uniform1i(gl.getUniformLocation(renderer.programGradP, "samplerP"), 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindTexture(gl.TEXTURE_2D, renderer.targetVelocity.sampler);
    gl.copyTexImage2D(gl.TEXTURE_2D,
        0, gl.RGBA32F, 0, 0, renderer.width, renderer.height, 0);

    // advect color
    gl.useProgram(renderer.programAdvectC);
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.targetColor.framebuffer);
    setPositionBuffer(renderer.programAdvectC);
    setUniforms(renderer.programAdvectC);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, renderer.targetColor.sampler);
    gl.uniform1i(gl.getUniformLocation(renderer.programAdvectC, "samplerC"), 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindTexture(gl.TEXTURE_2D, renderer.targetColor.sampler);
    gl.copyTexImage2D(gl.TEXTURE_2D,
        0, gl.RGBA32F, 0, 0, renderer.width, renderer.height, 0);

    // put results on canvas
    gl.useProgram(renderer.programDisplay);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    setPositionBuffer(renderer.programDisplay);
    setUniforms(renderer.programDisplay);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, renderer.targetColor.sampler);
    gl.uniform1i(gl.getUniformLocation(renderer.programDisplay, "samplerC"), 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// load renderer/interaction
window.onload = function () {
    // get context
    function onError(error) {
        console.error(error);
        let errorMessageContainer = document.getElementById("error-message");
        errorMessageContainer.style.display = visible;
        errorMessageContainer.innerHTML = error;
    }
    let canvas = document.getElementById("canvas");
    renderer.canvas = canvas;
    renderer.gl = canvas.getContext("webgl2") || canvas.getContext("experimental-webgl2");
    if (renderer.gl == null)
        return onError("Error: Your browser may not support WebGL.");
    if (renderer.gl.getExtension("EXT_color_buffer_float") == null)
        return onError("Error: Your device does not support the `EXT_color_buffer_float` extension.");
    canvas.addEventListener("webglcontextlost", function (event) {
        event.preventDefault();
        onError("Error: WebGL context lost.");
    }, false);
    renderer.width = canvas.width;
    renderer.height = canvas.height;
    // renderer.eps = 32.0 / Math.max(renderer.width, renderer.height);
    renderer.eps = 1.0 / 16.0;

    // load GLSL source
    console.time("load glsl code");
    renderer.vsSource = "#version 300 es\nin vec4 vertexPosition;" +
        "void main(){gl_Position=vertexPosition;}";
    renderer.fsAdvectU = loadShaderSource("fs-advect-u.glsl");
    renderer.fsDivU = loadShaderSource("fs-div-u.glsl");
    renderer.fsPressure = loadShaderSource("fs-pressure.glsl");
    renderer.fsGradP = loadShaderSource("fs-grad-p.glsl");
    renderer.fsAdvectC = loadShaderSource("fs-advect-c.glsl");
    renderer.fsDisplay = loadShaderSource("fs-display.glsl");
    console.timeEnd("load glsl code");

    // compile shaders
    console.time("compile shader");
    try {
        renderer.programAdvectU = createShaderProgram(renderer.vsSource, renderer.fsAdvectU);
        renderer.programDivU = createShaderProgram(renderer.vsSource, renderer.fsDivU);
        renderer.programPressure = createShaderProgram(renderer.vsSource, renderer.fsPressure);
        renderer.programGradP = createShaderProgram(renderer.vsSource, renderer.fsGradP);
        renderer.programAdvectC = createShaderProgram(renderer.vsSource, renderer.fsAdvectC);
        renderer.programDisplay = createShaderProgram(renderer.vsSource, renderer.fsDisplay);
    }
    catch (e) {
        return onError(e);
    }
    console.timeEnd("compile shader");

    // position buffer
    renderer.positionBuffer = renderer.gl.createBuffer();
    renderer.gl.bindBuffer(renderer.gl.ARRAY_BUFFER, renderer.positionBuffer);
    var positions = [-1, 1, 1, 1, -1, -1, 1, -1];
    renderer.gl.bufferData(renderer.gl.ARRAY_BUFFER, new Float32Array(positions), renderer.gl.STATIC_DRAW);

    // framebuffers
    renderer.targetVelocity = createRenderTarget(renderer.width, renderer.height);
    renderer.targetPressure = createRenderTarget(renderer.width, renderer.height);
    renderer.targetColor = createRenderTarget(renderer.width, renderer.height);

    // rendering
    function render() {
        drawScene();
        renderer.iFrame += 1;
        setTimeout(function () { requestAnimationFrame(render); }, 20);
    }
    requestAnimationFrame(render);

    // interactions
    var mouseDown = false;
    function updateMouse(event) {
        if (mouseDown)
            renderer.iMouse = [event.offsetX, renderer.height - 1 - event.offsetY, 1];
        else event[2] = -1;
    }
    window.addEventListener("resize", function (event) {
    });
    canvas.addEventListener("pointerdown", function (event) {
        //event.preventDefault();
        canvas.setPointerCapture(event.pointerId);
        mouseDown = true;
        updateMouse(event);
    });
    canvas.addEventListener("pointerup", function (event) {
        event.preventDefault();
        // render();
        mouseDown = false;
        updateMouse(event);
    });
    canvas.addEventListener("pointermove", function (event) {
        updateMouse(event);
    });

}
