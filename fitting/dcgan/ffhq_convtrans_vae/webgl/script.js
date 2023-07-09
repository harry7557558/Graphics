"use strict";


// rendering related
var renderer = {
    canvas: null,
    gl: null,
    width: -1,
    height: -1,
    iFrame: 0,
    iBrightness: 1.0,
    iGamma: 1.0,
    vsSource: "",
    w01: null,
    fs4x4: "",
    program4x4: null,
    target4x4: null,
    w02: null,
    fs8x8: "",
    program8x8: null,
    target8x8: "",
    w03: null,
    fs16x16: null,
    program16x16: null,
    target16x16: null,
    w04: null,
    fs32x32: null,
    program32x32: null,
    target32x32: null,
    w05: null,
    fs64x64: null,
    program64x64: null,
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
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.MIRRORED_REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.MIRRORED_REPEAT);
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

function loadWeightTexture(url, texture_name) {
    const gl = renderer.gl;

    function onload(weights) {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);

        var l = weights.length / 4, rl = Math.sqrt(l);
        var w = 1, h = l;
        for (var i = 1; i <= rl; i++) {
            var j = Math.round(l / i);
            if (i * j == l) w = i, h = j;
        }
        console.log(texture_name, w, h);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F,
            w, h, 0,
            gl.RGBA, gl.FLOAT,
            weights);

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, 0);

        renderer[texture_name] = tex;
    }

    var req = new XMLHttpRequest();
    req.open("GET", url, true);
    req.responseType = "arraybuffer";
    req.onerror = function (e) {
        alert("Failed to load texture " + texture_name);
    };
    req.onload = function (e) {
        if (req.status == 200) {
            var weights = new Float32Array(req.response);
            onload(weights);
        }
        else {
            req.onerror();
        }
    };
    req.send();
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

    gl.viewport(0, 0, renderer.width, renderer.height);

    // first layer - (32) => (512) => (32, 4, 4)
    gl.useProgram(renderer.program4x4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.target4x4.framebuffer);
    setPositionBuffer(renderer.program4x4);
    gl.uniform1i(gl.getUniformLocation(renderer.program4x4, "iFrame"), renderer.iFrame);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, renderer.w01);
    gl.uniform1i(gl.getUniformLocation(renderer.program4x4, "uWeights"), 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // second layer - (32, 4, 4) => (64, 8, 8)
    gl.useProgram(renderer.program8x8);
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.target8x8.framebuffer);
    setPositionBuffer(renderer.program8x8);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, renderer.target4x4.texture);
    gl.uniform1i(gl.getUniformLocation(renderer.program8x8, "uSrc"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, renderer.w02);
    gl.uniform1i(gl.getUniformLocation(renderer.program8x8, "uWeights"), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, renderer.b02);
    gl.uniform1i(gl.getUniformLocation(renderer.program8x8, "uBiases"), 2);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // third layer - (64, 8, 8) -> (32, 16, 16)
    gl.useProgram(renderer.program16x16);
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.target16x16.framebuffer);
    setPositionBuffer(renderer.program16x16);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, renderer.target8x8.texture);
    gl.uniform1i(gl.getUniformLocation(renderer.program16x16, "uSrc"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, renderer.w03);
    gl.uniform1i(gl.getUniformLocation(renderer.program16x16, "uWeights"), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, renderer.b03);
    gl.uniform1i(gl.getUniformLocation(renderer.program16x16, "uBiases"), 2);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // fourth layer - (32, 16, 16) -> (16, 32, 32)
    gl.useProgram(renderer.program32x32);
    gl.bindFramebuffer(gl.FRAMEBUFFER, renderer.target32x32.framebuffer);
    setPositionBuffer(renderer.program32x32);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, renderer.target16x16.texture);
    gl.uniform1i(gl.getUniformLocation(renderer.program32x32, "uSrc"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, renderer.w04);
    gl.uniform1i(gl.getUniformLocation(renderer.program32x32, "uWeights"), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, renderer.b04);
    gl.uniform1i(gl.getUniformLocation(renderer.program32x32, "uBiases"), 2);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // fifth layer - (16, 32, 32) -> (3, 64, 64)
    gl.useProgram(renderer.program64x64);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    setPositionBuffer(renderer.program64x64);
    gl.uniform1f(gl.getUniformLocation(renderer.program64x64, "iBrightness"), renderer.iBrightness);
    gl.uniform1f(gl.getUniformLocation(renderer.program64x64, "iGamma"), renderer.iGamma);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, renderer.target32x32.texture);
    gl.uniform1i(gl.getUniformLocation(renderer.program64x64, "uSrc"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, renderer.w05);
    gl.uniform1i(gl.getUniformLocation(renderer.program64x64, "uWeights"), 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

}


// load model
function loadModel() {
    let modelSelector = document.getElementById("model-select");
    let path = modelSelector.value;
    loadWeightTexture(path + "/w01_512_32.bin", "w01");
    loadWeightTexture(path + "/w02_32_64_4_4.bin", "w02");
    loadWeightTexture(path + "/b02_64.bin", "b02");
    loadWeightTexture(path + "/w03_64_32_4_4.bin", "w03");
    loadWeightTexture(path + "/b03_32.bin", "b03");
    loadWeightTexture(path + "/w04_32_16_4_4.bin", "w04");
    loadWeightTexture(path + "/b04_16.bin", "b04");
    loadWeightTexture(path + "/w05_16_3_4_4.bin", "w05");
}

// load renderer/interaction
window.onload = function () {
    // get context
    function onError(error) {
        console.error(error);
        let errorMessageContainer = document.getElementById("error-message");
        errorMessageContainer.style.display = "block";
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

    // weights/textures
    loadModel();
    document.getElementById("model-select").addEventListener("input", loadModel);

    // GLSL source
    console.time("load glsl code");
    renderer.vsSource = "#version 300 es\nin vec4 vertexPosition;" +
        "void main(){gl_Position=vertexPosition;}";
    renderer.fs4x4 = loadShaderSource("fs-4x4.glsl");
    renderer.fs8x8 = loadShaderSource("fs-8x8.glsl");
    renderer.fs16x16 = loadShaderSource("fs-16x16.glsl");
    renderer.fs32x32 = loadShaderSource("fs-32x32.glsl");
    renderer.fs64x64 = loadShaderSource("fs-64x64.glsl");
    console.timeEnd("load glsl code");

    // shaders
    console.time("compile shader");
    try {
        renderer.program4x4 = createShaderProgram(renderer.vsSource, renderer.fs4x4);
        renderer.program8x8 = createShaderProgram(renderer.vsSource, renderer.fs8x8);
        renderer.program16x16 = createShaderProgram(renderer.vsSource, renderer.fs16x16);
        renderer.program32x32 = createShaderProgram(renderer.vsSource, renderer.fs32x32);
        renderer.program64x64 = createShaderProgram(renderer.vsSource, renderer.fs64x64);
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
    renderer.target4x4 = createRenderTarget(renderer.width, renderer.height);
    renderer.target8x8 = createRenderTarget(renderer.width, renderer.height);
    renderer.target16x16 = createRenderTarget(renderer.width, renderer.height);
    renderer.target32x32 = createRenderTarget(renderer.width, renderer.height);
    renderer.target64x64 = createRenderTarget(renderer.width, renderer.height);

    // rendering
    function render() {
        drawScene();
        renderer.iFrame += 1;
        setTimeout(function () { requestAnimationFrame(render); }, 100);
    }
    requestAnimationFrame(render);

    // interactions
    window.addEventListener("resize", function (event) {
    });

}
