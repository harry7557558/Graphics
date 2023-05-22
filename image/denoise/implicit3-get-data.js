// https://harry7557558.github.io/spirula/implicit3/index.html

"use strict";

let functions = builtinFunctions;
let step = document.getElementById("select-step");

var data = [];
var fi = 0;
var imgs = [];

function renderFirst() {
    if (fi >= Math.min(functions.length, 100)) {
        console.log(JSON.stringify(data));
        return;
    }
    var name = functions[fi][0];
    var fun = functions[fi][1];
    fun = fun.replaceAll("&#32;", ' ');
    console.log(name);
    document.getElementById("equation-input").value = fun;
    imgs = [];
    step.value = "0.04";
    updateFunctionInput();
    state.renderNeeded = true;
    setTimeout(function() {
        imgs.push(canvas.toDataURL('img/png'));
        renderSecond();
    }, 1000);
}

function renderSecond() {
    step.value = "0.001";
    updateFunctionInput();
    state.renderNeeded = true;
    setTimeout(function() {
        imgs.push(canvas.toDataURL('img/png'));
        data.push(imgs);
        fi++;
        renderFirst();
    }, 1000);
}

renderFirst();
