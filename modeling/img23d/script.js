
function initDragDrop() {
    let dropArea = document.getElementById("control");
    window.addEventListener("dragover", function(e) {
        e.preventDefault();
        dropArea.style.backgroundColor = "white";
    });
    window.addEventListener("dragleave", function(e) {
        e.preventDefault();
        dropArea.style.backgroundColor = null;
    });
    window.addEventListener("drop", function(e) {
        e.preventDefault();
        dropArea.style.backgroundColor = null;
        let file = e.dataTransfer.files[0];
        if (file && (
            file.type == "image/png" ||
            file.type == "image/jpeg" ||
            file.type == "image/svg+xml"
            )) {
            let reader = new FileReader();
            reader.onload = function(event) {
                let img = new Image();
                img.onload = function() {
                    let canvas = document.createElement("canvas");
                    canvas.width = img.width;
                    canvas.height = img.height;
                    console.log(img.width, img.height);
                    let ctx = canvas.getContext("2d");
                    ctx.drawImage(img, 0, 0, img.width, img.height);
                    let imageData = ctx.getImageData(0, 0, img.width, img.height);
                    // let pixelData = new Uint8Array(imageData.data);
                    let pixelData = imageData.data;
                    var heapSpace = Module._malloc(pixelData.length);
                    Module.HEAP8.set(pixelData, heapSpace); // bool has 1 byte        
                    Module.ccall('updateImage',
                        null, ['number', 'number', 'number'],
                        [img.width, img.height, heapSpace]);
                }
                img.src = event.target.result;
            }
            reader.readAsDataURL(file);
        }
    })
}

function initInteraction() {
    let canvas = document.getElementById("emscripten-canvas");
    function onresize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        canvas.style.width = canvas.width + "px";
        canvas.style.height = canvas.height + "px";
        Module.ccall('resizeWindow',
            null,
            ['number', 'number'],
            [canvas.width, canvas.height]);
    };
    window.addEventListener("resize", onresize);
    onresize();
}

function onReady() {
    initInteraction();
    initDragDrop();
}

