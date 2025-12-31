const API_BASE = "/api";

const modeNoInput = document.getElementById("mode-no-input");
const modeWithInput = document.getElementById("mode-with-input");

const latentSection = document.getElementById("latent-section");
const inputSection = document.getElementById("input-section");
const strengthWrapper = document.getElementById("strength-wrapper");

const steps = document.getElementById("steps");
const stepsValue = document.getElementById("stepsValue");

const strength = document.getElementById("strength");
const strengthValue = document.getElementById("strengthValue");

const generateBtn = document.getElementById("generate");
const outputImage = document.getElementById("outputImage");

const imageInput = document.getElementById("imageInput");
const inputPreview = document.getElementById("inputPreview");

const seedToggle = document.getElementById("seedToggle");
const seedInput = document.getElementById("seedValue");

let seedEnabled = false;

let currentMode = "no-input";

/* --------- UI --------- */
steps.oninput = () => stepsValue.textContent = steps.value;
strength.oninput = () => strengthValue.textContent = strength.value;

modeNoInput.onclick = () => switchMode("no-input");
modeWithInput.onclick = () => switchMode("with-input");

function switchMode(mode) {
  currentMode = mode;

  modeNoInput.classList.toggle("active", mode === "no-input");
  modeWithInput.classList.toggle("active", mode === "with-input");

  inputSection.classList.toggle("hidden", mode === "no-input");
  strengthWrapper.classList.toggle("hidden", mode === "no-input");

  outputImage.classList.add("hidden");
}

imageInput.onchange = () => {
  const file = imageInput.files[0];
  if (!file) return;
  inputPreview.src = URL.createObjectURL(file);
  inputPreview.classList.remove("hidden");
};


function mockApi(endpoint, payload) {
  console.log("API CALL:", endpoint, payload);

  return new Promise(resolve => {
    setTimeout(() => {
      resolve({
        image: "https://picsum.photos/512?random=" + Math.random(),
        latent: Array.from({ length: 16 }, () => Math.random().toFixed(3))
      });
    }, 1000);
  });
}

seedToggle.onclick = () => {
  seedEnabled = !seedEnabled;

  seedInput.disabled = !seedEnabled;
  seedToggle.classList.toggle("active", seedEnabled);
  seedToggle.classList.toggle("inactive", !seedEnabled);

  seedToggle.textContent = seedEnabled ? "🎯 Fixed seed" : "🎲 Random";
}


/* --------- ACTIONS --------- */
generateBtn.onclick = async () => {
  outputImage.classList.add("hidden");
  spinner.classList.remove("hidden");

  if(currentMode == "with-input"){
    generate_img2img();
  }
  else{
    generate_random();
  }
};

function generate_random(){
  const stepsValue = steps.value;
  const seed = seedEnabled ? parseInt(seedInput.value) : 0;

  fetch(`/api/generate?steps=${stepsValue}&seed=${seed}`)
    .then(res => {
      if (!res.ok) throw new Error("API error");
      return res.json();
    })
    .then(data => {
      outputImage.src = data.image;
      outputImage.classList.remove("hidden");
    })
    .catch(err => {
      console.error("Generation failed:", err);
      alert("Generation failed");
    })
    .finally(() => {
      spinner.classList.add("hidden"); 
    });
  }

function generate_img2img(){
  const formData = new FormData();
  formData.append("image", imageInput.files[0]);
  formData.append("steps", steps.value);
  formData.append("strength", strength.value);
  formData.append("seed", seedEnabled ? parseInt(seedInput.value) : 0);

  spinner.classList.remove("hidden"); // pokaż loader

  fetch("/api/img2img", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      outputImage.src = data.image;
      outputImage.classList.remove("hidden");
    })
    .catch(err => {
      console.error("Img2Img failed:", err);
      alert("Img2Img generation failed");
    })
    .finally(() => {
      spinner.classList.add("hidden"); // ukryj loader
    });
  }