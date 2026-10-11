// Photo select karte hi uska chhota preview dikhao
document.querySelectorAll("input[data-preview]").forEach(function (input) {
  input.addEventListener("change", function () {
    var label = input.closest(".drop");
    var img = label.querySelector("img");
    var file = input.files && input.files[0];
    if (!file) { label.classList.remove("has-image"); img.removeAttribute("src"); return; }
    img.src = URL.createObjectURL(file);
    label.classList.add("has-image");
  });
});

// Form submit hone par "AI is comparing..." loader dikhao
var form = document.getElementById("inspection-form");
if (form) {
  form.addEventListener("submit", function () {
    document.getElementById("overlay").classList.add("show");
  });
}
