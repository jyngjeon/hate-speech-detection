var slider = document.getElementById("mainSlider");
var output = document.getElementById("output");
switch (slider.value) {
  case '0': 
    output.innerHTML = "차단 안함"
    break
  case '1': 
    output.innerHTML = "혐오 표현 차단"
    break
  case '2': 
    output.innerHTML = "공격적 표현부터 차단"
    break
}

// Update the current slider value (each time you drag the slider handle)
slider.oninput = function() {
  output.innerHTML = this.value;
  switch (this.value) {
    case '0': 
      output.innerHTML = "차단 안함"
      break
    case '1': 
      output.innerHTML = "혐오 표현 차단"
      break
    case '2': 
      output.innerHTML = "공격적 표현부터 차단"
      break
  }
} 