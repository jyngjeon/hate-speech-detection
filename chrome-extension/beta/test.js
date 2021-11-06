console.log("Test loaded")
chrome.runtime.onMessage.addListener(() => { document.body.style.background = "red"; console.log("testtesttest")})
console.log("sexsex")
chrome.runtime.onMessage.addListener(function() { document.body.style.background = "red"; console.log("testtesttest")})
