chrome.runtime.sendMessage({ msg: "hello" })
chrome.runtime.onInstalled.addListener(() => {console.log("test")})
console.log("test")