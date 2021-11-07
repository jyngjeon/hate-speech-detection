const LABEL_HATE = "hate"
const LABEL_OFFENSIVE = "offensive"
const LABEL_NONE = "none"

function getText() {
    return document.body.innerText
}

function parseString(innerText) {
    // 문장 별로 나누기
    const seperatorRegex = /[\.|\n|\!|\?|;]/
    let rawSentences = innerText.split(seperatorRegex)
    
    // 의미없는 문장 지우기
    const meaningLessFilterRegex = /\S\D/
    const meaningfulSentences = rawSentences.filter((rawSentence) => meaningLessFilterRegex.test(rawSentence))

    const trimmedSentences = meaningfulSentences.map((meaningfulSentence) => meaningfulSentence.trim())

    // 한 단어짜리 문장 지우기
    const longSentences = trimmedSentences.filter((trimmedSentence) => /\s/.test(trimmedSentence))
    return longSentences
}

function generateJson(inputArray) {
    var jsonObj = new Object()
    jsonObj.inputs = inputArray
    return JSON.stringify(jsonObj)
}

async function apiResponse(bodyString) {
    const url = "https://gkkhnyeypg.execute-api.ap-northeast-2.amazonaws.com/production/hate-speech"
    const response = await fetch(url, {
        method: "POST",
        mode: "cors",
        headers: {
            "Content-type": "application/json"
        },
        body: bodyString
    })
    return await response.json()
}

function findAndBlock(target) {
    const xpath = "//contains(text(), " + "'" + target + "')"
    const matchingElement = document.evaluate(xpath, document, null, XPathResult.ANY_TYPE, null)

    var blockedElement = matchingElement.iterateNext()
    while (blockedElement) {
        blockedElement.style.display = "none"
    }
}

function blockHate(parsedText, response, level) {
    const responseObject = response
    if (level == 1) {
        // Hate 막기
        for (let i = 0; i < responseObject.length; i++) {
            const targetSentece = parsedText[i]
            const label = response[i].label
            const score = response[i].score
            
            if (label == LABEL_HATE) {
                alert("판단")
                findAndBlock(targetSentece)
            }
        }
    } else if (level == 2) {
        // Hate와 Offensive 막기
        for (let i = 0; i < responseObject.length; i++) {
            const targetSentece = parsedText[i]
            const label = response[i].label
            const score = response[i].score
            
            if ((label == LABEL_HATE) || (label == LABEL_OFFENSIVE)) {
                findAndBlock(targetSentece)
            }
        }
    }
}

console.log("된다고 된다니까!")
const text = getText()
const parsedText = parseString(text)
const bodyJson = generateJson(parsedText)

apiResponse(bodyJson).then((res) => {
    blockHate(parsedText, res, 1)
})