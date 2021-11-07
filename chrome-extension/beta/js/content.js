const LABEL_HATE = "hate"
const LABEL_OFFENSIVE = "offensive"
const LABEL_NONE = "none"

function isParagraph(node) {
    if (!node.hasChildNodes()) return false
    if (node.childElementCount == 0) return true

    var children = node.childNodes
    for (let i = 0; i < children.length; i++) {
        if ((children instanceof Element) && (children.tagName != "BR")) return false
    }

    return true
}

class HateBlocker {
    constructor(querySelectorString, level) {
        const allPossibleNodes = document.querySelectorAll(querySelectorString)
        var textNodes = new Array()
        var texts = ""
        
        for (let i = 0; i < allPossibleNodes.length; i++) {
            if (isParagraph(allPossibleNodes[i])) {
                textNodes.push(allPossibleNodes[i])
                texts += allPossibleNodes[i].innerText
            }
        }

        this.nodes = textNodes
        this.text = texts
        this.level = level
    }

    parseText() {
        // 문장 별로 나누기
        const seperatorRegex = /[\.|\n|\!|\?|;]/
        let rawSentences = this.text.split(seperatorRegex)

        // 의미없는 문장 지우기
        const meaningLessFilterRegex = /\S\D/
        const meaningfulSentences = rawSentences.filter((rawSentence) => meaningLessFilterRegex.test(rawSentence))

        const trimmedSentences = meaningfulSentences.map((meaningfulSentence) => meaningfulSentence.trim())

        // 한 단어짜리 문장 지우기
        const longSentences = trimmedSentences.filter((trimmedSentence) => /\s/.test(trimmedSentence))
        this.text = longSentences
    }
    
    generateJson() {
        var jsonObj = new Object()
        jsonObj.inputs = this.text
        return JSON.stringify(jsonObj)
    }

    static async apiResponse(bodyString) {
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

    blockHate(response) {
        const blockIndex = new Array()

        for (let i = 0; i < response.length; i++) {
            if (this.level == 1) {
                if (response[i].label == LABEL_HATE) blockIndex.push(i)
            } else if (this.level == 2) {
                if (response[i].label != LABEL_NONE) blockIndex.push(i)
            }
        }
        
        while (blockIndex.length > 0) {
            this.nodes[blockIndex[0]].style.color = "transparent"
            blockIndex.shift()
        }
    }
}

console.log("된다고 된다니까!")
const blocker = new HateBlocker('h1, h2, h3, h4, h5, p, li, td, div, caption, span, a', 1)
blocker.parseText()
HateBlocker.apiResponse(blocker.generateJson()).then((res) => {
    console.log("Response was loaded")
    blocker.blockHate(res)
})